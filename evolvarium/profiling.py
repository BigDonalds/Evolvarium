"""Where the time in a move goes.

A move is three calls in `_Runner.play`: the game's observation is encoded into
inputs, the network picks an action from them, and the game applies it. Training
adds a cost around all three, handing genomes to worker processes and results
back. A batched evaluator only speeds up the network, so whether it is worth
building, and how, depends on how large that part is next to the others.

Four measurements:

* Moves. One genome plays a set of games twice, once untouched and once with
  the clock read between the three calls. The clock reads cost a little, so the
  rate comes from the untouched games and the split from the timed ones. Every
  timed game has to end the way its untouched twin did, or it was a different
  game.
* Inside the inputs. cProfile runs over the first few games and splits the
  encoder's time between the functions it calls. cProfile slows code that makes
  many small calls far more than code that makes a few large ones, so only the
  split is kept, never the times.
* The network alone, on states recorded from those games: one state at a time
  as played now, then batches through one genome, then batches with a different
  genome for every row, which is what a batched evaluator would run.
* The pool. A whole population is scored the way training scores it, and every
  worker's time is accounted for: starting up, scoring, between tasks and
  waiting for the round to end. A few of its tasks are also played alone before
  and after, which tells a pool that slows its workers apart from a machine
  that slows down under steady load.
"""

from __future__ import annotations

import cProfile
import dataclasses
import itertools
import os
import pickle
import platform
import sysconfig
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from .evolution.evaluation import EvalContext, EvalResult, make_runner, run_task
from .evolution.pool import EvaluationPool
from .nn.activations import resolve
from .nn.network import Architecture, PolicyNetwork

# Moves are grouped by the game's workload in bands this wide. Snake's workload
# is its length, and the space scan grows with the body, so the cost of a move
# is reported per band as well as overall.
BAND_WIDTH = 25

# The timed games keep a copy of the inputs on the first move and every this
# many after, and the network is timed on an even spread of up to
# NETWORK_STATES of them.
SAMPLE_EVERY = 64
NETWORK_STATES = 600
BATCH_SIZES = (1, 8, 64, 150, 600)

INSIDE_GAMES = 10

# Training draws base seeds uniformly below 2**30, so fixed seeds play the same
# kind of games. Round n uses POOL_SEED * n.
POOL_SEED = 1_000_000


# ------------------------------------------------------------------- moves

@dataclass
class WorkloadBand:
    low: int
    high: int
    moves: int
    inputs: float
    network: float
    game: float

    def line(self) -> str:
        per = 1e6 / max(self.moves, 1)
        return (
            f"      {self.low:>3}-{self.high - 1:<3}  {self.moves:>9,}  {self.inputs * per:>8.1f} "
            f"{self.network * per:>8.1f} {self.game * per:>8.1f}"
        )


@dataclass
class MoveProfile:
    games: int
    # Every move decided, one per agent each step, including one that ends the
    # game, which the game's own step count may leave out.
    moves: int
    # Seconds over all games, from game setup to the last move.
    untimed: float
    timed: float
    # Seconds inside each call, summed over the timed games.
    inputs: float
    network: float
    game: float
    bands: list[WorkloadBand]
    # What the bands count, in the game's words.
    workload: str = "workload"

    @property
    def rest(self) -> float:
        """The loop itself and the clock reads."""
        return max(self.timed - self.inputs - self.network - self.game, 0.0)

    @property
    def rate(self) -> float:
        return self.moves / self.untimed

    def us(self, seconds: float) -> float:
        return seconds * 1e6 / self.moves

    def lines(self) -> list[str]:
        out = [
            f"  moves: {self.games} games, {self.moves:,} moves",
            f"    untimed  {self.rate:,.0f} moves a second, {self.us(self.untimed):.1f} us a move",
            f"    timed    {self.moves / self.timed:,.0f} moves a second, the clock reads "
            f"cost {self.timed / self.untimed - 1:+.1%}",
            "",
            f"    {'':<9} {'us a move':>9}  {'share':>6}",
        ]
        for name, seconds in (("inputs", self.inputs), ("network", self.network),
                              ("game", self.game), ("rest", self.rest)):
            out.append(f"    {name:<9} {self.us(seconds):>9.1f}  {seconds / self.timed:>6.1%}")
        out += [
            "",
            f"    {self.workload:<12} {'moves':>8}  {'inputs':>8} {'network':>8} {'game':>8}"
            "  (us a move)",
        ]
        out += [band.line() for band in self.bands if band.moves]
        return out


def _timed_game(runner, genome: np.ndarray, seed: int, sums: list[list[int]],
                samples: list[np.ndarray], counter: list[int]) -> tuple:
    """`_Runner.play` with the clock read between its three calls.

    Exploration is left out: profiles run with it off, as held-out games do.
    The bands grow as the workload reaches them.
    """
    game = runner.game
    game.reset(seed)
    forward = runner.network.load(genome).forward
    agents = game.agents
    observe = game.observe
    encode = game.observation.encode
    decode = game.actions.decode
    step = game.step
    workload = game.workload
    clock = time.perf_counter_ns
    inputs, decide, played, moves = sums
    count = counter[0]

    while not game.over:
        band = workload() // BAND_WIDTH
        while band >= len(moves):
            for column in sums:
                column.append(0)
        actions = {}
        for agent in agents:
            t0 = clock()
            x = encode(observe(agent))
            t1 = clock()
            actions[agent] = decode(forward(x))
            t2 = clock()
            inputs[band] += t1 - t0
            decide[band] += t2 - t1
            moves[band] += 1
            if count % SAMPLE_EVERY == 0:
                samples.append(x)
            count += 1
        t3 = clock()
        step(actions)
        played[band] += clock() - t3

    counter[0] = count
    return game.score, game.steps, game.end


def _outcome(outcome) -> tuple:
    return outcome.score, outcome.steps, outcome.end


def profile_moves(ctx: EvalContext, genome: np.ndarray, seeds: list[int],
                  note: Callable[[str], None] | None = None
                  ) -> tuple[MoveProfile, np.ndarray]:
    """Time every move of one genome's games. Also returns the sampled inputs.

    Each seed is played untouched and then timed straight after, rather than all
    of one kind first, so a machine that slows down partway affects both alike.
    """
    ctx = dataclasses.replace(ctx, exploration_rate=0.0)
    runner = make_runner(ctx)
    sums: list[list[int]] = [[] for _ in range(4)]
    samples: list[np.ndarray] = []
    counter = [0]
    untimed = timed = 0.0

    for number, seed in enumerate(seeds, 1):
        started = time.perf_counter()
        plain = runner.play(genome, seed, False)
        middle = time.perf_counter()
        clocked = _timed_game(runner, genome, seed, sums, samples, counter)
        ended = time.perf_counter()
        if clocked != _outcome(plain):
            raise RuntimeError(
                f"seed {seed}: the timed game ended {clocked}, "
                f"the untouched one {_outcome(plain)}"
            )
        untimed += middle - started
        timed += ended - middle
        if note is not None and number % 10 == 0:
            note(f"moves: {number} of {len(seeds)} games")

    inputs, decide, played, moves = sums
    profile = MoveProfile(
        games=len(seeds),
        moves=sum(moves),
        untimed=untimed,
        timed=timed,
        inputs=sum(inputs) / 1e9,
        network=sum(decide) / 1e9,
        game=sum(played) / 1e9,
        bands=[
            WorkloadBand(
                low=i * BAND_WIDTH, high=(i + 1) * BAND_WIDTH, moves=moves[i],
                inputs=inputs[i] / 1e9, network=decide[i] / 1e9, game=played[i] / 1e9,
            )
            for i in range(len(moves))
        ],
        workload=runner.game.workload_label,
    )
    if samples:
        picks = np.linspace(0, len(samples) - 1, min(NETWORK_STATES, len(samples)))
        states = np.stack([samples[int(i)] for i in picks])
    else:
        states = np.zeros((0, runner.game.observation.size), dtype=np.float32)
    return profile, states


# -------------------------------------------------------- inside the inputs

@dataclass
class InputPart:
    name: str
    share: float


def _label(key: tuple) -> str:
    filename, _, name = key
    if filename == "~":
        return name
    return f"{Path(filename).name} {name}"


def split_inputs(ctx: EvalContext, genome: np.ndarray, seeds: list[int]) -> list[InputPart]:
    """Shares of the encoder's time, by the functions it calls directly.

    Generator expressions are looked through: what they call counts as called by
    the encoder, and their own time as the encoder's own.
    """
    ctx = dataclasses.replace(ctx, exploration_rate=0.0)
    runner = make_runner(ctx)
    profiler = cProfile.Profile()
    profiler.enable()
    for seed in seeds:
        runner.play(genome, seed, False)
    profiler.disable()
    profiler.create_stats()
    stats = profiler.stats

    code = runner.game.encoder().__code__
    root = (code.co_filename, code.co_firstlineno, code.co_name)
    if root not in stats:
        raise RuntimeError(f"cProfile saw no call to {code.co_name}")
    total = stats[root][3]
    own = stats[root][2]
    parts: dict[str, float] = {}

    def visit(parent: tuple) -> None:
        nonlocal own
        for key, (_, _, _, _, callers) in stats.items():
            if parent not in callers:
                continue
            _, _, inner, cumulative = callers[parent]
            if key[0] != "~" and key[2].startswith("<"):
                own += inner
                visit(key)
            else:
                parts[_label(key)] = parts.get(_label(key), 0.0) + cumulative

    visit(root)
    parts[f"{code.co_name} itself"] = own
    ranked = sorted(parts.items(), key=lambda item: item[1], reverse=True)
    return [InputPart(name, seconds / total) for name, seconds in ranked]


# ------------------------------------------------------------- the network

@dataclass
class NetworkBatch:
    size: int
    # Microseconds per state.
    same: float
    own: float


@dataclass
class NetworkProfile:
    states: int
    # Genomes in the population the rows were drawn from.
    genomes: int
    # Microseconds per state, one `act` call each, as games are played now.
    single: float
    batches: list[NetworkBatch]
    # Rows of the largest own-genome batch whose action differs from `act` on
    # the same genome and state, and the largest difference in any logit.
    disagreements: int
    largest_difference: float

    def lines(self) -> list[str]:
        out = [
            f"  network alone: {self.states} recorded states, us a state",
            f"    one state at a time, as played   {self.single:>6.2f}",
            f"    {'batch':>5}  {'one genome':>10}  {'own genome':>10}",
        ]
        out += [f"    {b.size:>5}  {b.same:>10.2f}  {b.own:>10.2f}" for b in self.batches]
        largest = self.batches[-1].size if self.batches else 0
        logits = ("the logits matched exactly" if self.largest_difference == 0.0
                  else f"logits differ by up to {self.largest_difference:.1e}")
        out.append(
            f"    batched with own genomes, {self.disagreements} of {largest} rows chose a "
            f"different move; {logits}"
        )
        return out


def stacked_layers(architecture: Architecture, genomes: np.ndarray
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every genome's weights as one array per layer, read as `PolicyNetwork.load` reads them."""
    genomes = np.asarray(genomes, dtype=np.float32)
    sizes = architecture.layer_sizes
    layers = []
    offset = 0
    for fan_in, fan_out in itertools.pairwise(sizes):
        count = fan_in * fan_out
        weight = genomes[:, offset:offset + count].reshape(-1, fan_in, fan_out)
        offset += count
        bias = genomes[:, offset:offset + fan_out]
        offset += fan_out
        layers.append((weight, bias))
    return layers


def forward_each(layers: list[tuple[np.ndarray, np.ndarray]], activations: list,
                 x: np.ndarray) -> np.ndarray:
    """Logits for row i of `x` under genome i, for all rows in one pass."""
    current = x[:, None, :]
    last = len(layers) - 1
    for i, (weight, bias) in enumerate(layers):
        current = current @ weight + bias[:, None, :]
        if i < last:
            current = activations[i](current)
    return current[:, 0, :]


def seconds_per_call(functions: list[Callable[[], object]], repeat: int) -> list[float]:
    """Fastest time per call for each function, timing them in turns.

    On the laptop these figures come from, the same call slows from 9 to 20
    microseconds within a second or two of steady work. Timed one after another,
    the first function would get the fast machine and the rest the slow one.
    Taking turns gives each the same chances, and the fastest round of each is
    kept.
    """
    timers = [timeit.Timer(function) for function in functions]
    numbers = [timer.autorange()[0] for timer in timers]
    best = [float("inf")] * len(timers)
    for _ in range(repeat):
        for i, (timer, number) in enumerate(zip(timers, numbers, strict=True)):
            best[i] = min(best[i], timer.timeit(number) / number)
    return best


def _act_all(network: PolicyNetwork, states: np.ndarray) -> None:
    for state in states:
        network.act(state)


def profile_network(architecture: Architecture, genome: np.ndarray, population: np.ndarray,
                    states: np.ndarray, batch_sizes=BATCH_SIZES, repeat: int = 5
                    ) -> NetworkProfile:
    """Microseconds per state for one genome and for a genome per row.

    Batches larger than the recorded states repeat them, and rows beyond the
    population repeat its genomes. Stacking the weights is left out of the time,
    since a batched evaluator would do it once per generation.

    The widest own-genome batch is also checked against `forward` run on each
    row alone. A batched evaluator that changes even one move fails the golden
    test.
    """
    if len(states) == 0:
        raise ValueError("no recorded states to time the network on")
    network = PolicyNetwork(architecture).load(genome)
    activations = [resolve(name) for name in architecture.hidden_activations]

    functions = [partial(_act_all, network, states)]
    for size in batch_sizes:
        x = np.resize(states, (size, states.shape[1]))
        rows = np.resize(population, (size, population.shape[1]))
        layers = stacked_layers(architecture, rows)
        functions += [partial(network.forward, x), partial(forward_each, layers, activations, x)]
    times = seconds_per_call(functions, repeat)

    single = times[0] / len(states)
    batches = [
        NetworkBatch(size, times[1 + 2 * i] / size * 1e6, times[2 + 2 * i] / size * 1e6)
        for i, size in enumerate(batch_sizes)
    ]

    # x, rows and layers are left at the widest batch.
    batched = forward_each(layers, activations, x)
    check = PolicyNetwork(architecture)
    disagreements, largest_difference = 0, 0.0
    for row, state, logits in zip(rows, x, batched, strict=True):
        alone = check.load(row).forward(state)
        disagreements += int(np.argmax(alone) != np.argmax(logits))
        largest_difference = max(largest_difference, float(np.max(np.abs(alone - logits))))

    return NetworkProfile(
        states=len(states), genomes=len(population), single=single * 1e6, batches=batches,
        disagreements=disagreements, largest_difference=largest_difference,
    )


# ---------------------------------------------------------------- the pool

def _timed_task(task: tuple) -> tuple[EvalResult, int, float, float]:
    """`run_task` with the worker's id and the times it started and ended.

    The performance counter is one clock for every process on the machine, so a
    worker's times compare with the parent's. Wall-clock time would as well, but
    on Windows before Python 3.13 it moves in steps of about 15.6 ms, longer than
    a whole task on a small board.
    """
    started = time.perf_counter()
    result = run_task(task)
    return result, os.getpid(), started, time.perf_counter()


@dataclass
class PoolRound:
    tasks: int
    moves: int
    wall: float
    # Worker-seconds, summing to workers * wall: before a worker's first task,
    # inside tasks, between them, and after its last until the round ended.
    starting: float
    scoring: float
    between: float
    end_wait: float
    result_bytes: float


def account(stamps: list[tuple[int, float, float]], workers: int, started: float,
            finished: float) -> dict[str, float]:
    """Split every worker's time in a round from (worker, start, end) stamps.

    A worker that never got a task spent the whole round starting.
    """
    by_worker: dict[int, list[tuple[float, float]]] = {}
    for worker, start, end in stamps:
        by_worker.setdefault(worker, []).append((start, end))
    wall = finished - started
    parts = {"starting": 0.0, "scoring": 0.0, "between": 0.0, "end_wait": 0.0}
    for spans in by_worker.values():
        spans.sort()
        parts["starting"] += spans[0][0] - started
        parts["scoring"] += sum(end - start for start, end in spans)
        parts["between"] += sum(b[0] - a[1] for a, b in itertools.pairwise(spans))
        parts["end_wait"] += finished - spans[-1][1]
    parts["starting"] += max(workers - len(by_worker), 0) * wall
    return parts


@dataclass
class PoolProfile:
    workers: int
    genomes: int
    trials: int
    rounds: list[PoolRound]
    # The first tasks of the last round, played one at a time in this process
    # before the pool starts and after it closes, against their time in it.
    # Seconds summed over the tasks.
    contention_tasks: int = 0
    contention_moves: int = 0
    before: float = 0.0
    in_pool: float = 0.0
    after: float = 0.0

    def lines(self) -> list[str]:
        games = f"{self.trials} game{'s' if self.trials != 1 else ''}"
        out = [
            f"  pool: {self.genomes} genomes, {games} each, {self.workers} workers, "
            "shares of worker time",
            f"    {'round':>5} {'wall s':>7} {'moves/s':>9}  {'starting':>8} {'scoring':>8} "
            f"{'between':>8} {'end wait':>8}  {'result KiB':>10}",
        ]
        for number, r in enumerate(self.rounds, 1):
            total = self.workers * r.wall
            out.append(
                f"    {number:>5} {r.wall:>7.1f} {r.moves / r.wall:>9,.0f}  "
                f"{r.starting / total:>8.1%} {r.scoring / total:>8.1%} "
                f"{r.between / total:>8.1%} {r.end_wait / total:>8.1%}  "
                f"{r.result_bytes / 1024:>10.1f}"
            )
        moves = self.contention_moves
        if self.contention_tasks and moves:
            last = self.rounds[-1]
            out += [
                f"    the first {self.contention_tasks} tasks of the last round, {moves:,} moves:",
                f"      one at a time before the pool  {self.before:>7.1f} s "
                f"{moves / self.before:>9,.0f} moves/s",
                f"      side by side in the pool       {self.in_pool:>7.1f} s "
                f"{moves / self.in_pool:>9,.0f} moves/s a worker",
                f"      one at a time after the pool   {self.after:>7.1f} s "
                f"{moves / self.after:>9,.0f} moves/s",
                f"    the last round ran {last.moves / last.wall / (moves / self.after):.2f} times "
                "as fast as one process after it",
            ]
        return out


def _play_alone(ctx: EvalContext, tasks: list[tuple]) -> tuple[float, list[EvalResult]]:
    runner = make_runner(ctx)
    results = []
    started = time.perf_counter()
    for task in tasks:
        results.append(runner.score(*task))
    return time.perf_counter() - started, results


def profile_pool(ctx: EvalContext, genomes: np.ndarray, trials: int, workers: int,
                 rounds: int, contention_tasks: int | None = None,
                 note: Callable[[str], None] | None = None) -> PoolProfile:
    """Score `genomes` `rounds` times on a pool, as training does.

    The first round's clock starts before the pool exists, so its starting share
    includes launching the workers. Later rounds show a warm pool.

    The first tasks of the last round start together and run side by side. They
    are also played one at a time in this process, before the pool starts and
    after it closes. Before against after shows how far the machine itself
    slowed over the rounds; the pool against those shows what running side by
    side cost each task. All three have to score the same.
    """
    if workers < 2 or rounds < 1:
        raise ValueError("the pool needs at least two workers and one round")
    count = min(contention_tasks if contention_tasks is not None else workers, len(genomes))
    replayed = [(genome, POOL_SEED * rounds, trials, True) for genome in genomes[:count]]
    before, first = _play_alone(ctx, replayed)

    played: list[PoolRound] = []
    started = time.perf_counter()
    with EvaluationPool(ctx, workers) as pool:
        for number in range(1, rounds + 1):
            if number > 1:
                started = time.perf_counter()
            tasks = [(genome, POOL_SEED * number, trials, True) for genome in genomes]
            stamped = pool.map(tasks, run=_timed_task)
            finished = time.perf_counter()
            parts = account([(pid, a, b) for _, pid, a, b in stamped], workers, started,
                            finished)
            results = [result for result, *_ in stamped]
            played.append(PoolRound(
                tasks=len(tasks),
                moves=sum(round(r.steps * r.episodes) for r in results),
                wall=finished - started,
                result_bytes=float(np.mean([len(pickle.dumps(r)) for r in results])),
                **parts,
            ))
            if note is not None:
                note(f"pool: round {number} of {rounds}, {finished - started:.0f} s")

    after, second = _play_alone(ctx, replayed)
    pooled = [result for result, *_ in stamped[:count]]
    for runs in zip(first, pooled, second, strict=True):
        if len({(r.fitness, r.steps) for r in runs}) > 1:
            raise RuntimeError("a task scored differently in the pool and alone")

    return PoolProfile(
        workers, len(genomes), trials, played,
        contention_tasks=count,
        contention_moves=sum(round(r.steps * r.episodes) for r in first),
        before=before,
        in_pool=sum(b - a for _, _, a, b in stamped[:count]),
        after=after,
    )


# ------------------------------------------------------------------ report

@dataclass
class Profile:
    source: str
    system: str
    moves: MoveProfile
    inputs: list[InputPart]
    inside_games: int
    network: NetworkProfile
    pool: PoolProfile | None = None
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        m = self.moves
        out = [f"  {self.source}", f"  {self.system}", ""]
        out += m.lines()
        out += ["", f"  inside the inputs: {self.inside_games} games under cProfile, "
                    "shares of the encoder's time"]
        shown = [p for p in self.inputs if p.share >= 0.005]
        width = max((len(p.name) for p in shown), default=0)
        out += [f"    {p.name:<{width}}  {p.share:>6.1%}" for p in shown]
        rest = sum(p.share for p in self.inputs if p.share < 0.005)
        if rest:
            out.append(f"    {'everything else':<{width}}  {rest:>6.1%}")

        out += [""] + self.network.lines()
        # Training would batch one row per genome, so the population's own size
        # is the batch that counts.
        batches = self.network.batches
        batch = next((b for b in batches if b.size == self.network.genomes),
                     batches[-1] if batches else None)
        if batch is not None:
            per_move = m.us(m.timed)
            saved = m.us(m.network) - batch.own
            out.append(
                f"    a move at {per_move:.1f} us would drop to about {per_move - saved:.1f} us "
                f"with the network batched {batch.size} wide, {saved / per_move:.1%} less"
            )

        if self.pool is not None:
            out += [""] + self.pool.lines()
        for text in self.notes:
            out += ["", f"  {text}"]
        return out

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["moves"]["rest"] = self.moves.rest
        return data


def system_line() -> str:
    """The machine and the Python build, which need not match.

    An x86-64 Python runs emulated on ARM64 Windows, and every figure here pays
    for it.
    """
    return (
        f"{platform.system()} {platform.release()} on {platform.machine()}, "
        f"Python {platform.python_version()} built for {sysconfig.get_platform()}, "
        f"NumPy {np.__version__}, {os.cpu_count()} logical cores"
    )


def profile(ctx: EvalContext, genome: np.ndarray, population: np.ndarray, seeds: list[int],
            *, source: str, workers: int, rounds: int, trials: int,
            inside_games: int = INSIDE_GAMES, batch_sizes=BATCH_SIZES, repeat: int = 5,
            contention_tasks: int | None = None,
            note: Callable[[str], None] | None = None) -> Profile:
    """All four measurements for one genome's games and one population's pool."""
    moves, states = profile_moves(ctx, genome, seeds, note)
    if note is not None:
        note("inside the inputs")
    inside = seeds[:inside_games]
    inputs = split_inputs(ctx, genome, inside)
    if note is not None:
        note("network")
    architecture = Architecture.from_dict(ctx.architecture)
    network = profile_network(architecture, genome, population, states, batch_sizes, repeat)

    pool = None
    notes = []
    if rounds > 0 and workers > 1:
        pool = profile_pool(ctx, population, trials, workers, rounds, contention_tasks, note)
    elif rounds > 0:
        notes.append("pool skipped: it needs at least two workers")
    return Profile(
        source=source, system=system_line(), moves=moves, inputs=inputs,
        inside_games=len(inside), network=network, pool=pool, notes=notes,
    )
