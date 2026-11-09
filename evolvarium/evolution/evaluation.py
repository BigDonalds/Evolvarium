"""Scoring a genome, in this process or another one.

Evaluation is the only expensive part of a generation and genomes never interact
during it, so it parallelises cleanly. The whole design here exists to make the
parallel path produce *identical* results to the serial one, because a speedup
that quietly changes selection is worse than no speedup.

Three rules keep them identical:

* Seeds are decided by the parent before any work is dispatched, and passed in.
  If a worker chose its own, genomes would be scored on different episodes and
  selection would reward luck instead of policy.
* No worker touches the population RNG. Exploration, when enabled, draws from a
  generator seeded off the episode seed, so it is a function of the task rather
  than of the order tasks happen to run in.
* Workers return data, never objects the parent mutates. The parent applies
  results in the original order after every task completes.

Replays are packed inside the worker as the episode plays. A recorded episode
can run to thousands of frames; packed it is a few kilobytes, and that
difference is the difference between a speedup and a serialisation bottleneck.

A game's code runs under the limits in `LimitsConfig`. The clock is read after
every step, and anything the game raises comes back as a `GameFault` naming the
call, the step and the seed. In this process a step that never returns cannot be
stopped; in a worker the pool sees it go quiet and ends it.
"""

from __future__ import annotations

import math
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import games
from ..config import GridConfig, LimitsConfig
from ..contract import Outcome
from ..fixes import Fixes, with_fixes
from ..grid import with_grid
from ..nn.network import Architecture, PolicyNetwork
from ..replays import Recorder


class GameFault(Exception):
    """A game's code failed or ran over its time, in words that say where.

    Only the message is kept, so it crosses between processes whatever the
    game raised.
    """


class TimeLimit(GameFault):
    """A step or a whole episode took longer than `LimitsConfig` allows."""

    def __init__(self, message: str, endless: bool = False):
        super().__init__(message)
        # The episode as a whole ran over, rather than one step of it.
        self.endless = endless


@dataclass(frozen=True)
class EvalContext:
    """Everything a worker needs to score a genome. Picklable by construction.

    `game` is a name the registry resolves, never a game object, so a worker
    builds its own copy and nothing about a live episode crosses processes.
    """

    game: str
    settings: Any
    architecture: dict
    exploration_rate: float
    replay_steps: int
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    # What the probe changed about the game, if it changed anything.
    fixes: Fixes | None = None
    # The grid encoder's settings, for a game played through it.
    grid: GridConfig | None = None


@dataclass
class EvalResult:
    """What scoring produces. Plain data, safe to send between processes."""

    fitness: float
    score: float
    best_episode_score: float
    # Steps of the best episode, the one whose replay is kept. `steps` is the
    # mean over every episode.
    best_episode_steps: int
    steps: float
    terms: dict[str, float]
    signals: dict[str, float]
    # How the worst episode ended, which is the informative failure.
    end: str
    episodes: int
    replay: dict | None = None
    # Problems met and got past while scoring, such as an episode that ran over
    # its time once and finished when played again.
    notes: tuple[str, ...] = ()


def beat_interval(limits: LimitsConfig) -> float:
    """Seconds between a worker's reports that it is still playing."""
    return min(1.0, limits.step_seconds / 4)


def _seconds(value: float) -> str:
    return f"{value:.1f} s" if value >= 1 else f"{value * 1000:.0f} ms"


# The calls `play` makes that are not the game's own, for a fault inside one.
_CALLS = {
    "load": "loading the genome into the network",
    "__init__": "starting the replay",
    "encode": "reading {game}.observe's output as inputs",
    "forward": "the network",
    "decode": "reading the network's outputs as actions",
    "explore": "picking a random action",
    "choose": "picking an action",
    "add": "recording {game}.draw's output",
    "pack": "packing the replay",
}

# The wrappers a game is played through, whose frames a fault looks past.
_WRAPPERS = (with_fixes.__code__.co_filename, with_grid.__code__.co_filename)


def game_fault(error: Exception, game: str, seed: int, turn: int, ended: bool) -> GameFault:
    """`error`, raised under a playing function's `try`, in words that say where."""
    # A fixed game, or one played through the grid encoder, is played through a
    # wrapper, and the call to blame is the one it made.
    raw = traceback.extract_tb(error.__traceback__)
    entries = [entry for entry in raw if entry.filename not in _WRAPPERS]
    # The first entry is `play` itself and the next is the call it made.
    if len(entries) > 1:
        call = entries[1].name
        what = _CALLS.get(call, "{game}." + call).format(game=game)
    elif raw[-1].filename == _WRAPPERS[1]:
        # The grid encoder refused what the game's view gave it.
        what = f"reading {game}.view's output as inputs"
    else:
        what = "playing the episode"
    if ended:
        when = f"after the episode with seed {seed} ended"
    elif turn == 0:
        when = f"while starting the episode with seed {seed}"
    else:
        when = f"on step {turn} of the episode with seed {seed}"
    raised = type(error).__name__
    if str(error):
        raised += f": {error}"
    where = f"{Path(entries[-1].filename).name}, line {entries[-1].lineno}"
    return GameFault(f"{what} raised {raised} {when} ({where})")


class _Runner:
    """Holds the game and network so they are built once, not once per genome."""

    def __init__(self, ctx: EvalContext):
        self.ctx = ctx
        game = games.create(ctx.game, ctx.settings)
        # Faults are blamed on the game, not on the wrappers around it.
        self.label = type(game).__name__
        self.game = with_fixes(with_grid(game, ctx.grid), ctx.fixes)
        self.network = PolicyNetwork(Architecture.from_dict(ctx.architecture))
        self.higher = self.game.direction == "higher"
        self.scene = self.game.scene()
        # Called with the seed and step every `beat_interval` seconds of play,
        # so a worker's pool can tell a long episode from a stuck one.
        self.beat: Callable[[int, int], None] | None = None
        self._every = beat_interval(ctx.limits)
        self._next_beat = 0.0

    def play(self, genome: np.ndarray, seed: int, record: bool) -> Outcome:
        game = self.game
        limits = self.ctx.limits
        step_limit = limits.step_seconds
        clock = time.perf_counter
        started = clock()
        deadline = started + limits.episode_seconds
        beat = self.beat
        next_beat = self._next_beat if beat is not None else math.inf
        # Steps finished so far, 0 while the episode starts.
        turn = 0
        ended = False
        try:
            game.reset(seed)
            recorder = None
            if record and self.scene is not None:
                recorder = Recorder(self.scene, self.ctx.replay_steps)
                recorder.add(game.draw())
            forward = self.network.load(genome).forward
            rate = self.ctx.exploration_rate
            # Seeded from the episode, so exploration is reproducible and identical
            # whether this runs here or in a worker process.
            rng = np.random.default_rng(seed ^ 0x5EED) if rate > 0.0 else None

            agents = game.agents
            observe = game.observe
            encode = game.observation.encode
            decode = game.actions.decode
            explore = game.actions.explore
            step = game.step

            last = clock()
            if last - started > step_limit or last >= deadline:
                raise self._late(seed, turn, last - started, last - started)
            while not game.over:
                turn += 1
                actions = {}
                for agent in agents:
                    if rng is not None and rng.random() < rate:
                        actions[agent] = explore(rng)
                    else:
                        actions[agent] = decode(forward(encode(observe(agent))))
                step(actions)
                if recorder is not None:
                    if recorder.full:
                        recorder.truncated = True
                    else:
                        recorder.add(game.draw())
                now = clock()
                if now - last > step_limit or now >= deadline:
                    raise self._late(seed, turn, now - last, now - started)
                if now >= next_beat:
                    beat(seed, turn)
                    next_beat = now + self._every
                last = now

            ended = True
            fitness, terms = game.goal()
            outcome = Outcome(
                score=game.score,
                steps=game.steps,
                end=game.end,
                fitness=fitness,
                terms=terms,
                signals=game.signals(),
                replay=None if recorder is None else recorder.pack(),
            )
        except GameFault:
            raise
        except Exception as error:
            raise game_fault(error, self.label, seed, turn, ended) from error
        self._next_beat = next_beat
        return outcome

    def _late(self, seed: int, turn: int, took: float, elapsed: float) -> TimeLimit:
        limits = self.ctx.limits
        if took > limits.step_seconds:
            what = f"step {turn}" if turn else "starting"
            return TimeLimit(
                f"{what} of the episode with seed {seed} took {_seconds(took)}, more than "
                f"limits.step_seconds allows ({limits.step_seconds:g} s)"
            )
        return TimeLimit(
            f"the episode with seed {seed} was still going after {_seconds(elapsed)} and "
            f"{turn:,} steps, more than limits.episode_seconds allows "
            f"({limits.episode_seconds:g} s)",
            endless=True,
        )

    def _play_in_time(self, genome: np.ndarray, seed: int, record: bool,
                      notes: list[str]) -> Outcome:
        """`play`, given a second go when the first runs over its time.

        A step can run long because the machine was busy or asleep, which says
        nothing about the game. An episode is a function of its seed, so the
        second go plays exactly the same one.
        """
        try:
            return self.play(genome, seed, record)
        except TimeLimit as first:
            try:
                outcome = self.play(genome, seed, record)
            except TimeLimit as second:
                message = f"{second}. It ran over the first time it was played as well"
                if second.endless:
                    message += (". A game whose episodes can go on forever needs a rule "
                                "that ends one after so many steps, or probe.enabled=true "
                                "to work one out")
                raise TimeLimit(message, second.endless) from None
            notes.append(f"{first}; played again, it finished in time")
            return outcome

    def score(self, genome: np.ndarray, base_seed: int, trials: int, record: bool) -> EvalResult:
        # Every genome in a generation meets the same seed sequence, so
        # differences between them reflect policy rather than luck.
        notes: list[str] = []
        outcomes = [
            self._play_in_time(genome, base_seed + trial, record, notes)
            for trial in range(trials)
        ]
        terms: dict[str, float] = {}
        signals: dict[str, float] = {}
        for outcome in outcomes:
            for key, value in outcome.terms.items():
                terms[key] = terms.get(key, 0.0) + value
            for key, value in outcome.signals.items():
                signals[key] = signals.get(key, 0.0) + value

        scores = [o.score for o in outcomes]
        if self.higher:
            best = max(scores)
            worst = min(outcomes, key=lambda o: o.score)
        else:
            best = min(scores)
            worst = max(outcomes, key=lambda o: o.score)
        # The replay kept is the best episode's, the one whose score is shown
        # with it. Of episodes that tie, the earliest.
        shown = next(o for o in outcomes if o.score == best)

        return EvalResult(
            fitness=float(np.mean([o.fitness for o in outcomes])),
            score=float(np.mean(scores)),
            best_episode_score=best,
            best_episode_steps=shown.steps,
            steps=float(np.mean([o.steps for o in outcomes])),
            terms={k: v / trials for k, v in terms.items()},
            signals={k: v / trials for k, v in signals.items()},
            end=worst.end,
            episodes=trials,
            replay=shown.replay,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------- worker side

_runner: _Runner | None = None


def init_worker(ctx: EvalContext, beat: Callable[[int, int], None] | None = None) -> None:
    """Process initialiser: build the game and network once per worker.

    Workers ignore Ctrl+C. The console delivers it to every process attached to
    it, and a worker that died on it would take the generation in progress with
    it, where the trainer wants to finish that generation and save.
    """
    global _runner
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _runner = _Runner(ctx)
    _runner.beat = beat


def run_task(task: tuple[np.ndarray, int, int, bool]) -> EvalResult:
    """Score one genome. Runs in a worker process."""
    genome, base_seed, trials, record = task
    assert _runner is not None, "worker was not initialised"
    return _runner.score(genome, base_seed, trials, record)


def make_runner(ctx: EvalContext) -> _Runner:
    """A runner for in-process use, so the serial path shares this exact code."""
    return _Runner(ctx)
