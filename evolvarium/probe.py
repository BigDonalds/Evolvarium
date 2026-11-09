"""Measuring a game before training it, and working out what to fix.

The probe plays a game with players that have learned nothing: one picking at
random, one repeating a single action, one holding each action for a while and
untrained networks. It needs them only to show what the game's inputs, scores
and episodes look like in play. From that it finds:

* inputs out of scale, which saturate the network's hidden units;
* inputs that never change, and inputs that only repeat another;
* whole-number codes given as numbers, which a network reads as amounts;
* episodes that nothing in the game is sure to end;
* episodes that a seed does not decide;
* rewards too rare to select on, and steps too slow to train with.

What it can fix goes in `Fixes`, and `with_fixes` plays the game with them. The
report says in plain words what was measured, what changed and why. A game with
nothing wrong gets no fixes at all and is played as it was written.

The players are weak on purpose and that is also the probe's limit. An input
that only moves once a player is good looks dead or well scaled here, and two
yes-or-no inputs can agree until a player reaches a situation these never did.
Only exact copies are dropped for that reason, and only inputs wide or narrow
enough to be a problem in weak play are rescaled.

Every episode comes from its own seeds, so a game that keeps its promise about
seeds gets the same report each time unless the time limit cuts the probe short.
"""

from __future__ import annotations

import math
import multiprocessing
import signal
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing import connection

import numpy as np

from . import games, grid
from .config import Config, NetworkConfig
from .contract import Category, Game, Observation
from .evolution.evaluation import GameFault, beat_interval, game_fault
from .evolution.pool import SLACK, RemoteTraceback, WorkerError
from .fixes import FixedGame, Fixes
from .grid import GridGame, with_grid
from .nn.network import Architecture, PolicyNetwork, random_genome

# The first episode seed, in a block of its own that nothing else scores on.
SEED = 600_000

_PLAYERS = ("random", "fixed", "held", "network")

# Of `probe.search_steps`, the share held actions get to move an input that has
# not moved yet. Networks looking for an episode that never ends get all of it.
_HELD_SEARCH = 0.5

# A held action lasts at least the first and less than the second, in steps.
_HOLD = (2, 20)

# An input this far from zero is out of scale. The widest input in a built-in
# game reads about 2.7 in play.
_WIDE = 8.0
# An input that moves by less than this, or sits this many times its own span
# from zero, is out of scale too: the network sees it as a constant.
_TINY = 1e-3
_OFF_CENTRE = 50.0

# Too few inputs seen to say anything about what they do.
_ENOUGH_ROWS = 200
# Rows kept for comparing inputs, and the most values those rows may hold.
_SAMPLE_ROWS = 100_000
_SAMPLE_VALUES = 4_000_000
_CHUNK_ROWS = 2048
# An input needs this many rows away from its usual value to be compared at all.
_MIN_AWAY = 10
_NEAR_ONE = 1.0 - 1e-6
# How far from an exact fit a copy may be, as a share of its span. Rounding in
# 32-bit floats comes on top, since it can be large against a big offset.
_FIT = 1e-5
_ROUNDING = 4.0 * float(np.finfo(np.float32).eps)

# A stall limit is this many times the longest wait for the score to change,
# and never below the floor.
_STALL_TIMES = 10
_STALL_FLOOR = 100
# With a stall limit in place a total limit is only a backstop, so it is loose.
_STEP_TIMES = 10
# A kind of player that has reached the ceiling this many times has shown what it
# can, and those are the costliest episodes there are.
_ENOUGH_CUT = 3

# Plays of the same seed, by the player and the seed's place in the block.
_REPEATS = (("random", 0), ("held", 1), ("network", 2),
            ("random", 3), ("held", 4), ("network", 5))
_REPEAT_STEPS = 1000
_REPEAT_SHARE = 0.25

# A number input is read as a code when it only ever reads whole numbers, and
# between these many of them. Two are a yes or no, which reads fine as a number.
_CODE_VALUES = (3, 8)

# A goal that differs from its usual value in less than this share of episodes
# is too rare to select on, given enough episodes to tell.
_RARE_SHARE = 0.02
_RARE_EPISODES = 20


# ---------------------------------------------------------------- the report

@dataclass(frozen=True)
class Finding:
    """One thing the probe found, what it did about it and the inputs concerned.

    `outcome` is "fixed" when the probe changed the game for it, "left" when it is
    a problem only the game's author can fix, and "noted" when nothing is wrong.
    """

    kind: str
    found: str
    did: str
    inputs: tuple[str, ...] = ()
    outcome: str = "left"


@dataclass(frozen=True)
class InputRow:
    """One of the game's inputs: what it read in play and what became of it."""

    name: str
    # None for an input that never gave a usable number.
    low: float | None
    high: float | None
    # "kept", "scaled", "never changes" or "repeats <name>".
    fate: str
    # What the network read with the fixes on. None when not played that way.
    after_low: float | None = None
    after_high: float | None = None


@dataclass(frozen=True)
class PlayerRow:
    """How one kind of player's episodes went, in the game as it was written."""

    player: str
    episodes: int
    # Those still going at the ceiling, which the rest never reached.
    cut: int
    mean_steps: float
    longest: int
    mean_score: float


@dataclass(frozen=True)
class Report:
    game: str
    episodes: int
    steps: int
    seconds: float
    steps_per_second: float
    ceiling: int
    inputs: tuple[InputRow, ...]
    players: tuple[PlayerRow, ...]
    findings: tuple[Finding, ...]
    fixes: Fixes
    # The probe reached `probe.seconds` and stopped early.
    out_of_time: bool = False

    def kinds(self) -> set[str]:
        return {finding.kind for finding in self.findings}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Report:
        return cls(
            game=data["game"],
            episodes=data["episodes"],
            steps=data["steps"],
            seconds=data["seconds"],
            steps_per_second=data["steps_per_second"],
            ceiling=data["ceiling"],
            inputs=tuple(InputRow(**row) for row in data["inputs"]),
            players=tuple(PlayerRow(**row) for row in data["players"]),
            findings=tuple(
                Finding(row["kind"], row["found"], row["did"], tuple(row["inputs"]),
                        row.get("outcome", "left"))
                for row in data["findings"]
            ),
            fixes=Fixes.from_dict(data["fixes"]) or Fixes(),
            out_of_time=data["out_of_time"],
        )

    def lines(self) -> list[str]:
        """The report as text, for a console."""
        lines = [f"Played {self.episodes:,} episodes of {self.game} in {self.seconds:.1f} s: "
                 f"{self.steps:,} steps, {self.steps_per_second:,.0f} a second."]
        if self.out_of_time:
            lines.append("Ran out of time, so some of this rests on fewer episodes than planned.")

        lines += ["", f"{'player':<10}{'episodes':>9}{'mean steps':>12}{'longest':>9}"
                      f"{'mean score':>12}{'still going':>13}"]
        for row in self.players:
            lines.append(f"{row.player:<10}{row.episodes:>9}{row.mean_steps:>12,.1f}"
                         f"{row.longest:>9,}{row.mean_score:>12,.2f}{row.cut:>13}")

        width = max(len(row.name) for row in self.inputs) + 2
        lines += ["", f"{'input':<{width}}{'low':>11}{'high':>11}{'fixed low':>11}"
                      f"{'fixed high':>11}  what became of it"]
        for row in self.inputs:
            lines.append(f"{row.name:<{width}}{_cell(row.low)}{_cell(row.high)}"
                         f"{_cell(row.after_low)}{_cell(row.after_high)}  {row.fate}")

        lines.append("")
        if not self.findings:
            lines.append("Found nothing to change.")
        for finding in self.findings:
            lines += [f"{finding.kind}: {finding.found}", f"  {finding.did}"]
        return lines

    def notes(self) -> list[str]:
        """The report in a few lines, for a run's log."""
        notes = [f"probe: played {self.episodes:,} episodes in {self.seconds:.1f} s"]
        if self.out_of_time:
            notes.append("probe: ran out of time, so some of this rests on fewer episodes")
        notes += [f"probe: {finding.found} {finding.did}" for finding in self.findings]
        return notes


def _cell(value: float | None) -> str:
    return f"{'':>11}" if value is None else f"{value:>11.4g}"


def _and(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _number(value: float) -> str:
    return f"{value:.4g}"


# ---------------------------------------------------------------- the players

class _Players:
    """Makes the probe's players for one game.

    A player is a closure with the name the engine's fault messages know it by,
    called with an agent and what that agent sees.
    """

    def __init__(self, game: Game, config: NetworkConfig):
        self.actions = game.actions
        self.config = config
        self.architecture = Architecture.from_config(
            config, game.observation.size, game.actions.size
        )
        self.network = PolicyNetwork(self.architecture)

    def make(self, kind: str, rng: np.random.Generator) -> Callable:
        actions = self.actions
        if kind == "random":
            def choose(agent, values):
                return actions.explore(rng)
        elif kind == "fixed":
            chosen: dict = {}

            def choose(agent, values):
                if agent not in chosen:
                    chosen[agent] = actions.decode(rng.uniform(-1.0, 1.0, actions.size))
                return chosen[agent]
        elif kind == "held":
            held: dict = {}

            def choose(agent, values):
                action, left = held.get(agent, (None, 0))
                if left == 0:
                    action, left = actions.explore(rng), int(rng.integers(*_HOLD))
                held[agent] = (action, left - 1)
                return action
        else:
            config = self.config
            forward = self.network.load(random_genome(
                self.architecture, rng, config.weight_init_scale, config.output_init_scale
            )).forward

            def choose(agent, values):
                return actions.decode(forward(values))
        return choose


# ---------------------------------------------------------------- what play shows

class _Inputs:
    """What one pass of play gave each input: ranges, bad values and a fair sample."""

    def __init__(self, size: int):
        self.low = np.full(size, np.inf)
        self.high = np.full(size, -np.inf)
        self.bad = np.zeros(size, dtype=np.int64)
        self.rows = 0
        capacity = min(_SAMPLE_ROWS, _SAMPLE_VALUES // max(size, 1))
        self._sample = np.empty((capacity, size), dtype=np.float32)
        self._chunk = np.empty((_CHUNK_ROWS, size), dtype=np.float32)
        self._filled = 0
        # Of its own, so sampling never moves a player's draws.
        self._rng = np.random.default_rng((SEED, 99))

    def add(self, values: np.ndarray) -> None:
        self._chunk[self._filled] = values
        self._filled += 1
        if self._filled == _CHUNK_ROWS:
            self._flush()

    def _flush(self) -> None:
        rows = self._chunk[:self._filled]
        self._filled = 0
        if not len(rows):
            return
        finite = np.isfinite(rows)
        self.bad += (~finite).sum(axis=0)
        clean = np.where(finite, rows, np.nan)
        self.low = np.fmin(self.low, np.fmin.reduce(clean, axis=0))
        self.high = np.fmax(self.high, np.fmax.reduce(clean, axis=0))

        # Each row has the same chance of ending up in the sample, however long
        # the play and however the episodes fell.
        capacity = len(self._sample)
        fill = min(max(capacity - self.rows, 0), len(rows))
        self._sample[self.rows:self.rows + fill] = rows[:fill]
        if fill < len(rows):
            picks = self._rng.integers(0, self.rows + np.arange(fill, len(rows)) + 1)
            chosen = picks < capacity
            self._sample[picks[chosen]] = rows[fill:][chosen]
        self.rows += len(rows)

    def finish(self) -> _Inputs:
        self._flush()
        return self

    def unmoved(self, columns: list[int]) -> bool:
        """Whether any of `columns` has read one number in everything so far."""
        self._flush()
        return any(self.low[c] == self.high[c] for c in columns)

    @property
    def sample(self) -> np.ndarray:
        return self._sample[:min(self.rows, len(self._sample))]


@dataclass(frozen=True)
class _Played:
    """One episode."""

    player: str
    seed: int
    steps: int
    # Still going at the ceiling, and ended there by the probe.
    cut: bool = False
    # Given up when the probe's time ran out, so it says nothing about length.
    abandoned: bool = False
    # A cut episode whose results could not be read.
    unreadable: bool = False
    score: float | None = None
    goal: float | None = None
    # The longest wait for the score to change. None if it never did.
    gap: int | None = None


@dataclass
class _Trace:
    """What one play of a seed showed at each step, for comparing with another."""

    seen: list[np.ndarray]
    scores: list[float]
    # By a rule of the game, rather than the probe's limit on these plays.
    ended: bool
    # The probe's time ran out during the play.
    abandoned: bool


@dataclass(frozen=True)
class _Split:
    seed: int
    player: str
    step: int
    what: str
    fresh: bool


def _split(first: _Trace, second: _Trace, names: tuple[str, ...]) -> tuple[int, str] | None:
    """The first step two plays differ at, and what differed there."""
    shared = min(len(first.seen), len(second.seen))
    size = len(names)
    for step in range(shared):
        a, b = first.seen[step], second.seen[step]
        if not np.array_equal(a, b, equal_nan=True):
            column = int(np.flatnonzero((a != b) & ~(np.isnan(a) & np.isnan(b)))[0])
            return step, f"input {names[column % size]}"
        if step < min(len(first.scores), len(second.scores)):
            x, y = first.scores[step], second.scores[step]
            if x != y and (x == x or y == y):
                return step + 1, "the score"
    if first.abandoned or second.abandoned:
        return None
    if len(first.seen) != len(second.seen) or first.ended != second.ended:
        return shared, "when the episode ended"
    return None


def _number_columns(observation: Observation) -> list[int]:
    """The columns of inputs that are numbers.

    A category's columns are left out of every check. A value weak play never
    saw proves nothing, and a two-way category's columns are opposites by design.
    """
    columns: list[int] = []
    at = 0
    for spec in observation.inputs:
        if not isinstance(spec, Category):
            columns.extend(range(at, at + spec.size))
        at += spec.size
    return columns


def _straight_lines(sample: np.ndarray, columns: list[int]) -> list[tuple[int, int, bool]]:
    """Pairs of columns where one is an exact straight line of the other.

    Each comes as (earlier, later, whether later is a copy). A straight line goes
    through any two points, so it fits every pair of yes-or-no inputs that flip
    together: being empty and being blocked, say, which only part ways once a
    player gets into a tight spot. A pair where either takes two values counts as
    a copy only when they are also zero together.
    """
    varied: list[int] = []
    two_valued: set[int] = set()
    for column in columns:
        values, counts = np.unique(sample[:, column], return_counts=True)
        if len(sample) - int(counts.max()) >= _MIN_AWAY:
            varied.append(column)
            if len(values) == 2:
                two_valued.add(column)
    if len(varied) < 2:
        return []

    rows = sample[:, varied].astype(np.float64)
    near = np.triu(np.abs(np.corrcoef(rows, rowvar=False)) > _NEAR_ONE, 1)
    found = []
    for a, b in np.argwhere(near):
        x, y = rows[:, a], rows[:, b]
        across = x - x.mean()
        slope = float((across * (y - y.mean())).sum() / (across * across).sum())
        allowed = (_FIT * float(y.max() - y.min())
                   + _ROUNDING * (float(np.abs(y).max()) + abs(slope) * float(np.abs(x).max())))
        line = y - (slope * x + (y.mean() - slope * x.mean()))
        if float(np.abs(line).max()) > allowed:
            continue
        first, second = varied[a], varied[b]
        if first in two_valued or second in two_valued:
            through_zero = float((x * y).sum() / (x * x).sum())
            copy = float(np.abs(y - through_zero * x).max()) <= allowed
        else:
            copy = True
        found.append((first, second, copy))
    return found


def _code(column: np.ndarray) -> tuple[float, ...] | None:
    """The values a column takes, if they look like a code rather than an amount."""
    values = np.unique(column)
    if not _CODE_VALUES[0] <= len(values) <= _CODE_VALUES[1]:
        return None
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return None
    return tuple(float(value) for value in values)


def _out_of_scale(low: float, high: float) -> bool:
    if max(abs(low), abs(high)) > _WIDE:
        return True
    span = high - low
    return span < _TINY or abs(low + high) / 2 / span >= _OFF_CENTRE


def _rescale(low: float, high: float) -> tuple[float, float]:
    """The offset and scale that bring an input's range to about -1 to 1.

    An input that crosses zero keeps its zero, which often means something.
    """
    if low <= 0 <= high:
        return 0.0, 1.0 / max(-low, high)
    return -(low + high) / 2, 2.0 / (high - low)


@dataclass
class _InputFixes:
    """What to do about the inputs, with the reasons in the report's words."""

    keep: tuple[int, ...] | None = None
    offset: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None
    codes: tuple[tuple[float, ...] | None, ...] | None = None
    fates: dict[int, str] | None = None


def _examine_inputs(inputs: _Inputs, earlier: _Inputs, observation: Observation,
                    findings: list[Finding]) -> _InputFixes:
    """What to do about `inputs`, the last pass played under the game's limits.

    An input that moved in the `earlier` pass has shown that it can, so it is
    not called dead whatever the last pass did.
    """
    names = observation.names
    numbers = _number_columns(observation)
    fates: dict[int, str] = {}
    if inputs.rows < _ENOUGH_ROWS:
        findings.append(Finding(
            "few_steps",
            f"Saw only {inputs.rows:,} steps' worth of inputs, too few to judge them by.",
            "Left the inputs as they are.",
        ))
        return _InputFixes(fates=fates)

    bad = [c for c in numbers if inputs.bad[c]]
    if bad:
        parts = [f"{names[c]} ({int(inputs.bad[c]):,} times)" for c in bad]
        findings.append(Finding(
            "bad_values",
            f"Some inputs gave values that are not numbers or are infinite: {_and(parts)}.",
            "Changed nothing about them. A network that reads one gives garbage from then "
            "on, so the game has to stop producing them.",
            tuple(names[c] for c in bad),
        ))
    usable = [c for c in numbers if not inputs.bad[c]]

    dead = [c for c in usable if inputs.low[c] == inputs.high[c]
            and earlier.low[c] == earlier.high[c]]
    moving = [c for c in usable if inputs.low[c] != inputs.high[c]]

    # Which earlier column each copy repeats, followed through to the earliest.
    repeats: dict[int, int] = {}
    opposites: list[tuple[int, int]] = []
    pairs = _straight_lines(inputs.sample, moving)
    for first, second, copy in pairs:
        if copy:
            a, b = repeats.get(first, first), repeats.get(second, second)
            a, b = min(a, b), max(a, b)
            if a != b:
                repeats = {k: (a if v == b else v) for k, v in repeats.items()}
                repeats[b] = a
    for first, second, copy in pairs:
        if not copy and first not in repeats and second not in repeats:
            opposites.append((first, second))

    dropped = set(dead) | set(repeats)
    if len(dropped) == observation.size:
        findings.append(Finding(
            "dead",
            f"No input carries anything of its own in {inputs.rows:,} steps of play.",
            "Kept them all, since a network needs something to read. It can only learn "
            "a fixed routine from these.",
            names,
        ))
        dead, repeats, dropped = [], {}, set()

    if dead:
        kept = observation.size - len(dropped)
        findings.append(Finding(
            "dead",
            f"{_and([names[c] for c in dead])} never changed in {inputs.rows:,} steps of play.",
            f"Dropped {'it' if len(dead) == 1 else 'them'}. "
            f"The network reads {kept} inputs instead of {observation.size}.",
            tuple(names[c] for c in dead),
            outcome="fixed",
        ))
        for c in dead:
            fates[c] = "never changes"
    if repeats:
        kept = observation.size - len(dropped)
        parts = [f"{names[c]} only repeats {names[repeats[c]]}" for c in sorted(repeats)]
        findings.append(Finding(
            "duplicate",
            f"{_and(parts)}, moved or scaled at most.",
            f"Dropped the repeat{'' if len(repeats) == 1 else 's'}. "
            f"The network reads {kept} inputs instead of {observation.size}.",
            tuple(names[c] for c in sorted(repeats)),
            outcome="fixed",
        ))
        for c in repeats:
            fates[c] = f"repeats {names[repeats[c]]}"
    for first, second in opposites:
        findings.append(Finding(
            "opposites",
            f"{names[first]} and {names[second]} flipped together in everything played. "
            "Yes-or-no inputs can do that until a player reaches a spot these players "
            "never did, so that does not make them copies.",
            "Kept both.",
            (names[first], names[second]),
            outcome="noted",
        ))

    coded: dict[int, tuple[float, ...]] = {}
    for c in usable:
        if c not in dropped:
            values = _code(inputs.sample[:, c])
            if values is not None:
                coded[c] = values
                fates[c] = f"a code of {len(values)} values"
    if coded:
        parts = [f"{names[c]} only read {_and([_number(v) for v in values])}"
                 for c, values in coded.items()]
        one = len(coded) == 1
        findings.append(Finding(
            "code",
            f"{_and(parts)}. Whole numbers like {'these' if one else 'each of these'} look "
            "like a code for one of a few things rather than an amount, and read as a number "
            "the last would count as larger than the first.",
            f"Gave {'it' if one else 'each'} one input per value, set to 1 for the value read "
            "and 0 for the rest, as for a category. A value play never showed reads as the "
            "nearest one it did.",
            tuple(names[c] for c in coded),
            outcome="fixed",
        ))

    scaled: dict[int, tuple[float, float]] = {}
    for c in usable:
        if c in dropped or c in coded:
            continue
        low, high = float(inputs.low[c]), float(inputs.high[c])
        if low != high and _out_of_scale(low, high):
            scaled[c] = _rescale(low, high)
            fates[c] = "scaled"
    if scaled:
        parts = [f"{names[c]} reads {_number(float(inputs.low[c]))} to "
                 f"{_number(float(inputs.high[c]))}" for c in scaled]
        findings.append(Finding(
            "unscaled",
            f"{_and(parts)}, which a network's hidden units cannot read well.",
            f"{'Moved and scaled' if any(o for o, _ in scaled.values()) else 'Scaled'} "
            f"{'it' if len(scaled) == 1 else 'each'} by fixed numbers, "
            "to read within -1 to 1.",
            tuple(names[c] for c in scaled),
            outcome="fixed",
        ))

    kept_columns = [c for c in range(observation.size) if c not in dropped]
    fixes = _InputFixes(fates=fates)
    if dropped:
        fixes.keep = tuple(kept_columns)
    if scaled:
        fixes.offset = tuple(float(scaled.get(c, (0.0, 1.0))[0]) for c in kept_columns)
        fixes.scale = tuple(float(scaled.get(c, (0.0, 1.0))[1]) for c in kept_columns)
    if coded:
        fixes.codes = tuple(coded.get(c) for c in kept_columns)
    return fixes


def _stall_cap(played: list[_Played], ceiling: int) -> tuple[int, int] | None:
    """The stall limit the episodes suggest, with the wait it comes from.

    A score that moves every step has nothing to wait for, and a limit past the
    ceiling would never come into play.
    """
    gaps = [episode.gap for episode in played if episode.gap is not None]
    if not gaps:
        return None
    gap = max(gaps)
    cap = max(_STALL_TIMES * gap, _STALL_FLOOR)
    if gap < 2 or cap >= ceiling:
        return None
    return cap, gap


def _summarise(played: list[_Played]) -> tuple[PlayerRow, ...]:
    rows = []
    for kind in _PLAYERS:
        mine = [episode for episode in played
                if episode.player == kind and not episode.abandoned]
        if not mine:
            continue
        scores = [e.score for e in mine if e.score is not None and math.isfinite(e.score)]
        rows.append(PlayerRow(
            player=kind,
            episodes=len(mine),
            cut=sum(e.cut for e in mine),
            mean_steps=float(np.mean([e.steps for e in mine])),
            longest=max(e.steps for e in mine),
            mean_score=float(np.mean(scores)) if scores else 0.0,
        ))
    return tuple(rows)


def _rare_reward(played: list[_Played]) -> Finding | None:
    goals = [round(e.goal, 9) for e in played if e.goal is not None and math.isfinite(e.goal)]
    if len(goals) < _RARE_EPISODES:
        return None
    usual, count = Counter(goals).most_common(1)[0]
    other = len(goals) - count
    if other >= _RARE_SHARE * len(goals):
        return None
    if other:
        found = (f"Only {other} of {len(goals)} episodes finished with a goal other than "
                 f"{_number(usual)}, too rare for selection to work with.")
    else:
        found = (f"All {len(goals)} episodes finished with a goal of {_number(usual)}, "
                 "so nothing tells one player from another.")
    return Finding(
        "rare",
        found,
        "Changed nothing, since the goal is the game's to set. A goal that rewards "
        "steps on the way there gives selection something to climb.",
    )


# ---------------------------------------------------------------- playing

@dataclass
class _Survey:
    played: list[_Played]
    inputs: _Inputs


def _label(game: Game | GridGame | FixedGame) -> str:
    return type(games.unwrap(game)).__name__


class _Probe:
    """One run of the probe. Its passes share a clock and a count of what was played."""

    def __init__(self, config: Config, beat: Callable[[int, int], None] | None):
        self.config = config
        self.ceiling = config.probe.ceiling
        self.beat = beat
        self._every = beat_interval(config.limits)
        self._next_beat = 0.0
        self.started = time.perf_counter()
        self.end = self.started + config.probe.seconds
        self.out_of_time = False
        self.episodes = 0
        self.steps = 0
        self.playing = 0.0
        self.slowest = 0.0

    def fresh(self) -> Game | GridGame:
        """The game as the run will play it, through the grid encoder if it asks for one."""
        game = games.create(self.config.game, self.config.settings)
        return with_grid(game, grid.choose(game, self.config.grid))

    def survey(self, game: Game | FixedGame, share: float, search: bool = False) -> _Survey:
        """Every kind of player plays its episodes, using `share` of the time left.

        They take turns, so if the time runs out it cuts all of them about equally.
        With `search`, play goes on for two things it is slow to show. Networks go
        on if nothing has reached the ceiling. Nearly all of them fail quickly in
        a game that takes any skill, so a game with no limit on its steps can look
        like one with a limit until the odd network lasts: about one in eighty of
        them, on cart-pole. Held actions go on if an input has not moved, since
        they get furthest from where an episode starts. Only they reach a fight in
        the arena, about one episode in twenty, and until one does, being hurt
        looks like an input that never changes.
        """
        now = time.perf_counter()
        deadline = now + max(self.end - now, 0.0) * share
        inputs = _Inputs(game.observation.size)
        players = _Players(game, self.config.network)
        played: list[_Played] = []
        cuts: Counter[str] = Counter()

        def turn(kind: str, n: int) -> _Played | None:
            if time.perf_counter() >= deadline:
                self.out_of_time = True
                return None
            rng = np.random.default_rng((SEED, _PLAYERS.index(kind), n))
            episode = self._play(game, players.make(kind, rng), kind, SEED + n, inputs, deadline)
            played.append(episode)
            cuts[kind] += episode.cut
            if episode.abandoned:
                self.out_of_time = True
            else:
                self.episodes += 1
            return episode

        episodes = self.config.probe.episodes
        for n in range(episodes):
            for kind in _PLAYERS:
                if cuts[kind] < _ENOUGH_CUT and turn(kind, n) is None:
                    return _Survey(played, inputs.finish())
        if search and not sum(cuts.values()):
            spent = 0
            n = episodes
            while spent < self.config.probe.search_steps:
                episode = turn("network", n)
                if episode is None or episode.cut:
                    break
                spent += episode.steps
                n += 1
        if search:
            spent = 0
            n = episodes
            numbers = _number_columns(game.observation)
            while (spent < self.config.probe.search_steps * _HELD_SEARCH
                   and cuts["held"] < _ENOUGH_CUT and inputs.unmoved(numbers)):
                episode = turn("held", n)
                if episode is None:
                    break
                spent += episode.steps
                n += 1
        return _Survey(played, inputs.finish())

    def _play(self, game: Game | FixedGame, choose: Callable, kind: str, seed: int,
              inputs: _Inputs, deadline: float) -> _Played:
        clock = time.perf_counter
        ceiling = self.ceiling
        beat = self.beat
        next_beat = self._next_beat if beat is not None else math.inf
        every = self._every
        started = last = clock()
        slowest = self.slowest
        turn = 0
        ended = cut = abandoned = unreadable = False
        score = goal = gap = None
        readable = True
        quiet = 0
        try:
            game.reset(seed)
            agents = game.agents
            observe = game.observe
            encode = game.observation.encode
            add = inputs.add
            step = game.step
            try:
                standing = game.score
            except Exception:
                readable = False
                standing = None

            while not game.over:
                if turn >= ceiling:
                    cut = True
                    game.stop(False)
                    break
                turn += 1
                actions = {}
                for agent in agents:
                    values = encode(observe(agent))
                    add(values)
                    actions[agent] = choose(agent, values)
                step(actions)
                if readable:
                    try:
                        now_standing = game.score
                    except Exception:
                        readable = False
                        gap = None
                    else:
                        quiet += 1
                        if now_standing != standing:
                            standing = now_standing
                            if gap is None or quiet > gap:
                                gap = quiet
                            quiet = 0
                now = clock()
                if now - last > slowest:
                    slowest = now - last
                last = now
                if now >= next_beat:
                    beat(seed, turn)
                    next_beat = now + every
                if now >= deadline:
                    abandoned = True
                    break

            if not abandoned:
                ended = True
                # All that a run reads at the end, so a fault there shows up now.
                goal, _ = game.goal()
                score = game.score
                _ = game.end
                game.signals()
        except GameFault:
            raise
        except Exception as error:
            if not (cut and ended):
                raise game_fault(error, _label(game), seed, turn, ended) from error
            unreadable = True
        self._next_beat = next_beat
        self.steps += turn
        self.slowest = slowest
        self.playing += clock() - started
        return _Played(
            player=kind, seed=seed, steps=turn, cut=cut, abandoned=abandoned,
            unreadable=unreadable,
            score=None if score is None else float(score),
            goal=None if goal is None else float(goal),
            gap=gap,
        )

    def _trace(self, game: Game | FixedGame, choose: Callable, seed: int,
               deadline: float) -> _Trace:
        clock = time.perf_counter
        beat = self.beat
        next_beat = self._next_beat if beat is not None else math.inf
        seen: list[np.ndarray] = []
        scores: list[float] = []
        turn = 0
        abandoned = False
        try:
            game.reset(seed)
            agents = game.agents
            observe = game.observe
            encode = game.observation.encode
            step = game.step
            readable = True
            while not game.over and turn < _REPEAT_STEPS:
                turn += 1
                actions = {}
                row = []
                for agent in agents:
                    values = encode(observe(agent))
                    row.append(values)
                    actions[agent] = choose(agent, values)
                seen.append(np.concatenate(row))
                step(actions)
                if readable:
                    try:
                        scores.append(float(game.score))
                    except Exception:
                        readable = False
                now = clock()
                if now >= next_beat:
                    beat(seed, turn)
                    next_beat = now + self._every
                if now >= deadline:
                    abandoned = True
                    break
            ended = game.over
        except GameFault:
            raise
        except Exception as error:
            raise game_fault(error, _label(game), seed, turn, False) from error
        self._next_beat = next_beat
        return _Trace(seen, scores, ended, abandoned)

    def first_split(self, game: Game | FixedGame, fresh: Callable[[], Game | FixedGame],
                    deadline: float) -> _Split | None:
        """The first place a seed played twice with the same moves came out differently.

        Each seed plays twice on `game`, which has already played a good deal,
        and once on a copy nothing has played. A game that carries something over
        from one episode to the next differs on the first, one that depends on
        anything outside itself on either.
        """
        players = _Players(game, self.config.network)
        names = game.observation.names
        for index, (kind, place) in enumerate(_REPEATS):
            seed = SEED + place
            if time.perf_counter() >= deadline:
                self.out_of_time = True
                return None
            traces = []
            for on in (game, game, fresh()):
                rng = np.random.default_rng((SEED, 1000, index))
                traces.append(self._trace(on, players.make(kind, rng), seed, deadline))
            for is_fresh, other in ((False, traces[1]), (True, traces[2])):
                split = _split(traces[0], other, names)
                if split is not None:
                    return _Split(seed, kind, split[0], split[1], is_fresh)
        return None

    def check_seeds(self, game: Game, findings: list[Finding]) -> bool:
        """Whether the game's generators have to be seeded for it to repeat."""
        budget = self.config.probe.seconds * _REPEAT_SHARE
        deadline = min(time.perf_counter() + budget, self.end)
        split = self.first_split(game, self.fresh, deadline)
        if split is None:
            return False

        where = (f"Seed {split.seed} played with the same moves went differently the second "
                 f"time: {split.what} differed at step {split.step}")
        if split.fresh:
            where += ", between the game that had played before and a fresh copy of it"

        def seeded() -> FixedGame:
            return FixedGame(self.fresh(), Fixes(reseed=True))

        deadline = min(time.perf_counter() + budget, self.end)
        if self.first_split(seeded(), seeded, deadline) is None:
            findings.append(Finding(
                "unseeded",
                f"{where}. The seed does not decide the episode.",
                "Seeded the random generators the game holds at the start of every "
                "episode, and the same seed then gave the same episode.",
                outcome="fixed",
            ))
            return True
        findings.append(Finding(
            "unseeded",
            f"{where}. The seed does not decide the episode.",
            "Seeding the generators the game holds did not help, so it probably makes one "
            "inside its own reset, or reads the clock. Candidates will not be compared on "
            "the same episodes and runs will not repeat, until reset uses the seed it is given.",
        ))
        return False

    def run(self) -> Report:
        config = self.config
        ceiling = self.ceiling
        findings: list[Finding] = []
        game = self.fresh()
        observation = game.observation

        first = self.survey(game, 0.5, search=True)
        reseed = self.check_seeds(game, findings)

        # The inputs of the last pass played under the limits are the ones to
        # scale by: an input measured against a limit the game never reaches
        # only means something against the limit it will really have.
        measured = first
        stall_cap = step_cap = None
        unreadable = any(episode.unreadable for episode in first.played)
        cut = sum(episode.cut for episode in first.played)
        complete = sum(not episode.abandoned for episode in first.played)
        if unreadable:
            findings.append(Finding(
                "needs_stop",
                f"Episodes were still going after {ceiling:,} steps, and the game's results "
                "cannot be read before an episode is over.",
                "Could not put a limit on its episodes. The game has to end them itself, or "
                "have a stop method that finishes its results.",
            ))
        elif cut:
            stall = _stall_cap(first.played, ceiling)
            still_going = True
            if stall is not None:
                stall_cap, gap = stall
                measured = self.survey(FixedGame(game, Fixes(stall_cap=stall_cap, reseed=reseed)),
                                       0.5)
                still_going = any(episode.cut for episode in measured.played)
            if still_going:
                step_cap = ceiling
                limit = (f"Episodes now end after {step_cap:,} steps. That is only where the "
                         "probe stopped waiting, and probe.ceiling can raise it.")
            else:
                longest = max((e.steps for e in measured.played), default=0)
                step_cap = max(_STEP_TIMES * longest, 2 * ceiling)
                limit = (f"Episodes now end after {step_cap:,} steps, far past anything played, "
                         "so that limit is only there as a backstop.")
            if stall_cap is not None:
                limit = (f"An episode now ends after {stall_cap:,} steps with the score standing "
                         f"still, and the longest wait seen was {gap:,}. {limit}")
            findings.append(Finding(
                "endless",
                f"{cut} of {complete} episodes {'was' if cut == 1 else 'were'} still going "
                f"after {ceiling:,} steps, so nothing in the game is sure to end one.",
                limit,
                outcome="fixed",
            ))

        input_fixes = _examine_inputs(measured.inputs, first.inputs, observation, findings)
        fixes = Fixes(
            keep=input_fixes.keep,
            offset=input_fixes.offset,
            scale=input_fixes.scale,
            codes=input_fixes.codes,
            step_cap=step_cap,
            stall_cap=stall_cap,
            reseed=reseed,
        )

        # A limit or a seed changes nothing about what the network reads, and what
        # either does has been measured already.
        after = None
        if fixes.inputs_changed:
            after = self.survey(FixedGame(game, fixes), 1.0)
            if (not unreadable and step_cap is None
                    and any(episode.cut for episode in after.played)):
                # Played better with its inputs fixed, a player stopped dying.
                fixes = Fixes(
                    keep=fixes.keep, offset=fixes.offset, scale=fixes.scale,
                    step_cap=ceiling, stall_cap=fixes.stall_cap, reseed=fixes.reseed,
                )
                findings.append(Finding(
                    "endless",
                    f"With the inputs fixed, some episodes were still going after {ceiling:,} "
                    "steps, so nothing in the game is sure to end one.",
                    f"Episodes now end after {ceiling:,} steps. That is only where the probe "
                    "stopped waiting, and probe.ceiling can raise it.",
                    outcome="fixed",
                ))

        rare = _rare_reward(measured.played)
        if rare is not None:
            findings.append(rare)
        self._note_speed(fixes, first.played, findings)

        seconds = time.perf_counter() - self.started
        return Report(
            game=config.game,
            episodes=self.episodes,
            steps=self.steps,
            seconds=seconds,
            steps_per_second=self.steps / self.playing if self.playing else 0.0,
            ceiling=ceiling,
            inputs=_input_rows(observation, measured.inputs, input_fixes, fixes, after),
            players=_summarise(first.played),
            findings=tuple(findings),
            fixes=fixes,
            out_of_time=self.out_of_time,
        )

    def _note_speed(self, fixes: Fixes, played: list[_Played], findings: list[Finding]) -> None:
        limits = self.config.limits
        if self.slowest > limits.step_seconds:
            findings.append(Finding(
                "slow",
                f"One step took {self.slowest:.1f} s, more than limits.step_seconds allows "
                f"({limits.step_seconds:g} s).",
                "Changed nothing. A run ends if a step takes that long twice in a row.",
            ))
        if not self.steps:
            return
        each = self.playing / self.steps
        longest = fixes.step_cap or max((e.steps for e in played), default=0)
        if longest * each > limits.episode_seconds:
            findings.append(Finding(
                "slow",
                f"At {each * 1000:.1f} ms a step, an episode of {longest:,} steps takes "
                f"{longest * each:.0f} s, more than limits.episode_seconds allows "
                f"({limits.episode_seconds:g} s).",
                "Changed nothing. Either the game has to run faster or that limit has to go up.",
            ))


def _input_rows(observation: Observation, inputs: _Inputs, input_fixes: _InputFixes,
                fixes: Fixes, after: _Survey | None) -> tuple[InputRow, ...]:
    fates = input_fixes.fates or {}
    afterwards: dict[int, tuple[float, float]] = {}
    if after is not None and after.inputs.rows:
        kept = range(observation.size) if fixes.keep is None else fixes.keep
        codes = fixes.codes or (None,) * len(kept)
        start = 0
        # A code's columns, one per value, read together as the range they span.
        for column, values in zip(kept, codes, strict=True):
            end = start + (1 if values is None else len(values))
            afterwards[column] = (float(after.inputs.low[start:end].min()),
                                  float(after.inputs.high[start:end].max()))
            start = end
    rows = []
    for column, name in enumerate(observation.names):
        low, high = float(inputs.low[column]), float(inputs.high[column])
        after_low, after_high = afterwards.get(column, (None, None))
        rows.append(InputRow(
            name=name,
            low=low if math.isfinite(low) else None,
            high=high if math.isfinite(high) else None,
            fate=fates.get(column, "kept"),
            after_low=None if after_low is None or not math.isfinite(after_low) else after_low,
            after_high=None if after_high is None or not math.isfinite(after_high) else after_high,
        ))
    return tuple(rows)


def run(config: Config, beat: Callable[[int, int], None] | None = None) -> Report:
    """Measure the game `config` names and work out what to fix.

    Raises `GameFault` if the game's code fails while being played. A step that
    never returns holds this up for good; `run_guarded` is the form that survives it.
    """
    return _Probe(games.resolve(config), beat).run()


# ---------------------------------------------------------------- its own process

def _serve(conn, config: Config) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    conn.send(("ready",))
    try:
        report = run(config, beat=lambda seed, step: conn.send(("beat", seed, step)))
        reply = ("done", report)
    except GameFault as fault:
        reply = ("fault", fault)
    except Exception as error:
        reply = ("error", f"{type(error).__name__}: {error}", traceback.format_exc())
    conn.send(reply)


def run_guarded(config: Config) -> Report:
    """`run` in a process of its own, so a step that never returns can be ended.

    Going quiet for longer than a step is allowed raises a `GameFault` naming the
    last step heard of. So does the process dying.
    """
    config = games.resolve(config)
    limits = config.limits
    spawn = multiprocessing.get_context()
    ours, theirs = spawn.Pipe()
    process = spawn.Process(target=_serve, args=(theirs, config), daemon=True)
    process.start()
    # The process has its own copy now. Closing this one lets the pipe end with it.
    theirs.close()
    clock = time.perf_counter
    allowed = limits.step_seconds + beat_interval(limits) + SLACK
    where = "before it reported a step"
    ready = False
    heard = clock()
    try:
        while True:
            before = clock()
            timeout = min(max(heard + allowed - before, 0.0), 1.0) if ready else 1.0
            connection.wait([ours, process.sentinel], timeout)
            now = clock()
            if now - before - timeout > SLACK:
                # This process was held up past the wait it asked for, as when
                # the machine sleeps, so the silence since is not the game's.
                heard = now
            # Asked before reading, so all a process that has gone had to say is read.
            alive = process.is_alive()
            try:
                while ours.poll():
                    message = ours.recv()
                    heard = now
                    kind = message[0]
                    if kind == "done":
                        return message[1]
                    if kind == "fault":
                        raise message[1]
                    if kind == "error":
                        raise WorkerError(f"the probe failed: {message[1]}") from (
                            RemoteTraceback(message[2]))
                    if kind == "ready":
                        ready = True
                    else:
                        where = f"last heard from at step {message[2]} of the episode " \
                                f"with seed {message[1]}"
            except (EOFError, OSError):
                pass
            if not alive:
                process.join(timeout=1.0)
                raise GameFault(f"the probe's process stopped with exit code "
                                f"{process.exitcode}, {where}")
            if ready and now - heard > allowed:
                raise GameFault(
                    f"the probe went {now - heard:.1f} s without a word, longer than "
                    f"limits.step_seconds allows ({limits.step_seconds:g} s), {where}"
                )
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5.0)
        ours.close()
