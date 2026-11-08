"""What the probe changed about a game, and the game played with those changes.

`Fixes` is plain data, recorded with a run and sent to every worker. `with_fixes` wraps
a game in `FixedGame` so the rest of the engine plays it through the same contract
as any other; with nothing to fix it returns the game itself, so a game the probe
found nothing wrong with is played by exactly the code that played it before.

Inputs are moved and scaled by fixed numbers measured once, never against running
statistics, so an input means the same thing in the first generation as the last
and a saved model can do the same sums. An input read as a code becomes one
input per value it takes, as a declared category does.
"""

from __future__ import annotations

import random
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

import numpy as np

from .contract import Game, Observation

# How many objects `seed_generators` looks through in one game before giving up.
_REACH = 10_000
_SEED_MASK = 2**32 - 1


@dataclass(frozen=True)
class Fixes:
    """The changes made to a game. Empty fields change nothing."""

    # Raw input columns the network reads, in order, or None for all of them.
    keep: tuple[int, ...] | None = None
    # Added to, then multiplied with, each column kept.
    offset: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None
    # For each column kept, the values it is read as a code of, or None for a
    # column read as a number. A value not among them reads as the nearest.
    codes: tuple[tuple[float, ...] | None, ...] | None = None
    # Steps an episode may run before it is ended.
    step_cap: int | None = None
    # Steps it may run with the score standing still before it is ended.
    stall_cap: int | None = None
    # Seed the generators the game holds, since its `reset` does not.
    reseed: bool = False

    @property
    def empty(self) -> bool:
        return self == Fixes()

    @property
    def inputs_changed(self) -> bool:
        return self.keep is not None or self.scale is not None or self.codes is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> Fixes | None:
        """The fixes a record holds; None for one written without any."""
        if not data:
            return None
        return cls(
            keep=None if data.get("keep") is None else tuple(data["keep"]),
            offset=None if data.get("offset") is None else tuple(data["offset"]),
            scale=None if data.get("scale") is None else tuple(data["scale"]),
            codes=None if data.get("codes") is None else tuple(
                None if values is None else tuple(values) for values in data["codes"]
            ),
            step_cap=data.get("step_cap"),
            stall_cap=data.get("stall_cap"),
            reseed=bool(data.get("reseed", False)),
        )


def seed_generators(game: Game, seed: int) -> None:
    """Seed the random generators a game holds, and the shared ones.

    Each generator's seed comes from where it sits in the game, so it gets the
    same one whatever else the game holds at the time, such as the last episode's
    generators. A generator the game builds unseeded inside its own `reset` cannot
    be reached from here.
    """
    seed &= _SEED_MASK
    random.seed(seed)
    np.random.seed(seed)
    for path, generator in _generators(game):
        derived = (seed ^ zlib.crc32(path.encode())) & _SEED_MASK
        if isinstance(generator, np.random.Generator):
            kind = type(generator.bit_generator)
            generator.bit_generator.state = kind(derived).state
        else:
            generator.seed(derived)


def _generators(game: Game) -> list[tuple[str, Any]]:
    """Every generator `game` can reach, with the path that reaches it."""
    found = []
    seen = {id(game)}
    waiting = deque([("", game)])
    looked = 0
    while waiting and looked < _REACH:
        path, item = waiting.popleft()
        looked += 1
        if isinstance(item, random.Random | np.random.Generator | np.random.RandomState):
            found.append((path, item))
            continue
        for name, part in _parts(item):
            if id(part) not in seen and _worth_a_look(part):
                seen.add(id(part))
                waiting.append((f"{path}.{name}", part))
    return found


def _parts(item: Any) -> list[tuple[Any, Any]]:
    if isinstance(item, dict):
        return list(item.items())
    if isinstance(item, list | tuple):
        return list(enumerate(item))
    parts = list(getattr(item, "__dict__", {}).items())
    for cls in type(item).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if hasattr(item, name):
                parts.append((name, getattr(item, name)))
    return parts


def _worth_a_look(item: Any) -> bool:
    # Sets are left out because the order they give things in does not repeat.
    if item is None or callable(item):
        return False
    return not isinstance(
        item, str | bytes | int | float | complex | set | frozenset | np.ndarray | ModuleType
    )


class FixedGame:
    """A game played with `Fixes` applied. Anything not changed goes to the game."""

    def __init__(self, game: Game, fixes: Fixes):
        self.base = game
        self.fixes = fixes
        self._encode = game.observation.encode
        self._keep = None if fixes.keep is None else np.array(fixes.keep, dtype=np.intp)
        self._offset = None if fixes.offset is None else np.array(fixes.offset, dtype=np.float32)
        self._scale = None if fixes.scale is None else np.array(fixes.scale, dtype=np.float32)
        self._codes = None
        if fixes.inputs_changed:
            names = game.observation.names
            kept = names if fixes.keep is None else [names[i] for i in fixes.keep]
            if fixes.codes is not None:
                kept = self._spread(kept, fixes.codes)
            self.observation = Observation(list(kept))
        else:
            self.observation = game.observation
        self._turns = 0
        self._quiet = 0
        self._last = None
        self._cut: str | None = None

    def _spread(self, kept: list[str], codes) -> list[str]:
        """The names of the network's inputs with each code given one per value.

        Works out where each column lands, for `observe`.
        """
        names, plain_from, plain_to, coded = [], [], [], []
        for column, (name, values) in enumerate(zip(kept, codes, strict=True)):
            if values is None:
                plain_from.append(column)
                plain_to.append(len(names))
                names.append(name)
            else:
                coded.append((column, len(names), np.asarray(values, dtype=np.float32)))
                names.extend(f"{name}={code_label(value)}" for value in values)
        self._codes = (len(names), np.array(plain_from, dtype=np.intp),
                       np.array(plain_to, dtype=np.intp), coded)
        return names

    def __getattr__(self, name: str):
        # Only reached for what this class does not define itself. `base` is
        # looked up here while unpickling, before it has been set.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def reset(self, seed: int) -> None:
        if self.fixes.reseed:
            seed_generators(self.base, seed)
        self.base.reset(seed)
        self._turns = 0
        self._quiet = 0
        self._last = None
        self._cut = None

    def observe(self, agent: str):
        if not self.fixes.inputs_changed:
            return self.base.observe(agent)
        values = self._encode(self.base.observe(agent))
        if self._keep is not None:
            values = values[self._keep]
        if self._offset is not None:
            values = (values + self._offset) * self._scale
        if self._codes is not None:
            width, plain_from, plain_to, coded = self._codes
            spread = np.zeros(width, dtype=np.float32)
            spread[plain_to] = values[plain_from]
            for column, start, codes in coded:
                spread[start + int(np.abs(codes - values[column]).argmin())] = 1.0
            values = spread
        return values

    def step(self, actions: dict) -> None:
        base = self.base
        base.step(actions)
        self._turns += 1
        if base.over:
            return
        stall_cap = self.fixes.stall_cap
        if stall_cap is not None:
            score = base.score
            if score != self._last:
                self._last = score
                self._quiet = 0
            else:
                self._quiet += 1
                if self._quiet >= stall_cap:
                    self._stop(True)
                    return
        step_cap = self.fixes.step_cap
        if step_cap is not None and self._turns >= step_cap:
            self._stop(False)

    def _stop(self, stalled: bool) -> None:
        self.base.stop(stalled)
        self._cut = "stalled" if stalled else "step limit"

    @property
    def over(self) -> bool:
        return self._cut is not None or self.base.over

    @property
    def end(self) -> str:
        # A game that needs no `stop` may still have no name for how it ended.
        if self.base.over or self._cut is None:
            return self.base.end
        return self._cut


def code_label(value: float) -> str:
    """A code's value as its input is named: whole numbers without a decimal point."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def with_fixes(game: Game, fixes: Fixes | None) -> Game | FixedGame:
    """`game` with `fixes` applied, or `game` itself when there is nothing to apply."""
    if fixes is None or fixes.empty:
        return game
    return FixedGame(game, fixes)
