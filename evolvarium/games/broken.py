"""The built-in games with a fault planted in each, for testing what finds them.

Every game comes in five versions, one for each fault a game written in a hurry
tends to have:

* unscaled: its inputs in units of their own, some in the hundreds and some
  in thousandths sitting far from zero;
* dead: four inputs that never change, spread among the real ones;
* duplicate: three inputs that copy others, one of them flipped and doubled;
* endless: no rule ends an episode after so many steps, so one a good agent
  plays can go on as long as the agent does;
* unseeded: an episode is drawn from a generator of the game's own, not from
  the seed it is given.

None of them is built in. A run names one as "evolvarium.games.broken:Class",
such as `--game evolvarium.games.broken:UnscaledSnake`, and `VARIANTS` lists
them all by game and fault.

Snake's hunger input is measured against its starvation limit, so in the
endless snake it stays close to zero.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..contract import Game, Observation
from .arena import settings as arena
from .arena.game import ArenaGame
from .cartpole import settings as cartpole
from .cartpole.game import CartPoleGame
from .snake import settings as snake
from .snake.game import SnakeGame

# Each input's scale and offset, in turn. One is left as it is, one runs into
# the tens, one sits in the hundreds and one moves by thousandths around -3.
_SCALES = (1.0, 30.0, 600.0, 0.002)
_OFFSETS = (0.0, 0.0, 250.0, -3.0)
# Zeros where an input was never filled in, and values such as a flag that is
# always set.
_CONSTANTS = np.array((0.0, 0.0, 1.0, -0.5), dtype=np.float32)
# A limit no episode reaches.
_NEVER = 10**9


class _Rewritten:
    """A game whose observations are changed on their way out of the game."""

    def __init__(self, settings=None):
        super().__init__(settings)
        # The game's own inputs, which `observe` builds on.
        self._base = self.observation

    def observe(self, agent: str):
        return self._rewrite(self._base.encode(super().observe(agent)))

    def _rewrite(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Unscaled(_Rewritten):
    def __init__(self, settings=None):
        super().__init__(settings)
        size = self._base.size
        self._scale = np.resize(np.array(_SCALES, dtype=np.float32), size)
        self._offset = np.resize(np.array(_OFFSETS, dtype=np.float32), size)

    def _rewrite(self, values: np.ndarray) -> np.ndarray:
        return values * self._scale + self._offset


class _Added(_Rewritten):
    """The game's inputs with more added, spread evenly among them."""

    def __init__(self, settings=None):
        super().__init__(settings)
        names = list(self._base.names)
        added = self._added(names)
        count = len(names)
        total = count + len(added)
        order = list(range(count))
        for k in range(len(added)):
            # Places in the finished order, rising, so each lands where it is put.
            order.insert((k + 1) * total // (len(added) + 1), count + k)
        self._order = np.array(order)
        every = names + added
        self.observation = Observation([every[i] for i in order])

    def _added(self, names: list[str]) -> list[str]:
        """Names of the inputs added to `names`, in the order `_extra` gives them."""
        raise NotImplementedError

    def _extra(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _rewrite(self, values: np.ndarray) -> np.ndarray:
        return np.concatenate((values, self._extra(values)))[self._order]


class _DeadInput(_Added):
    def _added(self, names: list[str]) -> list[str]:
        return [f"constant {n}" for n in range(1, len(_CONSTANTS) + 1)]

    def _extra(self, values: np.ndarray) -> np.ndarray:
        return _CONSTANTS


class _DuplicateInput(_Added):
    def _added(self, names: list[str]) -> list[str]:
        picks = list(dict.fromkeys((0, len(names) // 2, len(names) - 1)))
        self._picks = np.array(picks)
        # The last copy is flipped and doubled, which a check for equal inputs
        # misses and a correlation does not.
        self._factors = np.ones(len(picks), dtype=np.float32)
        self._factors[-1] = -2.0
        return [f"copy of {names[i]}" for i in picks]

    def _extra(self, values: np.ndarray) -> np.ndarray:
        return values[self._picks] * self._factors


class _Unseeded:
    def __init__(self, settings=None):
        super().__init__(settings)
        # Seeded by the operating system, so no two games draw the same episodes.
        self._chance = random.Random()

    def reset(self, seed: int) -> None:
        super().reset(self._chance.getrandbits(32))


# ---------------------------------------------------------------- endless rules

@dataclass(frozen=True)
class EndlessSnakeSettings(snake.SnakeSettings):
    episode: snake.EpisodeConfig = field(
        default_factory=lambda: snake.EpisodeConfig(starvation_limit=_NEVER, max_steps=_NEVER)
    )


@dataclass(frozen=True)
class EndlessCartPoleSettings(cartpole.CartPoleSettings):
    episode: cartpole.EpisodeConfig = field(
        default_factory=lambda: cartpole.EpisodeConfig(max_steps=_NEVER)
    )


@dataclass(frozen=True)
class EndlessArenaSettings(arena.ArenaSettings):
    match: arena.MatchConfig = field(
        default_factory=lambda: arena.MatchConfig(max_steps=_NEVER, idle_limit=_NEVER)
    )


# ---------------------------------------------------------------- snake

class UnscaledSnake(_Unscaled, SnakeGame):
    name = "snake-unscaled"


class DeadInputSnake(_DeadInput, SnakeGame):
    name = "snake-dead"


class DuplicateInputSnake(_DuplicateInput, SnakeGame):
    name = "snake-duplicate"


class EndlessSnake(SnakeGame):
    name = "snake-endless"
    Settings = EndlessSnakeSettings


class UnseededSnake(_Unseeded, SnakeGame):
    name = "snake-unseeded"


# ---------------------------------------------------------------- cart-pole

class UnscaledCartPole(_Unscaled, CartPoleGame):
    name = "cartpole-unscaled"


class DeadInputCartPole(_DeadInput, CartPoleGame):
    name = "cartpole-dead"


class DuplicateInputCartPole(_DuplicateInput, CartPoleGame):
    name = "cartpole-duplicate"


class EndlessCartPole(CartPoleGame):
    name = "cartpole-endless"
    Settings = EndlessCartPoleSettings


class UnseededCartPole(_Unseeded, CartPoleGame):
    name = "cartpole-unseeded"


# ---------------------------------------------------------------- arena

class UnscaledArena(_Unscaled, ArenaGame):
    name = "arena-unscaled"


class DeadInputArena(_DeadInput, ArenaGame):
    name = "arena-dead"


class DuplicateInputArena(_DuplicateInput, ArenaGame):
    name = "arena-duplicate"


class EndlessArena(ArenaGame):
    name = "arena-endless"
    Settings = EndlessArenaSettings


class UnseededArena(_Unseeded, ArenaGame):
    name = "arena-unseeded"


FAULTS = ("unscaled", "dead", "duplicate", "endless", "unseeded")

# Every version, by the built-in game it comes from and its fault.
VARIANTS: dict[str, dict[str, type[Game]]] = {
    "snake": dict(zip(FAULTS, (UnscaledSnake, DeadInputSnake, DuplicateInputSnake,
                               EndlessSnake, UnseededSnake), strict=True)),
    "cartpole": dict(zip(FAULTS, (UnscaledCartPole, DeadInputCartPole, DuplicateInputCartPole,
                                  EndlessCartPole, UnseededCartPole), strict=True)),
    "arena": dict(zip(FAULTS, (UnscaledArena, DeadInputArena, DuplicateInputArena,
                               EndlessArena, UnseededArena), strict=True)),
}
