"""Hand-written teams, the bar an evolved team is measured against.

Each policy reads the inputs a network reads and gives one of the game's moves,
so a scripted team and an evolved one play the same match from the same view.
Standing still and moving at random show what a team scores with no skill at
all. The forager only eats, healing when hurt. The hunter chases the rival and
eats when the rival is far off, and when hurt backs off to heal unless the
rival is next to it, resting after a strike, and can be caught. The reckless
hunter chases whatever its health, which shows what backing off is worth.

    python -m evolvarium.games.arena.scripted

plays every policy on the held-out seeds and prints what each scored.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from statistics import fmean as mean

from ...config import HeldOutConfig
from .engine import INPUTS, MOVES, Match
from .settings import ArenaSettings

Policy = Callable[[Sequence[float], random.Random], str]

_AT = {name: index for index, name in enumerate(INPUTS)}
_BLOCKED = {move: _AT[f"blocked_{move}"] for move in ("up", "down", "left", "right")}
_SIDES = {
    "up": ("left", "right"), "down": ("left", "right"),
    "left": ("up", "down"), "right": ("up", "down"),
}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _toward(view: Sequence[float], dx: float, dy: float) -> str:
    """The move that closes most on the way (dx, dy), stepping round a pillar."""
    ways = []
    if dx:
        ways.append((abs(dx), "right" if dx > 0 else "left"))
    if dy:
        ways.append((abs(dy), "down" if dy > 0 else "up"))
    if not ways:
        return "stay"
    ways.sort(reverse=True)
    for _, move in ways:
        if not view[_BLOCKED[move]]:
            return move
    # Every way that closes in is blocked: step aside before stepping back.
    first = ways[0][1]
    for move in (*_SIDES[first], _OPPOSITE[first]):
        if not view[_BLOCKED[move]]:
            return move
    return "stay"


def _seen(view: Sequence[float], name: str) -> bool:
    # Something absent, or under the agent, has no way to it.
    return bool(view[_AT[f"{name}_x"]] or view[_AT[f"{name}_y"]])


def _go(view: Sequence[float], name: str, sign: float = 1.0) -> str:
    return _toward(view, sign * view[_AT[f"{name}_x"]], sign * view[_AT[f"{name}_y"]])


def still(view: Sequence[float], rng: random.Random) -> str:
    return "stay"


def wander(view: Sequence[float], rng: random.Random) -> str:
    return rng.choice(tuple(MOVES))


def forager(view: Sequence[float], rng: random.Random) -> str:
    """Eats, and heads for a pack when hurt."""
    if view[_AT["hurt"]] and _seen(view, "pack"):
        return _go(view, "pack")
    return _go(view, "food")


def hunter(view: Sequence[float], rng: random.Random) -> str:
    """Chases the rival when it is within half the floor, and eats otherwise.

    When hurt it catches a rival next to it and otherwise heads for a pack, or
    away from the rival when no pack is on the floor.
    """
    if view[_AT["hurt"]] and not view[_AT["rival_adjacent"]]:
        if _seen(view, "pack"):
            return _go(view, "pack")
        return _go(view, "rival", -1.0)
    if view[_AT["rival_distance"]] <= 0.5:
        return _go(view, "rival")
    return _go(view, "food")


def reckless(view: Sequence[float], rng: random.Random) -> str:
    """The hunter without the backing off."""
    if view[_AT["rival_distance"]] <= 0.5:
        return _go(view, "rival")
    return _go(view, "food")


POLICIES: dict[str, Policy] = {
    "still": still,
    "wander": wander,
    "forager": forager,
    "hunter": hunter,
    "reckless": reckless,
}


def play(settings: ArenaSettings, policy: Policy, seed: int) -> Match:
    """One match with every agent playing `policy`."""
    match = Match(settings, seed)
    rng = random.Random(seed ^ 0x5EED)
    agents = range(len(match.team))
    while not match.finished:
        match.step([policy(match.observe(index), rng) for index in agents])
    return match


def main(argv: list[str]) -> int:
    held_out = HeldOutConfig()
    seeds = range(held_out.first_seed, held_out.first_seed + held_out.episodes)
    settings = ArenaSettings()
    print(f"{'policy':<9} {'score':>7} {'food':>6} {'catches':>8} {'healed':>7} "
          f"{'fallen':>7} {'steps':>6}  ends")
    for name, policy in POLICIES.items():
        matches = [play(settings, policy, seed) for seed in seeds]
        ends = Counter(m.end.value for m in matches)
        print(
            f"{name:<9} {mean(m.score for m in matches):>7.2f} "
            f"{mean(m.eaten for m in matches):>6.1f} "
            f"{mean(m.catches for m in matches):>8.2f} "
            f"{mean(m.healed for m in matches):>7.2f} "
            f"{mean(len(m.team) - m.standing for m in matches):>7.2f} "
            f"{mean(m.steps for m in matches):>6.0f}  "
            + ", ".join(f"{end} {n}" for end, n in ends.most_common())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
