"""The arena's rules.

A match is a pure function of the team's moves and the seed, as a game of Snake
is: two teams that play the same seed meet the same floor and the same pickups,
and a rival that answers each of them by the same rules.

Each step plays out in a fixed order. Every standing agent moves, and one that
moves onto the rival catches it. Agents then eat the food and use the packs they
stand on, and packs that are due come back. The rival moves, or rests, and then
strikes the weakest agent in reach. It always rests the step after a strike,
which gives the agent it struck one step to back away or to catch it. An agent
whose health has run out falls and plays no further.
"""

from __future__ import annotations

import random
from enum import Enum

from .settings import ArenaSettings

Cell = tuple[int, int]

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0)}
# The rival weighs its moves in this order, so the earliest wins a tie.
_RIVAL_STEPS = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))

# What each agent sees, in order. See `Match.observe`.
INPUTS = (
    "health", "hurt",
    "food_x", "food_y", "food_distance",
    "pack_x", "pack_y", "pack_distance",
    "rival_x", "rival_y", "rival_distance", "rival_adjacent",
    "mate_x", "mate_y", "mate_distance",
    "blocked_up", "blocked_down", "blocked_left", "blocked_right",
)
# A fallen agent sees nothing near it. The engine still asks it for a move, and
# the move is ignored.
FALLEN_VIEW = tuple(1.0 if name.endswith("_distance") else 0.0 for name in INPUTS)


class EndCause(str, Enum):
    FALLEN = "fallen"
    TIMEOUT = "timeout"
    IDLE = "idle"


def distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pillar_sites(width: int, height: int) -> list[Cell]:
    """Where pillars may stand: cells with both coordinates odd, off the edge.

    Every column and row with an even coordinate stays open, and every other
    open cell lies next to one, so the open floor is always in one piece.
    """
    return [(x, y) for y in range(1, height - 1, 2) for x in range(1, width - 1, 2)]


class Member:
    """One agent of the team."""

    __slots__ = ("cell", "health")

    def __init__(self, cell: Cell, health: int):
        self.cell = cell
        self.health = health

    @property
    def standing(self) -> bool:
        return self.health > 0


class Match:
    """A single match."""

    def __init__(self, settings: ArenaSettings, seed: int):
        self.settings = settings
        floor, team, pickups = settings.floor, settings.team, settings.pickups
        self.width = floor.width
        self.height = floor.height
        # The longest Manhattan distance on the floor, which scales distances
        # in what an agent sees.
        self.span = self.width + self.height - 2
        self.rng = random.Random(seed)

        sites = pillar_sites(self.width, self.height)
        self.pillars = frozenset(self.rng.sample(sites, floor.pillars))
        self.open = [
            (x, y) for y in range(self.height) for x in range(self.width)
            if (x, y) not in self.pillars
        ]
        self.team = [Member(cell, team.health) for cell in self.rng.sample(self.open, team.size)]
        self.rival = self._far_cell()
        self.food: list[Cell] = []
        # A used pack is None until the step it is due back.
        self.packs: list[Cell | None] = []
        for _ in range(pickups.food):
            self.food.append(self._empty_cell())
        for _ in range(pickups.packs):
            self.packs.append(self._empty_cell())
        self._due = [0] * pickups.packs

        self.score = 0
        self.steps = 0
        self.idle = 0
        self.finished = False
        self.end: EndCause | None = None
        self.eaten = 0
        self.catches = 0
        self.healed = 0
        self.strikes = 0
        self._winded = False

    # -------------------------------------------------------------- the floor

    def passable(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.pillars

    def _empty_cell(self) -> Cell:
        """A random open cell with no agent, rival or pickup on it.

        The settings leave room for the team, the rival and every pickup at
        once, so there is always one.
        """
        taken = {m.cell for m in self.team if m.standing}
        taken.add(self.rival)
        taken.update(self.food)
        taken.update(self.packs)
        return self.rng.choice([cell for cell in self.open if cell not in taken])

    def _far_cell(self) -> Cell:
        """Where the rival starts or comes back after a catch.

        A random open cell at least `respawn_distance` from every standing
        agent, or, on a floor without one, among the farthest there are.
        """
        standing = [m.cell for m in self.team if m.standing]
        gaps = [
            (cell, min(distance(cell, other) for other in standing))
            for cell in self.open if cell not in standing
        ]
        wanted = min(self.settings.rival.respawn_distance, max(gap for _, gap in gaps))
        return self.rng.choice([cell for cell, gap in gaps if gap >= wanted])

    @property
    def standing(self) -> int:
        return sum(1 for m in self.team if m.standing)

    def hurt(self, member: Member) -> bool:
        return member.health <= self.settings.team.hurt_at

    # -------------------------------------------------------------- playing

    def step(self, moves: list[str]) -> bool:
        """Play one step, given a move for each agent in team order.

        Returns True while the match goes on.
        """
        if self.finished:
            return False
        settings = self.settings
        team, rival = settings.team, settings.rival
        pickups, points = settings.pickups, settings.points
        now = self.steps + 1
        scored = 0

        for member, move in zip(self.team, moves, strict=True):
            if member.standing:
                dx, dy = MOVES[move]
                target = (member.cell[0] + dx, member.cell[1] + dy)
                if self.passable(target):
                    member.cell = target

        if any(m.standing and m.cell == self.rival for m in self.team):
            self.catches += 1
            scored += points.catch
            self.rival = self._far_cell()
            self._winded = False

        for member in self.team:
            if not member.standing:
                continue
            cell = member.cell
            if cell in self.food:
                self.eaten += 1
                scored += points.food
                self.food[self.food.index(cell)] = self._empty_cell()
            # A pack is left where it is for an agent that would gain nothing.
            if member.health < team.health and cell in self.packs:
                index = self.packs.index(cell)
                member.health = min(team.health, member.health + pickups.heal)
                self.packs[index] = None
                self._due[index] = now + pickups.pack_delay
                self.healed += 1

        for index, cell in enumerate(self.packs):
            if cell is None and now >= self._due[index]:
                self.packs[index] = self._empty_cell()

        if not (self._winded or rival.rest_every and now % rival.rest_every == 0):
            self.rival = self._rival_move()
        # A rival that does no damage never strikes, so it never rests for one.
        in_reach = [
            m for m in self.team if m.standing and distance(m.cell, self.rival) <= rival.reach
        ] if rival.damage else []
        self._winded = bool(in_reach)
        if in_reach:
            weakest = min(in_reach, key=lambda m: m.health)
            weakest.health = max(0, weakest.health - rival.damage)
            self.strikes += 1

        self.steps = now
        self.score += scored
        self.idle = 0 if scored else self.idle + 1
        if not any(m.standing for m in self.team):
            return self._finish(EndCause.FALLEN)
        if self.steps >= settings.match.max_steps:
            return self._finish(EndCause.TIMEOUT)
        if self.idle >= settings.match.idle_limit:
            return self._finish(EndCause.IDLE)
        return True

    def _rival_move(self) -> Cell:
        """The rival's next cell.

        It hunts the nearest hurt agent it can see, closing in on it. With none
        in sight it runs from the team, to the cell farthest from the agent
        nearest it and then farthest from the team as a whole. It never steps
        onto an agent.
        """
        standing = [m.cell for m in self.team if m.standing]
        if not standing:
            return self.rival
        x, y = self.rival
        options = [
            cell for cell in ((x + dx, y + dy) for dx, dy in _RIVAL_STEPS)
            if cell == self.rival or (self.passable(cell) and cell not in standing)
        ]
        sight = self.settings.rival.sight
        prey = [
            m.cell for m in self.team
            if m.standing and self.hurt(m) and distance(m.cell, self.rival) <= sight
        ]
        if prey:
            target = min(prey, key=lambda cell: distance(cell, self.rival))
            return min(options, key=lambda cell: distance(cell, target))
        return max(options, key=lambda cell: (
            min(distance(cell, other) for other in standing),
            sum(distance(cell, other) for other in standing),
        ))

    def _finish(self, cause: EndCause) -> bool:
        self.finished = True
        self.end = cause
        return False

    # -------------------------------------------------------------- seeing

    def observe(self, index: int) -> list[float]:
        """What agent `index` sees, in the order of `INPUTS`.

        Its own health as a share of the most it can have, and whether it is
        hurt. The way to the nearest food, the nearest pack on the floor, the
        rival and the nearest standing teammate, each with its distance, and
        whether the rival is one step away. Then whether a wall or a pillar
        stands next to it above, below, left and right.
        """
        member = self.team[index]
        if not member.standing:
            return list(FALLEN_VIEW)
        team = self.settings.team
        cell = member.cell
        x, y = cell
        mates = [
            other.cell for number, other in enumerate(self.team)
            if number != index and other.standing
        ]
        return [
            member.health / team.health,
            1.0 if member.health <= team.hurt_at else 0.0,
            *self._toward(cell, self.food),
            *self._toward(cell, [p for p in self.packs if p is not None]),
            *self._toward(cell, (self.rival,)),
            1.0 if distance(cell, self.rival) == 1 else 0.0,
            *self._toward(cell, mates),
            0.0 if self.passable((x, y - 1)) else 1.0,
            0.0 if self.passable((x, y + 1)) else 1.0,
            0.0 if self.passable((x - 1, y)) else 1.0,
            0.0 if self.passable((x + 1, y)) else 1.0,
        ]

    def _toward(self, cell: Cell, targets) -> tuple[float, float, float]:
        """The way to the nearest of `targets` and how far it is.

        The way is the step in x and in y over their sum, so the two add up to
        1 in size, and the distance is over the longest on the floor. With no
        target the way is 0 and the distance 1.
        """
        nearest = None
        best = 0
        for target in targets:
            gap = distance(cell, target)
            if nearest is None or gap < best:
                nearest, best = target, gap
        if nearest is None:
            return 0.0, 0.0, 1.0
        if best == 0:
            return 0.0, 0.0, 0.0
        return (nearest[0] - cell[0]) / best, (nearest[1] - cell[1]) / best, best / self.span
