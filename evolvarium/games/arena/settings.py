"""The arena's settings: the floor, the team, the rival, the pickups, the points and the end."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FloorConfig:
    """The floor, in cells, and how many pillars stand on it."""

    width: int = 15
    height: int = 15
    # Pillars stand only where both coordinates are odd and away from the edge,
    # so no two touch and every open cell can reach every other.
    pillars: int = 10


@dataclass(frozen=True)
class TeamConfig:
    """The agents. Every one of them plays the same network."""

    size: int = 3
    health: int = 6
    # An agent at or below this much health counts as hurt: the rival hunts it
    # rather than running from it.
    hurt_at: int = 2


@dataclass(frozen=True)
class RivalConfig:
    """The scripted character the team chases."""

    damage: int = 2
    # How far it strikes, in steps. It strikes one agent each step, the one with
    # the least health in reach.
    reach: int = 1
    # How close a hurt agent has to be before the rival turns on it.
    sight: int = 5
    # It stands still one step in this many, which is how a chasing agent gains
    # on it. 0 keeps it moving every step.
    rest_every: int = 3
    # A caught rival comes back at least this far from every agent, or as far as
    # the floor allows.
    respawn_distance: int = 6


@dataclass(frozen=True)
class PickupConfig:
    """Food is always on the floor; a used health pack comes back after a delay."""

    food: int = 3
    packs: int = 2
    heal: int = 4
    pack_delay: int = 40


@dataclass(frozen=True)
class PointsConfig:
    """What the score counts.

    A catch is worth ten bites, so a team that only eats scores less than one
    that chases the rival and heals between catches.
    """

    food: int = 1
    catch: int = 10


@dataclass(frozen=True)
class MatchConfig:
    """Rules that end a match."""

    max_steps: int = 400
    # A match ends when the team goes this many steps without a point, which
    # stops a team that stands still from running out the clock.
    idle_limit: int = 120


@dataclass(frozen=True)
class ArenaSettings:
    floor: FloorConfig = field(default_factory=FloorConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    rival: RivalConfig = field(default_factory=RivalConfig)
    pickups: PickupConfig = field(default_factory=PickupConfig)
    points: PointsConfig = field(default_factory=PointsConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
