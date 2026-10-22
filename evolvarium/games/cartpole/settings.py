"""Cart-pole's settings: the cart, the pole, the world they move in and an episode."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CartConfig:
    """The cart and the track it runs on."""

    mass: float = 1.0
    # The hardest push either way, in newtons. The network chooses any force
    # in between.
    force: float = 10.0
    # How far the cart's centre may go from the middle of the track, in metres,
    # before it runs off the end.
    track: float = 2.4


@dataclass(frozen=True)
class PoleConfig:
    """The pole, hinged at its foot to the top of the cart."""

    mass: float = 0.1
    # Its whole length in metres, with the mass spread evenly along it.
    length: float = 1.0
    # How far the pole may lean from upright, in degrees, before it has fallen.
    fall_angle: float = 12.0


@dataclass(frozen=True)
class WorldConfig:
    """Gravity, and the time one step stands for."""

    gravity: float = 9.8
    # In seconds. The network chooses a force once a step and the cart is
    # pushed with it for the whole step.
    step_time: float = 0.02


@dataclass(frozen=True)
class EpisodeConfig:
    """How an episode starts and how long it can last."""

    max_steps: int = 500
    # The cart's position and speed and the pole's angle and spin each start
    # at a random value within this much of zero, in metres, metres a second,
    # radians and radians a second.
    start_spread: float = 0.05


@dataclass(frozen=True)
class CartPoleSettings:
    cart: CartConfig = field(default_factory=CartConfig)
    pole: PoleConfig = field(default_factory=PoleConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
