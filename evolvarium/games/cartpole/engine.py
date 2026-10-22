"""Cart-pole's rules.

A cart runs on a straight, level track with a pole hinged to its top, and the
only way to keep the pole up is to push the cart. Each step the cart is pushed
with a force anywhere from full left to full right. The episode goes on until
the pole leans past the fall angle, the cart runs off the end of the track, or
the last step is played.

The motion is that of a cart and a uniform pole without friction. With F the
push, M the mass of cart and pole together, m the pole's mass, l half its
length, g gravity, x the cart's position and theta the pole's lean from upright,
positive toward positive x, the accelerations are

    a        = (F + m l theta'^2 sin(theta)) / M
    theta''  = (g sin(theta) - a cos(theta)) / (l (4/3 - m cos(theta)^2 / M))
    x''      = a - m l theta'' cos(theta) / M

and a step of `world.step_time` seconds moves the state on by Euler's method,
each position from the speed at the start of the step:

    x += t x'    x' += t x''    theta += t theta'    theta' += t theta''

The seed sets only the starting state, so an episode is a pure function of
the seed and the forces.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from .settings import CartPoleSettings

# What the network sees, in order. See `Cart.observe`.
INPUTS = ("cart_position", "cart_speed", "pole_angle", "pole_spin")


class EndCause(str, Enum):
    FELL = "fell"
    OFF_TRACK = "off_track"
    BALANCED = "balanced"


class Cart:
    """A single episode: the cart, its pole, and how they have done so far."""

    def __init__(self, settings: CartPoleSettings, seed: int):
        self.settings = settings
        cart, pole = settings.cart, settings.pole
        self.track = cart.track
        self.limit = cart.force
        self.fall = math.radians(pole.fall_angle)
        self._mass = cart.mass + pole.mass
        self._half = pole.length / 2
        self._moment = pole.mass * self._half

        rng = random.Random(seed)
        spread = settings.episode.start_spread
        self.x, self.speed, self.angle, self.spin = (
            rng.uniform(-spread, spread) for _ in range(4)
        )

        # Steps after which the pole was still up and the cart on the track.
        self.balanced = 0
        self.steps = 0
        self.finished = False
        self.end: EndCause | None = None
        self.force = 0.0
        # Totals over the steps played, for the means an episode reports.
        self._drift = 0.0
        self._lean = 0.0
        self._effort = 0.0

    # -------------------------------------------------------------- playing

    def step(self, force: float) -> bool:
        """Push the cart with `force` newtons for one step.

        A force beyond the cart's hardest push is cut back to it. Returns True
        while the episode goes on.
        """
        if self.finished:
            return False
        settings = self.settings
        limit = self.limit
        force = min(limit, max(-limit, float(force)))
        gravity, tau = settings.world.gravity, settings.world.step_time

        sin, cos = math.sin(self.angle), math.cos(self.angle)
        shared = (force + self._moment * self.spin * self.spin * sin) / self._mass
        turning = (gravity * sin - cos * shared) / (
            self._half * (4.0 / 3.0 - settings.pole.mass * cos * cos / self._mass)
        )
        rolling = shared - self._moment * turning * cos / self._mass

        self.x += tau * self.speed
        self.speed += tau * rolling
        self.angle += tau * self.spin
        self.spin += tau * turning

        self.steps += 1
        self.force = force
        self._drift += abs(self.x) / self.track
        self._lean += abs(self.angle) / self.fall
        self._effort += abs(force) / limit
        if abs(self.angle) > self.fall:
            return self._finish(EndCause.FELL)
        if abs(self.x) > self.track:
            return self._finish(EndCause.OFF_TRACK)
        self.balanced += 1
        if self.steps >= settings.episode.max_steps:
            return self._finish(EndCause.BALANCED)
        return True

    def _finish(self, cause: EndCause) -> bool:
        self.finished = True
        self.end = cause
        return False

    # -------------------------------------------------------------- results

    @property
    def drift(self) -> float:
        """How far the cart was from the middle, on average, over `cart.track`."""
        return self._drift / self.steps if self.steps else 0.0

    @property
    def lean(self) -> float:
        """How far the pole leant, on average, over the fall angle."""
        return self._lean / self.steps if self.steps else 0.0

    @property
    def effort(self) -> float:
        """How hard the cart was pushed, on average, over its hardest push."""
        return self._effort / self.steps if self.steps else 0.0

    # -------------------------------------------------------------- seeing

    def observe(self) -> list[float]:
        """What the network sees, in the order of `INPUTS`.

        The cart's position over `cart.track` and the pole's angle over the
        fall angle, so both run from -1 to 1 while the episode goes on.
        The cart's speed in metres a second and the pole's spin in radians a
        second, which stay within a few units of zero while the pole is up.
        """
        return [self.x / self.track, self.speed, self.angle / self.fall, self.spin]
