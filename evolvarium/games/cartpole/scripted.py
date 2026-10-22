"""Hand-written controllers, the bar an evolved one is measured against.

Each policy reads the inputs a network reads and gives a push as a share of the
cart's hardest, from -1 for full left to 1 for full right, and anything beyond
that is cut back to it. A scripted cart and an evolved one so play the same
episode from the same view. Doing nothing and pushing at random show how long
the pole stays up with no skill at all. Lean pushes full force whichever way the
pole leans. Upright pushes in proportion to the lean and the spin and pays the
cart no attention, so it keeps the pole up while the cart drifts. Steady also
pushes toward the side the cart has drifted to and the way it is rolling, which
tips the pole back toward the middle and the cart after it. Its gains suit the
default settings.

    python -m evolvarium.games.cartpole.scripted

plays every policy on the held-out seeds and prints how each did.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from statistics import fmean as mean

from ...config import HeldOutConfig
from .engine import INPUTS, Cart
from .settings import CartPoleSettings

Policy = Callable[[Sequence[float], random.Random], float]

_AT = {name: index for index, name in enumerate(INPUTS)}


def still(view: Sequence[float], rng: random.Random) -> float:
    return 0.0


def wander(view: Sequence[float], rng: random.Random) -> float:
    return rng.uniform(-1.0, 1.0)


def lean(view: Sequence[float], rng: random.Random) -> float:
    """Full force the way the pole leans."""
    return 1.0 if view[_AT["pole_angle"]] > 0 else -1.0


def upright(view: Sequence[float], rng: random.Random) -> float:
    """In proportion to the pole's lean and spin, wherever the cart is."""
    return 1.0 * view[_AT["pole_angle"]] + 1.2 * view[_AT["pole_spin"]]


def steady(view: Sequence[float], rng: random.Random) -> float:
    """Upright, and toward where the cart has drifted and the way it is rolling."""
    return (upright(view, rng)
            + 0.7 * view[_AT["cart_position"]] + 0.5 * view[_AT["cart_speed"]])


POLICIES: dict[str, Policy] = {
    "still": still,
    "wander": wander,
    "lean": lean,
    "upright": upright,
    "steady": steady,
}


def play(settings: CartPoleSettings, policy: Policy, seed: int) -> Cart:
    """One episode with `policy` pushing the cart."""
    cart = Cart(settings, seed)
    rng = random.Random(seed ^ 0x5EED)
    while not cart.finished:
        cart.step(policy(cart.observe(), rng) * settings.cart.force)
    return cart


def main(argv: list[str]) -> int:
    held_out = HeldOutConfig()
    seeds = range(held_out.first_seed, held_out.first_seed + held_out.episodes)
    settings = CartPoleSettings()
    print(f"{'policy':<8} {'steps':>7} {'balanced':>9} {'drift':>6} {'lean':>6} "
          f"{'effort':>7}  ends")
    for name, policy in POLICIES.items():
        carts = [play(settings, policy, seed) for seed in seeds]
        ends = Counter(c.end.value for c in carts)
        print(
            f"{name:<8} {mean(c.balanced for c in carts):>7.1f} "
            f"{ends['balanced']:>3} of {len(carts)} "
            f"{mean(c.drift for c in carts):>6.3f} "
            f"{mean(c.lean for c in carts):>6.3f} "
            f"{mean(c.effort for c in carts):>7.3f}  "
            + ", ".join(f"{end} {n}" for end, n in ends.most_common())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
