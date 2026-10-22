"""Cart-pole, described to the engine through the game contract."""

from __future__ import annotations

import math

from ...contract import (
    Actions,
    Game,
    Label,
    Layer,
    Milestone,
    Observation,
    Problems,
    Range,
    Scene,
)
from .engine import INPUTS, Cart, EndCause
from .settings import CartPoleSettings

OBSERVATION = Observation(INPUTS)
# How the cart is drawn, in metres. The motion treats the cart as a point on
# the track, so none of these change it.
_CART_WIDTH = 0.5
_CART_HEIGHT = 0.3
_CLEARANCE = 0.05
_HINGE = _CLEARANCE + _CART_HEIGHT
_MARGIN = 0.15
# A pole that has fallen is drawn apart from one still up.
_LAYERS = (
    Layer("track", "line", "wall", size=0.03),
    Layer("stops", "line", "hazard", size=0.05),
    Layer("cart", "box", "agent"),
    Layer("pole", "line", "ink", size=0.08),
    Layer("fallen", "line", "alert", size=0.08),
)


class CartPoleGame(Game):
    """A pole hinged to a cart, kept upright by pushing the cart along a track.

    The network gives the push itself, as a force between the cart's hardest
    push either way, rather than choosing among a few moves.
    """

    name = "cartpole"
    Settings = CartPoleSettings
    observation = OBSERVATION

    def __init__(self, settings: CartPoleSettings | None = None):
        super().__init__(settings)
        force = self.settings.cart.force
        self.actions = Actions(Range("force", -force, force))
        self.cart: Cart | None = None

    # -------------------------------------------------------------- episode

    def reset(self, seed: int) -> None:
        self.cart = Cart(self.settings, seed)

    def observe(self, agent: str):
        return self.cart.observe()

    def step(self, actions: dict) -> None:
        self.cart.step(actions["agent"])

    @property
    def over(self) -> bool:
        return self.cart.finished

    @property
    def end(self) -> str:
        return self.cart.end.value

    @property
    def score(self) -> int:
        return self.cart.balanced

    @property
    def steps(self) -> int:
        return self.cart.steps

    # -------------------------------------------------------------- results

    def signals(self) -> dict[str, float]:
        cart = self.cart
        return {
            "balanced": 1.0 if cart.end is EndCause.BALANCED else 0.0,
            "drift": cart.drift,
            "lean": cart.lean,
            "effort": cart.effort,
        }

    def scene(self) -> Scene:
        cart, pole = self.settings.cart, self.settings.pole
        # Room for the cart at either end of the track and for the pole leaning
        # as far as it can without having fallen.
        lean = pole.length * math.sin(math.radians(pole.fall_angle))
        side = cart.track + max(_CART_WIDTH / 2, lean) + _MARGIN
        return Scene(2 * side, _HINGE + pole.length + 2 * _MARGIN, _LAYERS,
                     [Label("steps"), Label("angle", 1), Label("position", 2)],
                     origin=(-side, -_MARGIN), y_up=True, unit=0.005)

    def draw(self) -> dict:
        cart, length = self.cart, self.settings.pole.length
        # The stops stand half a cart beyond the track's ends, so the cart
        # touches one just as its centre reaches the end.
        end = cart.track + _CART_WIDTH / 2
        pole = [(cart.x, _HINGE,
                 cart.x + length * math.sin(cart.angle), _HINGE + length * math.cos(cart.angle))]
        fell = cart.end is EndCause.FELL
        return {
            "track": [(-end, 0.0, end, 0.0)],
            "stops": [(-end, 0.0, -end, _CART_HEIGHT), (end, 0.0, end, _CART_HEIGHT)],
            "cart": [(cart.x - _CART_WIDTH / 2, _CLEARANCE, _CART_WIDTH, _CART_HEIGHT)],
            "pole": [] if fell else pole,
            "fallen": pole if fell else [],
            "steps": cart.balanced,
            "angle": math.degrees(cart.angle),
            "position": cart.x,
        }

    def milestones(self, summaries: list[dict]) -> list[Milestone]:
        return milestones(summaries)

    # -------------------------------------------------------------- settings

    @classmethod
    def setting_problems(cls, settings: CartPoleSettings) -> list[str]:
        cart, pole = settings.cart, settings.pole
        world, episode = settings.world, settings.episode
        problems = Problems()
        for name in ("mass", "force", "track"):
            problems.above(f"cart.{name}", getattr(cart, name), 0)
        problems.above("pole.mass", pole.mass, 0)
        problems.above("pole.length", pole.length, 0)
        if not 0 < pole.fall_angle <= 90:
            problems.append(
                f"pole.fall_angle must be above 0 and at most 90, not {pole.fall_angle}"
            )
        problems.at_least("world.gravity", world.gravity, 0)
        problems.above("world.step_time", world.step_time, 0)
        problems.at_least("episode.max_steps", episode.max_steps, 1)
        problems.at_least("episode.start_spread", episode.start_spread, 0)
        if not problems:
            # A start drawn any wider could have the pole down or the cart off
            # the track before the first push.
            widest = min(cart.track, math.radians(pole.fall_angle))
            if episode.start_spread >= widest:
                problems.append(
                    f"episode.start_spread must be below {widest:.4g}, the smaller of "
                    f"cart.track and pole.fall_angle in radians, not {episode.start_spread}"
                )
        return problems


# ---------------------------------------------------------------- milestones

_STEP_TIERS = (
    (50, "Fifty steps", "Kept the pole up for 50 steps"),
    (100, "A hundred steps", "Kept the pole up for 100 steps"),
    (250, "Two hundred and fifty", "Kept the pole up for 250 steps"),
)


def _first_generation_where(summaries: list[dict], predicate) -> int | None:
    for entry in summaries:
        try:
            if predicate(entry):
                return int(entry["generation"])
        except (KeyError, TypeError):
            continue
    return None


def _champion(entry: dict) -> dict:
    return entry["best_individual"]["signals"]


def milestones(summaries: list[dict]) -> list[Milestone]:
    """Milestones reached at or before the last summary supplied.

    The champion's signals are its means over the episodes it played, so a
    champion whose `balanced` is 1 went the full length in every one of them.
    """
    found: list[Milestone] = []
    for threshold, label, detail in _STEP_TIERS:
        gen = _first_generation_where(
            summaries, lambda e, t=threshold: e.get("max_episode_score", 0) >= t
        )
        found.append(Milestone(f"steps_{threshold}", label, detail, gen))

    found.append(Milestone(
        "full_length", "Full length", "A champion kept the pole up to the end of an episode",
        _first_generation_where(summaries, lambda e: _champion(e)["balanced"] > 0),
    ))
    found.append(Milestone(
        "every_time", "Every time",
        "A champion kept the pole up to the end of every episode it played",
        _first_generation_where(summaries, lambda e: _champion(e)["balanced"] >= 1),
    ))
    found.append(Milestone(
        "near_the_middle", "Near the middle",
        "A champion went the full length every time and kept the cart, on average, "
        "within a tenth of the track of the middle",
        _first_generation_where(
            summaries,
            lambda e: _champion(e)["balanced"] >= 1 and _champion(e)["drift"] <= 0.1,
        ),
    ))
    return found
