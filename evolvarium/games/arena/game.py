"""The arena, described to the engine through the game contract."""

from __future__ import annotations

from ...contract import (
    Actions,
    Choice,
    Game,
    Label,
    Layer,
    Milestone,
    Observation,
    Problems,
    Scene,
)
from .engine import INPUTS, MOVES, Match, pillar_sites
from .settings import ArenaSettings

OBSERVATION = Observation(INPUTS)
ACTIONS = Actions(Choice("move", tuple(MOVES)))
# A hurt agent is drawn apart from the rest of the team, and a fallen one stays
# where it fell.
_LAYERS = (
    Layer("pillars", "cell", "wall", size=0.92),
    Layer("food", "cell", "goal", size=0.56, rounded=True),
    Layer("packs", "cell", "bonus", size=0.7),
    Layer("fallen", "cell", "ink", size=0.4, rounded=True),
    Layer("rival", "cell", "hazard", size=0.9, rounded=True),
    Layer("team", "cell", "agent", size=0.84),
    Layer("hurt", "cell", "alert", size=0.84),
)


class ArenaGame(Game):
    """A team with health on a floor of pillars, eating, healing and chasing a rival.

    Every agent plays the same network, and the team shares one score.
    """

    name = "arena"
    Settings = ArenaSettings
    workload_label = "agents standing"
    observation = OBSERVATION
    actions = ACTIONS

    def __init__(self, settings: ArenaSettings | None = None):
        super().__init__(settings)
        self.agents = tuple(f"agent{n}" for n in range(1, self.settings.team.size + 1))
        self._index = {agent: index for index, agent in enumerate(self.agents)}
        self.match: Match | None = None

    # -------------------------------------------------------------- episode

    def reset(self, seed: int) -> None:
        self.match = Match(self.settings, seed)

    def observe(self, agent: str):
        return self.match.observe(self._index[agent])

    def step(self, actions: dict) -> None:
        self.match.step([actions[agent] for agent in self.agents])

    @property
    def over(self) -> bool:
        return self.match.finished

    @property
    def end(self) -> str:
        return self.match.end.value

    @property
    def score(self) -> int:
        return self.match.score

    @property
    def steps(self) -> int:
        return self.match.steps

    # -------------------------------------------------------------- results

    def goal(self) -> tuple[float, dict[str, float]]:
        """The score, less a share of it for every agent that fell.

        The terms add up to the fitness.
        """
        match, settings = self.match, self.settings
        points = settings.points
        fallen = (len(match.team) - match.standing) / len(match.team)
        terms = {
            "food": float(match.eaten * points.food),
            "catches": float(match.catches * points.catch),
            "falls": 0.0 - settings.fitness.fall_fraction * fallen * match.score,
        }
        return sum(terms.values()), terms

    def signals(self) -> dict[str, float]:
        match = self.match
        return {
            "food": float(match.eaten),
            "catches": float(match.catches),
            "healed": float(match.healed),
            "struck": float(match.strikes),
            "fallen": float(len(match.team) - match.standing),
        }

    def scene(self) -> Scene:
        floor = self.settings.floor
        return Scene(floor.width, floor.height, _LAYERS,
                     [Label("score"), Label("catches"), Label("standing")])

    def draw(self) -> dict:
        match = self.match
        team, hurt, fallen = [], [], []
        for member in match.team:
            if not member.standing:
                fallen.append(member.cell)
            elif match.hurt(member):
                hurt.append(member.cell)
            else:
                team.append(member.cell)
        return {
            "pillars": sorted(match.pillars),
            "food": list(match.food),
            "packs": [cell for cell in match.packs if cell is not None],
            "fallen": fallen,
            "rival": [match.rival],
            "team": team,
            "hurt": hurt,
            "score": match.score,
            "catches": match.catches,
            "standing": match.standing,
        }

    def milestones(self, summaries: list[dict]) -> list[Milestone]:
        return milestones(summaries)

    # -------------------------------------------------------------- profiling

    def workload(self) -> int:
        return self.match.standing

    # -------------------------------------------------------------- settings

    @classmethod
    def setting_problems(cls, settings: ArenaSettings) -> list[str]:
        floor, team, rival = settings.floor, settings.team, settings.rival
        pickups, points, match = settings.pickups, settings.points, settings.match
        problems = Problems()
        problems.at_least("floor.width", floor.width, 5)
        problems.at_least("floor.height", floor.height, 5)
        sites = len(pillar_sites(floor.width, floor.height))
        problems.between("floor.pillars", floor.pillars, 0, sites)
        problems.at_least("team.size", team.size, 1)
        problems.at_least("team.health", team.health, 1)
        # Agents start at full health, so at the top of that range any damage
        # at all leaves one hurt.
        problems.between("team.hurt_at", team.hurt_at, 0, team.health - 1)
        for name in ("damage", "reach", "sight", "rest_every", "respawn_distance"):
            problems.at_least(f"rival.{name}", getattr(rival, name), 0)
        for name in ("food", "packs", "heal", "pack_delay"):
            problems.at_least(f"pickups.{name}", getattr(pickups, name), 0)
        problems.at_least("points.food", points.food, 0)
        problems.at_least("points.catch", points.catch, 0)
        problems.at_least("match.max_steps", match.max_steps, 1)
        problems.at_least("match.idle_limit", match.idle_limit, 1)
        problems.between("fitness.fall_fraction", settings.fitness.fall_fraction, 0, 1)
        if not problems:
            # Pickups go only where nothing else is, so the open floor has to
            # hold the team, the rival and every pickup at once.
            room = floor.width * floor.height - floor.pillars
            needed = team.size + 1 + pickups.food + pickups.packs
            if needed > room:
                problems.append(
                    f"team.size, pickups.food and pickups.packs need {needed} open cells "
                    f"with the rival, and the floor has {room}"
                )
        return problems

    @classmethod
    def unused_settings(cls, settings: ArenaSettings) -> dict[str, str]:
        unused = {}
        if settings.pickups.packs == 0:
            unused["pickups.heal"] = "pickups.packs is 0"
            unused["pickups.pack_delay"] = "pickups.packs is 0"
        if settings.pickups.food == 0:
            unused["points.food"] = "pickups.food is 0"
        # Without damage no agent is ever hurt or falls, so nothing that hunts,
        # marks or heals a hurt agent comes into play, and no fall is charged.
        # The packs still lie on the floor.
        if settings.rival.damage == 0:
            for name in ("rival.reach", "rival.sight", "team.hurt_at",
                         "pickups.heal", "pickups.pack_delay", "fitness.fall_fraction"):
                unused[name] = "rival.damage is 0, so no agent loses health"
        return unused


# ---------------------------------------------------------------- milestones

_SCORE_TIERS = (
    (1, "First point", "The team scored at least once"),
    (10, "Double digits", "Reached a score of 10"),
    (50, "Half century", "Reached a score of 50"),
    (100, "Century", "Reached a score of 100"),
    (200, "Double century", "Reached a score of 200"),
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

    The champion's signals are its means over the matches it played, so a
    champion with no fallen agents lost none in any of them.
    """
    found: list[Milestone] = []
    for threshold, label, detail in _SCORE_TIERS:
        gen = _first_generation_where(
            summaries, lambda e, t=threshold: e.get("max_episode_score", 0) >= t
        )
        found.append(Milestone(f"score_{threshold}", label, detail, gen))

    found.append(Milestone(
        "first_catch", "First catch", "A champion's team caught the rival",
        _first_generation_where(summaries, lambda e: _champion(e)["catches"] > 0),
    ))
    found.append(Milestone(
        "patched_up", "Patched up", "A champion's team used a health pack",
        _first_generation_where(summaries, lambda e: _champion(e)["healed"] > 0),
    ))
    found.append(Milestone(
        "regular_catches", "Regular catches", "A champion's team averaged a catch a match",
        _first_generation_where(summaries, lambda e: _champion(e)["catches"] >= 1),
    ))
    found.append(Milestone(
        "nobody_left", "Nobody left behind",
        "A champion's team averaged a catch a match and lost no one in any",
        _first_generation_where(
            summaries,
            lambda e: _champion(e)["fallen"] == 0 and _champion(e)["catches"] >= 1,
        ),
    ))
    found.append(Milestone(
        "majority_scoring", "Majority scoring",
        "Over half the population scored in one generation",
        _first_generation_where(summaries, lambda e: e.get("scoring_fraction", 0) >= 0.5),
    ))
    return found
