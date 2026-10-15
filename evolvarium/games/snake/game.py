"""Snake, described to the engine through the game contract."""

from __future__ import annotations

import dataclasses

from ...contract import Actions, Choice, Game, Milestone, Observation, load_settings
from .board import Turn
from .engine import Episode
from .features import FeatureSpec, available_specs
from .fitness import evaluate
from .replay import pack_replay
from .settings import BoardConfig, EpisodeConfig, FeatureConfig, FitnessConfig, SnakeSettings

ACTIONS = Actions(Choice("turn", ("left", "straight", "right")))
_TURNS = {"left": Turn.LEFT, "straight": Turn.STRAIGHT, "right": Turn.RIGHT}


class SnakeGame(Game):
    """One snake on a grid, turning left, right or carrying on each step."""

    name = "snake"
    Settings = SnakeSettings
    workload_label = "snake length"

    def __init__(self, settings: SnakeSettings | None = None):
        super().__init__(settings)
        features = self.settings.features
        self.spec = FeatureSpec(features.spec, features)
        self.observation = Observation(self.spec.names)
        self.actions = ACTIONS
        self.episode: Episode | None = None
        # Bound once: observe runs every step of every game.
        self._encode = self.spec._encode
        self._features = features

    # -------------------------------------------------------------- episode

    def reset(self, seed: int, *, record: bool = False) -> None:
        self.episode = Episode(self.settings.board, self.settings.episode, seed, record=record)

    def observe(self, agent: str):
        return self._encode(self.episode, self._features)

    def step(self, actions: dict) -> None:
        self.episode.step(_TURNS[actions["agent"]])

    @property
    def over(self) -> bool:
        return self.episode.finished

    @property
    def end(self) -> str:
        return self.episode.result.death.value

    @property
    def score(self) -> int:
        return self.episode.score

    @property
    def steps(self) -> int:
        return self.episode.steps

    # -------------------------------------------------------------- results

    def goal(self) -> tuple[float, dict[str, float]]:
        return evaluate(self.episode.result, self.settings.fitness, self.settings.board)

    def signals(self) -> dict[str, float]:
        result = self.episode.result
        return {
            "length": float(result.final_length),
            "cells_visited": float(result.cells_visited),
            "avoidable_death": float(result.avoidable_death),
            "looped": float(result.looped),
        }

    def replay(self, max_steps: int) -> dict:
        return pack_replay(self.episode.result, max_steps)

    def scene(self) -> dict:
        board = self.settings.board
        return {"kind": "grid", "width": board.width, "height": board.height}

    def milestones(self, summaries: list[dict]) -> list[Milestone]:
        return milestones(summaries)

    # -------------------------------------------------------------- profiling

    def workload(self) -> int:
        return len(self.episode.snake)

    def encoder(self):
        return self.spec._encode

    # -------------------------------------------------------------- model file

    def model_inputs(self) -> dict:
        settings = dataclasses.asdict(self.settings.features)
        return {"encoding": settings.pop("spec"), "settings": settings}

    def model_rules(self) -> dict:
        return {
            "board": dataclasses.asdict(self.settings.board),
            "episode": dataclasses.asdict(self.settings.episode),
        }

    @classmethod
    def settings_from_model(cls, document: dict) -> SnakeSettings:
        game = document.get("game") or {}
        inputs = document.get("inputs") or {}
        encoding = inputs.get("encoding")
        if encoding not in available_specs():
            raise ValueError(f"snake has no {encoding!r} inputs")
        features = load_settings(FeatureConfig, inputs.get("settings"), strict=False)
        return SnakeSettings(
            board=load_settings(BoardConfig, game.get("board"), strict=False),
            episode=load_settings(EpisodeConfig, game.get("episode"), strict=False),
            features=dataclasses.replace(features, spec=encoding),
            # A model file does not record fitness weights, and playing a game
            # does not read them.
            fitness=FitnessConfig(),
        )


# ---------------------------------------------------------------- milestones

_SCORE_TIERS = (
    (1, "First bite", "An agent ate at least once"),
    (5, "Five course", "Reached a score of 5"),
    (10, "Double digits", "Reached a score of 10"),
    (20, "Twenty", "Reached a score of 20"),
    (35, "Coiled", "Reached a score of 35"),
    (50, "Half century", "Reached a score of 50"),
)


def _first_generation_where(summaries: list[dict], predicate) -> int | None:
    for entry in summaries:
        try:
            if predicate(entry):
                return int(entry["generation"])
        except (KeyError, TypeError):
            continue
    return None


def _ends(entry: dict) -> dict:
    # Summaries written before games named their own endings called them deaths.
    return entry.get("ends", entry.get("deaths", {}))


def milestones(summaries: list[dict]) -> list[Milestone]:
    """Milestones reached at or before the last summary supplied."""
    found: list[Milestone] = []
    for threshold, label, detail in _SCORE_TIERS:
        gen = _first_generation_where(
            summaries, lambda e, t=threshold: e.get("max_episode_score", 0) >= t
        )
        found.append(Milestone(f"score_{threshold}", label, detail, gen))

    found.append(Milestone(
        "majority_feeding", "Majority feeding",
        "Over half the population ate in one generation",
        _first_generation_where(summaries, lambda e: e.get("scoring_fraction", 0) >= 0.5),
    ))
    found.append(Milestone(
        "population_feeding", "Whole population feeding",
        "Nine in ten agents ate in one generation",
        _first_generation_where(summaries, lambda e: e.get("scoring_fraction", 0) >= 0.9),
    ))
    found.append(Milestone(
        "endurance", "Endurance", "Mean survival passed 100 steps",
        _first_generation_where(summaries, lambda e: e.get("mean_steps", 0) >= 100),
    ))
    found.append(Milestone(
        "wall_wise", "Wall wise", "A generation passed with no wall deaths",
        _first_generation_where(
            summaries, lambda e: _ends(e).get("wall", 1) == 0 and sum(_ends(e).values()) > 0
        ),
    ))
    found.append(Milestone(
        "consistent", "Consistent", "Mean score across the population reached 1.0",
        _first_generation_where(summaries, lambda e: e.get("mean_score", 0) >= 1.0),
    ))
    return found
