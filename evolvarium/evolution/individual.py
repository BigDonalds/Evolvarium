"""Population member.

An individual carries a genome and the result of its most recent evaluation.
Nothing about a past evaluation survives into a new generation: `reset_scores`
is called on every member at the start of each generation, and `descendant`
deliberately does not copy fitness or score.

That rule matters more than it looks. If a member could keep the best figure it
ever recorded, a long-lived elite would accumulate many more evaluation attempts
than a newcomer and win selection on sampling luck alone, regardless of whether
its policy was any better.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

_ids = itertools.count(1)


def reserve_ids(highest: int) -> None:
    """Start new ids above `highest`.

    Members restored from a checkpoint keep the ids they were saved with, and
    the history already refers to them by those ids. A fresh process counts from
    1 again, so without this a new member could be handed an id that already
    belongs to someone in the run.
    """
    global _ids
    _ids = itertools.count(max(next(_ids), highest + 1))


def id_number(individual_id: str) -> int:
    return int(individual_id.removeprefix("ind"))


@dataclass
class Individual:
    genome: np.ndarray
    generation: int = 0
    # Generations this lineage has survived, for display only. It never
    # influences selection.
    age: int = 0
    origin: str = "seed"
    parents: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: f"ind{next(_ids):06d}")

    fitness: float = 0.0
    score: float = 0.0
    best_episode_score: float = 0
    steps: float = 0.0
    # How the member's worst episode ended, in the game's own words.
    end: str | None = None
    terms: dict[str, float] = field(default_factory=dict)
    signals: dict[str, float] = field(default_factory=dict)
    species_id: int = 0
    evaluated: bool = False
    # Packed replay of this individual's first evaluation episode, when one was
    # recorded. Produced in whichever process scored the genome.
    replay: dict | None = None

    def reset_scores(self) -> None:
        """Clear every evaluation-derived field before a new generation."""
        self.fitness = 0.0
        self.score = 0.0
        self.best_episode_score = 0
        self.steps = 0.0
        self.end = None
        self.terms = {}
        self.signals = {}
        self.replay = None
        self.evaluated = False

    def descendant(self, genome: np.ndarray, origin: str, generation: int,
                   parents: tuple[str, ...] = ()) -> Individual:
        """A new individual from this one's line, with no inherited results."""
        return Individual(
            genome=genome,
            generation=generation,
            age=self.age + 1 if origin == "elite" else 0,
            origin=origin,
            parents=parents or (self.id,),
        )

    def summary(self) -> dict:
        return {
            "id": self.id,
            "generation": self.generation,
            "age": self.age,
            "origin": self.origin,
            "fitness": round(self.fitness, 2),
            "score": round(self.score, 3),
            "best_episode_score": self.best_episode_score,
            "steps": round(self.steps, 1),
            "end": self.end,
            "terms": {k: round(v, 2) for k, v in self.terms.items()},
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "species": self.species_id,
        }
