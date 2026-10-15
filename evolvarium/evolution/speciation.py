"""Genome clustering with fitness sharing.

Disabled by default. Euclidean distance between flat weight vectors is a weak
similarity measure for neural genomes: in high dimensions the distances between
unrelated networks concentrate, so clusters drift toward being arbitrary. It is
kept because it is cheap to try and occasionally helps preserve a promising
variant, but it is opt-in rather than always-on.

Unlike a plain cluster count, this implementation actually applies the sharing
it computes: `apply_sharing` divides each member's fitness by its cluster size,
which is the mechanism that stops one cluster taking over the population.
"""

from __future__ import annotations

import numpy as np

from .individual import Individual


class Species:
    __slots__ = ("id", "representative", "members", "best_fitness", "stagnation")

    def __init__(self, species_id: int, representative: np.ndarray):
        self.id = species_id
        self.representative = representative.copy()
        self.members: list[Individual] = []
        self.best_fitness = 0.0
        self.stagnation = 0


def assign(population: list[Individual], species: list[Species], threshold: float,
           next_id: int) -> tuple[list[Species], int]:
    """Cluster the population, returning live clusters and the next free id."""
    for group in species:
        group.members = []

    for individual in population:
        placed = False
        for group in species:
            if float(np.linalg.norm(individual.genome - group.representative)) < threshold:
                group.members.append(individual)
                individual.species_id = group.id
                placed = True
                break
        if not placed:
            group = Species(next_id, individual.genome)
            next_id += 1
            group.members.append(individual)
            individual.species_id = group.id
            species.append(group)

    live = [g for g in species if g.members]
    for group in live:
        best = max(group.members, key=lambda i: i.fitness)
        group.representative = best.genome.copy()
        if best.fitness > group.best_fitness:
            group.best_fitness = best.fitness
            group.stagnation = 0
        else:
            group.stagnation += 1
    return live, next_id


def apply_sharing(species: list[Species]) -> dict[str, float]:
    """Shared fitness per individual: raw fitness divided by cluster size.

    Dividing shrinks a positive figure but lifts a negative one toward zero,
    which would favour the largest cluster. When any fitness is negative, each
    is measured up from the lowest before it is divided.
    """
    lowest = min((m.fitness for group in species for m in group.members), default=0.0)
    floor = min(lowest, 0.0)
    shared: dict[str, float] = {}
    for group in species:
        size = max(len(group.members), 1)
        for member in group.members:
            shared[member.id] = (member.fitness - floor) / size
    return shared


def adjust_threshold(threshold: float, count: int, target: int) -> float:
    """Nudge the distance threshold toward the requested cluster count."""
    if count > target:
        return min(threshold * 1.05, 100.0)
    if count < max(2, target // 2):
        return max(threshold * 0.95, 0.25)
    return threshold
