"""Parent selection.

Every scheme reads `fitness` from the current generation only. Because
`Individual.reset_scores` runs before each generation is evaluated, no member
can be carrying a figure it earned under an earlier one.
"""

from __future__ import annotations

import numpy as np

from .individual import Individual

METHODS = ("tournament", "rank", "roulette")


def tournament(population: list[Individual], rng: np.random.Generator, size: int) -> Individual:
    size = max(2, min(size, len(population)))
    picks = rng.choice(len(population), size=size, replace=False)
    best = population[int(picks[0])]
    for index in picks[1:]:
        contender = population[int(index)]
        if contender.fitness > best.fitness:
            best = contender
    return best


def rank(population: list[Individual], rng: np.random.Generator) -> Individual:
    """Linear rank weighting: robust when a few fitness values dominate."""
    ordered = sorted(population, key=lambda i: i.fitness)
    weights = np.arange(1, len(ordered) + 1, dtype=np.float64)
    weights /= weights.sum()
    return ordered[int(rng.choice(len(ordered), p=weights))]


def roulette(population: list[Individual], rng: np.random.Generator) -> Individual:
    weights = np.array([i.fitness for i in population], dtype=np.float64)
    # A game that wants its score low has negative fitness throughout, so the
    # wheel is measured up from the lowest member rather than from zero.
    lowest = weights.min()
    if lowest < 0.0:
        weights -= lowest
    total = weights.sum()
    if total <= 0.0:
        return population[int(rng.integers(len(population)))]
    return population[int(rng.choice(len(population), p=weights / total))]


def select(population: list[Individual], rng: np.random.Generator, method: str,
           tournament_size: int) -> Individual:
    if method == "tournament":
        return tournament(population, rng, tournament_size)
    if method == "rank":
        return rank(population, rng)
    if method == "roulette":
        return roulette(population, rng)
    raise ValueError(f"unknown selection method {method!r}")
