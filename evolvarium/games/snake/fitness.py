"""Fitness scoring for a single episode.

Kept as a free function over an `EpisodeResult` so it can be unit-tested without
constructing a population, and so the weighting is visible in one place.

Food is superlinear and dominant. The remaining terms only shape behaviour among
agents that score equally: they must never add up to more than one food's worth,
or agents will farm the shaping terms instead of eating. `terms()` returns the
breakdown so the dashboard can show where a score came from.
"""

from __future__ import annotations

import numpy as np

from .engine import DeathCause, EpisodeResult
from .settings import BoardConfig, FitnessConfig


def evaluate(
    result: EpisodeResult,
    config: FitnessConfig,
    board: BoardConfig,
) -> tuple[float, dict[str, float]]:
    """Return (fitness, term breakdown)."""
    terms: dict[str, float] = {}

    # Superlinear in food so that going from four to five is worth more than
    # going from zero to one, which keeps pressure on once eating is solved.
    terms["food"] = config.food_reward * (result.score**config.food_exponent)

    # Bounded, so surviving without eating can never rival eating.
    terms["survival"] = min(
        result.steps * config.survival_per_step,
        config.survival_cap,
    )

    # Rewards net movement toward food. This is the gradient that gets a
    # population off zero before any individual has eaten by accident. Clamped
    # symmetrically so it cannot be farmed past the value of a single food.
    terms["approach"] = float(
        np.clip(result.approach * config.approach_reward, -config.approach_cap, config.approach_cap)
    )

    terms["coverage"] = (result.cells_visited / board.cell_count) * config.efficiency_reward

    penalties = 0.0
    if result.death is DeathCause.STARVATION:
        penalties -= config.starvation_penalty
    if result.steps < config.early_death_steps and result.death in (
        DeathCause.WALL,
        DeathCause.SELF,
    ):
        penalties -= config.early_death_penalty
    if result.looped and result.score == 0:
        penalties -= config.loop_penalty
    terms["penalties"] = penalties

    # Charged against what the episode had already earned, so the cost of dying
    # grows with the score. A flat penalty would be negligible next to a food
    # term worth tens of thousands, and a penalty charged to every episode
    # equally would only rescale food rather than change what selection prefers.
    stake = terms["food"]
    death = 0.0
    if config.avoidable_death_fraction and result.avoidable_death:
        death -= config.avoidable_death_fraction * stake
    if config.death_fraction and result.death in (
        DeathCause.WALL,
        DeathCause.SELF,
        DeathCause.STARVATION,
    ):
        death -= config.death_fraction * stake
    terms["death"] = death

    total = sum(terms.values())
    # A floor keeps fitness positive for selection schemes that treat it as a
    # weight, without compressing the differences between real performers.
    return max(total, 1.0), terms
