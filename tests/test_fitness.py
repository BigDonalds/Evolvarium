"""Fitness shaping.

The dominance test is the one that matters. Shaping terms exist to give a
gradient to agents that have not eaten yet; if they can add up to more than a
single food is worth, a population will farm them instead of learning to eat.
"""

from __future__ import annotations

import pytest

from evolvarium.config import CONFIG
from evolvarium.evolution.fitness import evaluate
from evolvarium.game.engine import DeathCause, EpisodeResult


def result(**fields) -> EpisodeResult:
    base = EpisodeResult(
        score=0, steps=50, death=DeathCause.WALL, death_cell=(0, 0),
        final_length=3, cells_visited=20, approach=0.0, steps_since_food=0,
    )
    for key, value in fields.items():
        setattr(base, key, value)
    return base


def score_of(**fields) -> float:
    fitness, _ = evaluate(result(**fields), CONFIG.fitness, CONFIG.board)
    return fitness


def test_fitness_is_monotonic_in_food():
    scores = [score_of(score=n) for n in range(6)]
    assert scores == sorted(scores)
    assert all(b > a for a, b in zip(scores, scores[1:], strict=False))


def test_food_is_superlinear():
    """Later food must be worth more than earlier food, to keep pressure on."""
    steps = [score_of(score=n + 1) - score_of(score=n) for n in range(5)]
    assert all(b > a for a, b in zip(steps, steps[1:], strict=False))


def test_one_food_outweighs_every_shaping_term():
    starved = result(
        score=0, steps=CONFIG.episode.max_steps, cells_visited=CONFIG.board.cell_count,
        approach=500.0,
    )
    best_without_food, _ = evaluate(starved, CONFIG.fitness, CONFIG.board)
    with_one_food, _ = evaluate(result(score=1, steps=20, cells_visited=6),
                                CONFIG.fitness, CONFIG.board)
    assert with_one_food > best_without_food


def test_survival_term_is_capped():
    # Both samples sit above early_death_steps so only the survival term differs.
    short = score_of(score=0, steps=30, approach=0.0)
    long = score_of(score=0, steps=CONFIG.episode.max_steps, approach=0.0)
    assert long - short <= CONFIG.fitness.survival_cap + 1e-6


def test_approaching_food_beats_retreating():
    assert score_of(approach=25.0) > score_of(approach=-25.0)


def test_early_death_is_penalised():
    early = score_of(steps=3, death=DeathCause.WALL)
    later = score_of(steps=60, death=DeathCause.WALL)
    assert later > early


def test_starvation_is_penalised_relative_to_a_clean_death():
    starved = score_of(steps=200, death=DeathCause.STARVATION)
    collided = score_of(steps=200, death=DeathCause.WALL)
    assert collided > starved


def test_looping_without_eating_is_penalised():
    assert score_of(score=0, looped=False) > score_of(score=0, looped=True)


def test_looping_is_not_penalised_once_eating():
    """A long snake circling to stay alive is behaving sensibly."""
    a, _ = evaluate(result(score=5, looped=True), CONFIG.fitness, CONFIG.board)
    b, _ = evaluate(result(score=5, looped=False), CONFIG.fitness, CONFIG.board)
    assert a == pytest.approx(b)


def test_fitness_never_drops_below_the_floor():
    worst = result(score=0, steps=1, approach=-1000.0, death=DeathCause.WALL, cells_visited=1)
    fitness, _ = evaluate(worst, CONFIG.fitness, CONFIG.board)
    assert fitness >= 1.0


def test_terms_sum_to_the_total_above_the_floor():
    outcome = result(score=3, steps=140, approach=30.0, cells_visited=40)
    fitness, terms = evaluate(outcome, CONFIG.fitness, CONFIG.board)
    assert sum(terms.values()) == pytest.approx(fitness)


def _cfg(**changes):
    import dataclasses
    return dataclasses.replace(CONFIG.fitness, **changes)


def test_death_charges_are_off_by_default():
    """Every run measured so far assumed these were zero."""
    assert CONFIG.fitness.avoidable_death_fraction == 0.0
    assert CONFIG.fitness.death_fraction == 0.0
    _, terms = evaluate(result(score=10, avoidable_death=True), CONFIG.fitness, CONFIG.board)
    assert terms["death"] == 0.0


def test_avoidable_death_costs_more_than_being_boxed_in():
    cfg = _cfg(avoidable_death_fraction=0.25)
    avoidable, _ = evaluate(result(score=10, avoidable_death=True), cfg, CONFIG.board)
    boxed, _ = evaluate(result(score=10, avoidable_death=False), cfg, CONFIG.board)
    assert avoidable < boxed


def test_the_charge_scales_with_what_was_at_stake():
    """Dying at 46 must cost more than dying at 5, or it changes nothing."""
    cfg = _cfg(avoidable_death_fraction=0.25)
    _, low = evaluate(result(score=5, avoidable_death=True), cfg, CONFIG.board)
    _, high = evaluate(result(score=46, avoidable_death=True), cfg, CONFIG.board)
    assert abs(high["death"]) > abs(low["death"]) * 5


def test_running_out_the_clock_is_exempt_from_the_death_charge():
    """The charge must reward staying alive, not merely reaching a score."""
    cfg = _cfg(death_fraction=0.3)
    died, _ = evaluate(result(score=20, death=DeathCause.SELF), cfg, CONFIG.board)
    survived, _ = evaluate(result(score=20, death=DeathCause.STEP_LIMIT), cfg, CONFIG.board)
    assert survived > died

    filled, _ = evaluate(result(score=20, death=DeathCause.BOARD_FILLED), cfg, CONFIG.board)
    assert filled > died


def test_death_charge_applies_to_starvation_too():
    cfg = _cfg(death_fraction=0.3)
    starved, _ = evaluate(result(score=20, death=DeathCause.STARVATION), cfg, CONFIG.board)
    clocked, _ = evaluate(result(score=20, death=DeathCause.STEP_LIMIT), cfg, CONFIG.board)
    assert starved < clocked


def test_more_food_still_beats_dying_well():
    """The charges must not invert the objective: eating more has to win."""
    cfg = _cfg(avoidable_death_fraction=0.3, death_fraction=0.3)
    greedy, _ = evaluate(result(score=30, death=DeathCause.SELF, avoidable_death=True),
                         cfg, CONFIG.board)
    timid, _ = evaluate(result(score=10, death=DeathCause.STEP_LIMIT), cfg, CONFIG.board)
    assert greedy > timid


def test_terms_still_sum_to_the_total():
    cfg = _cfg(avoidable_death_fraction=0.2, death_fraction=0.1)
    fitness, terms = evaluate(result(score=8, steps=400, avoidable_death=True), cfg, CONFIG.board)
    assert sum(terms.values()) == pytest.approx(fitness)
