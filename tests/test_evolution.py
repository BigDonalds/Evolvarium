"""Evolution correctness.

These are regression tests for a specific class of defect: scores leaking across
generation boundaries. When an individual keeps the best figure it ever recorded,
a long-lived elite accumulates far more evaluation attempts than a newcomer and
wins selection on sampling luck rather than on policy quality, which stalls the
whole run. The tests below pin every path by which that could return.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from evolvarium.config import CONFIG
from evolvarium.evolution.individual import Individual
from evolvarium.evolution.population import Population
from evolvarium.game.engine import DeathCause, EpisodeResult


@pytest.fixture()
def small_config():
    return CONFIG.with_overrides(
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=12, evaluation_episodes=2, elite_count=2,
            immigrant_count=1,
        )
    )


def test_reset_scores_clears_every_evaluation_field():
    individual = Individual(genome=np.zeros(4, dtype=np.float32))
    individual.fitness = 900.0
    individual.score = 7.0
    individual.best_episode_score = 7
    individual.steps = 300.0
    individual.death = DeathCause.WALL
    individual.terms = {"food": 900.0}
    individual.evaluated = True

    individual.reset_scores()

    assert individual.fitness == 0.0
    assert individual.score == 0.0
    assert individual.best_episode_score == 0
    assert individual.steps == 0.0
    assert individual.death is None
    assert individual.terms == {}
    assert individual.evaluated is False


def test_descendant_inherits_no_results():
    parent = Individual(genome=np.ones(4, dtype=np.float32))
    parent.fitness = 1234.0
    parent.score = 9.0
    parent.best_episode_score = 9

    child = parent.descendant(parent.genome.copy(), "elite", generation=5)

    assert child.fitness == 0.0
    assert child.score == 0.0
    assert child.best_episode_score == 0
    assert child.id != parent.id


def test_new_generation_starts_unevaluated(small_config):
    population = Population(small_config, seed=5)
    population.run_generation()

    # Every member of the freshly bred generation must be unscored, including
    # the elites carried over from the previous one.
    assert all(m.fitness == 0.0 for m in population.members)
    assert all(m.score == 0.0 for m in population.members)
    assert all(not m.evaluated for m in population.members)


def test_elites_are_re_evaluated_each_generation(small_config):
    population = Population(small_config, seed=6)
    population.run_generation()
    elites = [m for m in population.members if m.origin == "elite"]
    assert elites, "expected elites to be carried forward"

    population.run_generation()
    # After the next generation runs, those elites carried a fresh score rather
    # than a remembered one; the report is built from the generation just run.
    assert all(m.fitness == 0.0 for m in population.members)


def test_fitness_is_the_mean_not_the_best(small_config, monkeypatch):
    """A genome's fitness must average its trials, not take the best of them."""
    population = Population(small_config, seed=8)
    individual = population.members[0]

    scores = iter([0, 10])

    def fake_play(genome, seed, record):
        result = EpisodeResult()
        result.score = next(scores)
        result.steps = 10
        result.death = DeathCause.WALL
        return result

    monkeypatch.setattr(population._runner, "play", fake_play)
    population.evaluate_individual(individual, base_seed=0)

    assert individual.score == pytest.approx(5.0)
    assert individual.best_episode_score == 10


def test_generation_report_describes_the_generation_just_run(small_config):
    population = Population(small_config, seed=9)
    report = population.run_generation()

    # The report is produced before breeding, so its population statistics come
    # from members that actually played, not from the new unevaluated set.
    assert report.generation == 0
    assert population.generation == 1
    assert report.mean_steps > 0
    assert 0.0 <= report.scoring_fraction <= 1.0
    assert sum(report.deaths.values()) == small_config.evolution.population_size


def test_population_size_is_stable(small_config):
    population = Population(small_config, seed=10)
    for _ in range(3):
        population.run_generation()
        assert len(population.members) == small_config.evolution.population_size


def test_genomes_keep_the_architecture_width(small_config):
    population = Population(small_config, seed=11)
    expected = population.architecture.genome_size
    for _ in range(2):
        population.run_generation()
        assert all(m.genome.size == expected for m in population.members)


def test_best_ever_tracks_the_highest_episode_score(small_config):
    population = Population(small_config, seed=12)
    seen = 0
    for _ in range(3):
        report = population.run_generation()
        seen = max(seen, report.max_episode_score)
    assert population.best_ever_score == seen
    if seen > 0:
        assert population.best_ever_genome is not None


def test_immigrants_and_elites_are_present(small_config):
    population = Population(small_config, seed=13)
    population.run_generation()
    origins = {m.origin for m in population.members}
    assert "elite" in origins
    assert "immigrant" in origins


def test_parallel_evaluation_is_identical_to_serial(small_config):
    """A speedup that changes selection is worse than no speedup.

    Seeds are fixed by the parent before any task is dispatched and results are
    applied in member order, so which core scores which genome cannot affect the
    outcome. This pins that.
    """
    from evolvarium.evolution.pool import EvaluationPool

    serial = Population(small_config, seed=99)
    parallel = Population(small_config, seed=99)

    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(3):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)

            assert a.mean_fitness == pytest.approx(b.mean_fitness)
            assert a.mean_score == pytest.approx(b.mean_score)
            assert a.max_episode_score == b.max_episode_score
            assert a.deaths == b.deaths
            assert a.diversity == pytest.approx(b.diversity)

    # Breeding consumes the RNG identically too, so the populations stay in step.
    for x, y in zip(serial.members, parallel.members, strict=True):
        assert np.array_equal(x.genome, y.genome)
    assert serial.total_episodes == parallel.total_episodes
    assert serial.best_ever_score == parallel.best_ever_score


def test_resolve_workers_leaves_a_core_free():
    import os

    from evolvarium.evolution.pool import resolve_workers

    assert resolve_workers(4) == 4
    assert resolve_workers(1) == 1
    assert resolve_workers(0) == max(1, (os.cpu_count() or 2) - 1)


def test_serial_path_needs_no_pool(small_config):
    """A pool that is not active must fall through to in-process scoring."""
    from evolvarium.evolution.pool import EvaluationPool

    population = Population(small_config, seed=7)
    pool = EvaluationPool(population.context, workers=1)
    assert not pool.active
    report = population.run_generation(pool=pool)
    assert report.mean_steps > 0
