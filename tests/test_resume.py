"""A resumed run continues exactly as the run that never stopped.

Each test plays a short run straight through, then plays it again with a save
and a load partway, and compares the two. Genomes are compared bit for bit and
every generation's report must agree, so a single random draw taken in a
different order shows up as a failure.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from evolvarium.config import CONFIG
from evolvarium.evolution.individual import Individual, id_number
from evolvarium.evolution.population import Population
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.persistence import checkpoint
from evolvarium.trainer import Trainer

TOTAL = 6
SPLIT = 3


def make_config(tmp_path, speciation: bool = False):
    # A small board and a short starvation limit keep the games cheap. Nothing
    # in saving or loading depends on either.
    return CONFIG.with_overrides(
        settings=SnakeSettings(
            board=BoardConfig(width=8, height=8),
            episode=EpisodeConfig(starvation_limit=40, max_steps=400),
        ),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=12, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1, use_speciation=speciation, target_species=3,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def report_record(report) -> tuple:
    return (
        report.generation, report.best_fitness, report.best_score, report.mean_fitness,
        report.mean_score, report.max_episode_score, tuple(sorted(report.ends.items())),
        report.diversity, report.species_count, report.mutation_rate, report.stagnation,
        report.episodes, report.record,
        tuple(m.fitness for m in report.evaluated),
    )


def state_record(population: Population) -> dict:
    """Everything that decides what the population does next.

    Ids are left out: they come from a counter shared by the whole process, so
    two runs in one test process never hand out the same ones.
    """
    return {
        "generation": population.generation,
        "total_episodes": population.total_episodes,
        "best_ever": (
            population.best_ever_score, population.best_ever_fitness,
            population.best_ever_generation,
        ),
        "best_genome": (
            None if population.best_ever_genome is None
            else population.best_ever_genome.tobytes()
        ),
        "stagnation": population.stagnation,
        "rng": population.rng.bit_generator.state,
        "threshold": population.compatibility_threshold,
        "next_species_id": population._next_species_id,
        "species": [
            (s.id, s.best_fitness, s.stagnation, s.representative.tobytes())
            for s in population.species
        ],
        "members": [
            (m.genome.tobytes(), m.genome.dtype.str, m.origin, m.age, m.generation)
            for m in population.members
        ],
    }


@pytest.mark.parametrize("speciation", [False, True], ids=["plain", "speciation"])
def test_resumed_population_matches_an_uninterrupted_one(tmp_path, speciation):
    config = make_config(tmp_path, speciation)

    straight = Population(config, seed=11)
    straight_reports = [report_record(straight.run_generation()) for _ in range(TOTAL)]

    first = Population(config, seed=11)
    resumed_reports = [report_record(first.run_generation()) for _ in range(SPLIT)]
    path = tmp_path / "split.npz"
    checkpoint.save(first, path)

    # A different seed on purpose: the saved state has to win over it.
    second = checkpoint.load(path, config, seed=999)
    resumed_reports += [report_record(second.run_generation()) for _ in range(TOTAL - SPLIT)]

    assert resumed_reports == straight_reports
    assert state_record(second) == state_record(straight)
    if speciation:
        assert straight.species, "the speciation case never formed a species"


def test_resumed_trainer_matches_an_uninterrupted_one(tmp_path):
    """Same check through the trainer, resuming from the checkpoint the run left."""
    config = make_config(tmp_path)

    def run(trainer: Trainer) -> Trainer:
        trainer.start()
        trainer.join()
        trainer.stop()
        return trainer

    straight = run(Trainer(config, run_id="straight", seed=5, max_generations=TOTAL))

    first = run(Trainer(config, run_id="first", seed=5, max_generations=SPLIT))
    saved = first.checkpoint_dir / "latest.npz"
    resumed = run(Trainer(config, run_id="resumed", resume_from=saved,
                          max_generations=TOTAL))

    assert state_record(resumed.population) == state_record(straight.population)


def test_restored_members_keep_their_ids_and_parents(tmp_path):
    config = make_config(tmp_path)
    population = Population(config, seed=12)
    population.run_generation()
    path = tmp_path / "ids.npz"
    checkpoint.save(population, path)

    restored = checkpoint.load(path, config)
    assert [m.id for m in restored.members] == [m.id for m in population.members]
    assert [m.parents for m in restored.members] == [m.parents for m in population.members]

    # New members must not reuse an id already in the run.
    newcomer = Individual(genome=np.zeros(1, dtype=np.float32))
    assert id_number(newcomer.id) > max(id_number(m.id) for m in restored.members)


def test_checkpoint_without_random_state_still_loads(tmp_path):
    """Format 2 checkpoints load, seeded from the seed they are given."""
    config = make_config(tmp_path)
    population = Population(config, seed=13)
    population.run_generation()
    path = tmp_path / "old.npz"
    checkpoint.save(population, path)
    assert checkpoint.restores_exactly(path)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    meta = json.loads(bytes(arrays["meta"]).decode("utf-8"))
    for key in ("rng", "ids", "parents", "compatibility_threshold", "next_species_id",
                "species"):
        meta.pop(key)
    meta["format_version"] = 2
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    assert not checkpoint.restores_exactly(path)
    restored = checkpoint.load(path, config, seed=13)
    assert restored.rng.bit_generator.state == Population(config, seed=13).rng.bit_generator.state
    for original, copy in zip(population.members, restored.members, strict=True):
        assert np.array_equal(original.genome, copy.genome)
