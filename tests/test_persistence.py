"""Checkpoints, history retention and export reporting."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from evolvarium.config import CONFIG
from evolvarium.evolution.population import Population
from evolvarium.persistence import checkpoint, export_torch
from evolvarium.telemetry.history import HistoryStore, lineage_summaries
from evolvarium.web.control import TrainingGate


@pytest.fixture()
def small_config(tmp_path):
    return CONFIG.with_overrides(
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=8, evaluation_episodes=1, elite_count=2,
            immigrant_count=1,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def test_checkpoint_round_trip(small_config, tmp_path):
    population = Population(small_config, seed=3)
    population.run_generation()
    path = tmp_path / "cp.npz"
    info = checkpoint.save(population, path)

    assert path.exists()
    assert info.generation == population.generation
    assert info.feature_spec == population.feature_spec.name

    restored = checkpoint.load(path, small_config, seed=3)
    assert restored.generation == population.generation
    assert len(restored.members) == len(population.members)
    assert restored.best_ever_score == population.best_ever_score
    for original, copy in zip(population.members, restored.members, strict=True):
        assert np.array_equal(original.genome, copy.genome)


def test_checkpoint_refuses_a_feature_spec_mismatch(small_config, tmp_path):
    population = Population(small_config, seed=4)
    path = tmp_path / "cp.npz"
    checkpoint.save(population, path)

    # Rewriting the recorded spec simulates loading under a different encoding.
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    import json

    meta = json.loads(bytes(arrays["meta"]).decode("utf-8"))
    meta["feature_spec"] = "something_else"
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)

    with pytest.raises(checkpoint.CheckpointMismatch, match="feature spec"):
        checkpoint.load(path, small_config)


def test_checkpoint_contains_no_torch_types(small_config, tmp_path):
    """Training-path persistence must not depend on a deep-learning framework."""
    population = Population(small_config, seed=5)
    path = tmp_path / "cp.npz"
    checkpoint.save(population, path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) <= {
            "genomes", "meta", "best_genome", "species_representatives",
        }


def test_history_keeps_summaries_but_bounds_memory(small_config, tmp_path):
    history_config = dataclasses.replace(CONFIG.history, memory_window=2, replay_retention=3)
    store = HistoryStore(tmp_path / "run", history_config)
    population = Population(small_config, seed=6)

    for _ in range(6):
        report = population.run_generation()
        store.append(report, [{"id": "a", "fitness": 1.0, "replay": None}], [0.5])

    # Every generation keeps a summary line.
    assert len(store.summaries()) == 6
    assert len(store.curve()["generation"]) == 6
    # Memory holds only the configured window.
    assert len(store._memory) <= 2
    # Any generation still on disk can be re-read.
    available = store.available()
    assert available
    record = store.get(available[-1])
    assert record is not None
    assert record.best_genome == [0.5]


def test_history_pins_record_generations(small_config, tmp_path):
    history_config = dataclasses.replace(CONFIG.history, memory_window=1, replay_retention=2)
    store = HistoryStore(tmp_path / "run", history_config)
    population = Population(small_config, seed=7)

    record_generations = []
    for _ in range(8):
        report = population.run_generation()
        if report.record:
            record_generations.append(report.generation)
        store.append(report, [], None)

    available = set(store.available())
    for generation in record_generations:
        assert generation in available, "record generations must survive eviction"


def test_lineage_follows_resumed_runs(tmp_path):
    def make_run(name, generations, resumed_from=None):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"resumed_from": resumed_from}))
        lines = [json.dumps({"generation": g}) for g in generations]
        (run_dir / "history.jsonl").write_text("\n".join(lines) + "\n")
        return run_dir

    # the first run went two generations past the checkpoint the second resumed from
    make_run("first", range(0, 12))
    make_run("second", range(10, 20), "runs\\first\\checkpoints\\latest.npz")
    third = make_run("third", range(20, 25), "runs/second/checkpoints/latest.npz")

    assert [s["generation"] for s in lineage_summaries(third)] == list(range(25))
    assert [s["generation"] for s in lineage_summaries(tmp_path / "first")] == list(range(12))


def test_export_report_writes_nothing(small_config, tmp_path):
    population = Population(small_config, seed=8)
    population.run_generation()
    destination = tmp_path / "exports" / "evolvarium.pt"

    report = export_torch.build_report(
        generation=0,
        genome=population.members[0].genome,
        summaries=[{"generation": 0, "max_episode_score": 3, "scoring_fraction": 0.6,
                    "mean_score": 0.4, "mean_steps": 120, "episodes": 8, "elapsed": 1.0,
                    "best_fitness": 500.0, "deaths": {"wall": 8}, "record": True}],
        config=small_config,
        architecture=population.architecture,
        feature_spec=population.feature_spec,
        run_id="run-test",
        destination=destination,
    )

    assert not destination.exists()
    assert report["parameter_count"] == population.architecture.genome_size
    assert report["feature_spec"]["size"] == population.feature_spec.size
    unlocked = [a for a in report["achievements"] if a["unlocked"]]
    assert any(a["key"] == "score_1" for a in unlocked)
    assert report["training"]["best_score"] == 3


def test_achievements_only_count_generations_up_to_the_target():
    summaries = [
        {"generation": 0, "max_episode_score": 1, "scoring_fraction": 0.1, "mean_score": 0.1,
         "mean_steps": 10, "episodes": 1, "elapsed": 0.1, "best_fitness": 1.0, "deaths": {}},
        {"generation": 9, "max_episode_score": 12, "scoring_fraction": 0.95, "mean_score": 1.4,
         "mean_steps": 210, "episodes": 1, "elapsed": 0.1, "best_fitness": 9.0, "deaths": {}},
    ]
    early = export_torch.compute_achievements([s for s in summaries if s["generation"] <= 0])
    late = export_torch.compute_achievements(summaries)

    assert not any(a.unlocked for a in early if a.key == "score_10")
    assert any(a.unlocked for a in late if a.key == "score_10")


def test_gate_holds_only_while_a_viewer_is_present():
    gate = TrainingGate(viewer_timeout=0.2, hold_enabled=True)

    # No heartbeat yet, so nothing should block.
    gate.wait_at_boundary(1)

    gate.heartbeat()
    assert gate.viewer_present() is True
    gate.release(1)
    gate.wait_at_boundary(2)  # consumes the credit and returns immediately

    state = gate.state()
    assert state.hold_enabled is True
    assert state.holding is False


def test_gate_resumes_when_the_viewer_lease_expires():
    import time

    gate = TrainingGate(viewer_timeout=0.15, hold_enabled=True)
    gate.heartbeat()
    started = time.time()
    gate.wait_at_boundary(3)
    # Blocks until the lease lapses rather than waiting for a browser forever.
    assert 0.1 <= time.time() - started < 3.0


def test_run_free_disables_holding():
    gate = TrainingGate(viewer_timeout=5.0, hold_enabled=True)
    gate.heartbeat()
    gate.set_hold(False)
    gate.wait_at_boundary(4)  # returns immediately despite a live viewer
    assert gate.state().hold_enabled is False


def test_replay_path_is_contiguous(small_config):
    """Every stored step must be adjacent to the last, or bodies cannot be drawn."""
    import base64

    population = Population(small_config, seed=21)
    outcome = population.evaluate_individual(population.members[0], base_seed=1, record=True)
    packed = outcome.replay

    assert packed is not None, "recording was requested, so a replay must come back"
    directions = base64.b64decode(packed["dirs"])
    assert len(directions) == max(packed["steps"] - 1, 0)

    # Walking the stored directions must trace a connected path on the board.
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    x, y = packed["start"]
    seen = [(x, y)]
    for step in directions:
        dx, dy = deltas[step & 3]
        x, y = x + dx, y + dy
        assert abs(x - seen[-1][0]) + abs(y - seen[-1][1]) == 1
        seen.append((x, y))
    assert len(seen) == packed["steps"]


def test_replay_growth_matches_score(small_config):
    population = Population(small_config, seed=22)
    for member in population.members:
        outcome = population.evaluate_individual(member, base_seed=5, record=True)
        packed = outcome.replay
        # The recorded episode is trial 0; growth points in it must match the
        # food that episode ate.
        if packed["grow"]:
            assert packed["food"], "a scoring episode must record food changes"
            assert len(packed["grow"]) <= outcome.best_episode_score
            return


def test_every_run_writes_a_manifest_with_its_seed(small_config):
    """A run must say what produced it before it produces anything."""
    import json

    from evolvarium.trainer import Trainer

    trainer = Trainer(small_config, seed=31, run_id="run-manifest-test")
    manifest = json.loads((trainer.run_dir / "run.json").read_text(encoding="utf-8"))

    assert manifest["seed"] == 31
    assert manifest["run_id"] == "run-manifest-test"
    evolution = manifest["config"]["evolution"]
    assert evolution["population_size"] == small_config.evolution.population_size
    assert manifest["config"]["features"]["spec"] == small_config.features.spec


def test_default_replays_record_the_whole_game():
    """A replay shorter than the game shows its end game as a death that never happened."""
    steps = CONFIG.history.replay_steps
    assert steps == 0 or steps >= CONFIG.episode.max_steps


def test_replay_marks_itself_truncated_only_when_cut():
    from evolvarium.game.engine import EpisodeResult, Frame
    from evolvarium.telemetry.history import pack_replay

    result = EpisodeResult()
    result.frames = [Frame(head=(i % 14, i // 14), food=(0, 0), length=3, score=0)
                     for i in range(50)]

    whole = pack_replay(result, 0)
    assert whole["steps"] == 50 and whole["truncated"] is False

    cut = pack_replay(result, 20)
    assert cut["steps"] == 20 and cut["truncated"] is True
