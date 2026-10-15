"""The profiler: its accounting adds up and its batched network plays the same moves."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from test_golden import GOLDEN, golden_settings

from evolvarium import profiling
from evolvarium.cli import main
from evolvarium.config import CONFIG
from evolvarium.evolution.evaluation import EvalContext
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.nn.activations import resolve
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.trainer import Trainer

GENOME = np.array(GOLDEN["genome"], dtype=np.float32)
ARCHITECTURE = Architecture.from_dict(GOLDEN["architecture"])

# The two shortest held-out games, 433 and 1,318 moves.
SEEDS = [900029, 900041]


def golden_context(**changes) -> EvalContext:
    ctx = EvalContext(
        game="snake",
        settings=golden_settings(),
        architecture=GOLDEN["architecture"],
        exploration_rate=0.0,
        replay_steps=0,
    )
    return dataclasses.replace(ctx, **changes)


def random_genomes(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, (count, len(GENOME))).astype(np.float32)


def test_every_move_is_counted_once_and_the_games_are_unchanged():
    # Exploration on in the context: the profile has to turn it off, or the
    # games would not be the recorded ones.
    moves, states = profiling.profile_moves(golden_context(exploration_rate=0.5), GENOME, SEEDS)

    # Both games end in a collision. The move that collides is decided like any
    # other but not counted in the game's steps.
    recorded = {game["seed"]: game["steps"] + 1 for game in GOLDEN["games"]}
    assert moves.moves == sum(recorded[seed] for seed in SEEDS)
    assert sum(band.moves for band in moves.bands) == moves.moves
    assert [band.low for band in moves.bands] == [
        i * profiling.BAND_WIDTH for i in range(len(moves.bands))
    ]
    assert moves.workload == "snake length"
    assert "snake length" in "\n".join(moves.lines())
    assert moves.inputs > 0 and moves.network > 0 and moves.game > 0
    assert moves.inputs + moves.network + moves.game <= moves.timed
    # The first move of the run and every 64th after it.
    assert states.shape == (-(-moves.moves // profiling.SAMPLE_EVERY), ARCHITECTURE.input_size)


def test_input_shares_cover_the_whole_encoder():
    parts = profiling.split_inputs(golden_context(), GENOME, SEEDS[:1])

    assert sum(part.share for part in parts) == pytest.approx(1.0, abs=1e-3)
    assert all(part.share >= 0 for part in parts)
    assert [part.share for part in parts] == sorted((part.share for part in parts), reverse=True)
    assert any(part.name.endswith(" itself") for part in parts)


def test_batched_forward_matches_each_genome_alone():
    genomes = random_genomes(7)
    x = np.random.default_rng(1).normal(0.0, 1.0, (7, ARCHITECTURE.input_size))
    x = x.astype(np.float32)
    activations = [resolve(name) for name in ARCHITECTURE.hidden_activations]

    batched = profiling.forward_each(profiling.stacked_layers(ARCHITECTURE, genomes),
                                     activations, x)

    network = PolicyNetwork(ARCHITECTURE)
    alone = np.stack([network.load(g).forward(row) for g, row in zip(genomes, x, strict=True)])
    assert batched.shape == alone.shape
    np.testing.assert_allclose(batched, alone, rtol=1e-5, atol=1e-6)
    assert np.array_equal(batched.argmax(axis=1), alone.argmax(axis=1))


def test_network_times_land_on_the_right_batches(monkeypatch):
    # Real timings take seconds per function. Numbering the calls instead
    # shows which time ends up where.
    timed = []

    def numbered(functions, repeat):
        timed.extend(functions)
        return [(i + 1) * 1e-6 for i in range(len(functions))]

    monkeypatch.setattr(profiling, "seconds_per_call", numbered)
    states = np.random.default_rng(2).normal(0.0, 1.0, (5, ARCHITECTURE.input_size))
    result = profiling.profile_network(ARCHITECTURE, GENOME, random_genomes(3),
                                       states.astype(np.float32), batch_sizes=(1, 6))

    assert len(timed) == 5
    assert result.states == 5 and result.genomes == 3
    assert result.single == pytest.approx(1 / 5)
    assert [(b.size, b.same, b.own) for b in result.batches] == [
        (1, pytest.approx(2.0), pytest.approx(3.0)),
        (6, pytest.approx(4 / 6), pytest.approx(5 / 6)),
    ]
    assert result.disagreements == 0


def test_worker_time_adds_up_to_the_round():
    # Worker 1 ran two tasks with a gap, worker 2 one task to the end, and
    # worker 3 never got one.
    stamps = [(1, 1.0, 4.0), (2, 2.0, 10.0), (1, 5.0, 9.0)]
    parts = profiling.account(stamps, workers=3, started=0.0, finished=10.0)

    assert parts == {"starting": 1.0 + 2.0 + 10.0, "scoring": 3.0 + 8.0 + 4.0,
                     "between": 1.0, "end_wait": 1.0}
    assert sum(parts.values()) == 3 * 10.0


def test_pool_round_on_a_small_board():
    ctx = golden_context(settings=dataclasses.replace(
        golden_settings(),
        board=BoardConfig(width=8, height=8),
        episode=dataclasses.replace(EpisodeConfig(**GOLDEN["episode"]), starvation_limit=40),
    ))
    genomes = random_genomes(4, seed=3)
    pool = profiling.profile_pool(ctx, genomes, trials=1, workers=2, rounds=1)

    (played,) = pool.rounds
    assert played.tasks == 4 and played.moves > 0
    total = played.starting + played.scoring + played.between + played.end_wait
    assert total == pytest.approx(2 * played.wall)
    # The replays outside the pool raise if any result differs.
    assert pool.contention_tasks == 2 and pool.contention_moves > 0
    assert pool.before > 0 and pool.in_pool > 0 and pool.after > 0
    assert pool.lines()[-1].startswith("    the last round ran")

    with pytest.raises(ValueError):
        profiling.profile_pool(ctx, genomes, trials=1, workers=2, rounds=0)


def test_profile_command_on_a_saved_run(tmp_path, monkeypatch, capsys):
    config = CONFIG.with_overrides(
        settings=SnakeSettings(
            board=BoardConfig(width=8, height=8),
            episode=EpisodeConfig(starvation_limit=40, max_steps=400),
        ),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=6, evaluation_episodes=1, elite_count=1,
            immigrant_count=1, workers=1,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )
    trainer = Trainer(config, seed=4, run_id="run-p")
    trainer.population.run_generation()
    saved = trainer.save_latest()
    # The command's wiring is under test here, not the timings, which take
    # seconds per batch size.
    monkeypatch.setattr(profiling, "seconds_per_call",
                        lambda functions, repeat: [1e-6] * len(functions))

    destination = tmp_path / "profile.json"
    code = main(["profile", str(saved.path), "--member", "2", "--games", "2", "--rounds", "0",
                 "--json", str(destination)])

    assert code == 0
    assert "population member 2 of run-p" in capsys.readouterr().out
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["moves"]["games"] == 2
    assert data["network"]["genomes"] == 6
    assert data["pool"] is None

    assert main(["profile", str(saved.path), "--member", "6"]) == 1
