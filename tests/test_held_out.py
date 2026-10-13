"""Held-out scoring of finished champions."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evolvarium import held_out
from evolvarium.cli import main
from evolvarium.config import CONFIG
from evolvarium.evolution.evaluation import make_runner
from evolvarium.trainer import Trainer


@pytest.fixture()
def tiny_config(tmp_path):
    return CONFIG.with_overrides(
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=6, evaluation_episodes=1, elite_count=1,
            immigrant_count=1,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def make_run(config, seed, run_id, generations=2):
    """A real run directory: manifest, history and a checkpoint holding a champion."""
    trainer = Trainer(config, seed=seed, run_id=run_id)
    population = trainer.population
    for _ in range(generations):
        population.run_generation()
    if population.best_ever_genome is None:
        # A tiny population can go two generations without eating. These tests
        # need a genome to score, not a good one.
        population.best_ever_genome = population.members[0].genome.copy()
    trainer.save_latest()
    return trainer.run_dir


def latest(run_dir):
    return run_dir / "checkpoints" / "latest.npz"


def scored(mean):
    """A champion whose held-out episodes all scored `mean`."""
    return held_out.Champion(
        run_id="run", path=Path("run"), seed=None, generation=60, train_best=0,
        genome=np.zeros(1, dtype=np.float32), context=None, settings={},
        scores=np.full(10, float(mean)),
    )


def test_targets_resolve_from_directories_ids_files_and_patterns(tiny_config):
    runs = tiny_config.paths.runs_dir
    first = make_run(tiny_config, 1, "run-a")
    second = make_run(tiny_config, 2, "run-b")

    assert held_out.resolve_targets([str(first)], runs) == [latest(first).resolve()]
    assert held_out.resolve_targets(["run-b"], runs) == [latest(second).resolve()]
    assert held_out.resolve_targets([str(latest(first))], runs) == [latest(first).resolve()]
    # PowerShell passes patterns through unexpanded, so they are expanded here.
    assert len(held_out.resolve_targets(["run-*"], runs)) == 2
    # Naming a run more than once scores it once.
    assert len(held_out.resolve_targets([str(first), "run-a", "run-*"], runs)) == 2


def test_a_target_that_names_nothing_is_an_error(tiny_config):
    runs = tiny_config.paths.runs_dir
    with pytest.raises(held_out.TargetError, match="not a checkpoint"):
        held_out.resolve_targets(["run-missing"], runs)
    with pytest.raises(held_out.TargetError, match="matches no checkpoint"):
        held_out.resolve_targets(["nothing-*"], runs)


def test_each_episode_is_one_seed_played_once_without_exploration(tiny_config):
    champion = held_out.load_champion(latest(make_run(tiny_config, 3, "run-c")))
    seeds = [900000, 900001, 900002]
    held_out.score([champion], seeds, workers=1)

    runner = make_runner(champion.context)
    expected = [runner.score(champion.genome, seed, 1, False).score for seed in seeds]
    assert champion.scores.tolist() == expected
    assert champion.context.exploration_rate == 0.0
    assert sum(champion.deaths.values()) == len(seeds)


def test_parallel_scoring_matches_serial(tiny_config):
    path = latest(make_run(tiny_config, 4, "run-d"))
    serial, parallel = held_out.load_champion(path), held_out.load_champion(path)
    seeds = list(range(900000, 900006))

    held_out.score([serial], seeds, workers=1)
    held_out.score([parallel], seeds, workers=2)

    assert serial.scores.tolist() == parallel.scores.tolist()
    assert serial.deaths == parallel.deaths


def test_runs_differing_only_by_seed_or_workers_share_a_configuration(tiny_config):
    faster = tiny_config.with_overrides(
        evolution=dataclasses.replace(tiny_config.evolution, workers=3)
    )
    other_encoding = tiny_config.with_overrides(
        features=dataclasses.replace(tiny_config.features, spec="egocentric_v2")
    )
    runs = [
        make_run(tiny_config, 5, "run-e"),
        make_run(faster, 6, "run-f"),
        make_run(other_encoding, 7, "run-g"),
    ]
    champions = [held_out.load_champion(latest(r)) for r in runs]

    groups = held_out.group(champions)

    assert [len(g.champions) for g in groups] == [2, 1]
    assert "workers" not in groups[0].label
    assert "features.spec=egocentric_v2" in groups[1].label
    assert "features.spec" not in groups[0].label


def test_unrecorded_runs_stay_apart_unless_declared_the_same(tiny_config):
    runs = [make_run(tiny_config, 8, "run-h"), make_run(tiny_config, 9, "run-i")]
    (runs[1] / "run.json").unlink()
    champions = [held_out.load_champion(latest(r)) for r in runs]

    apart = held_out.group(champions)
    assert len(apart) == 2
    assert "not recorded" in apart[1].label
    assert apart[1].champions[0].seed is None

    together = held_out.group(champions, same_config=True)
    assert len(together) == 1
    assert together[0].label == apart[0].label


def test_comparison_uses_spread_between_runs_not_between_episodes():
    # Every episode of a champion scores the same, so any interval must come
    # from the differences between champions.
    reference = held_out.Group("defaults", 60, [scored(m) for m in (40, 50, 60)])
    other = held_out.Group("changed", 60, [scored(m) for m in (55, 60, 65)])

    result = held_out.compare(reference, other)

    assert result.difference == pytest.approx(10.0)
    # Welch degrees of freedom come to 2.94 here, which rounds down to 2.
    assert result.half_width == pytest.approx(4.303 * math.sqrt(100 / 3 + 25 / 3))
    assert result.verdict == "within seed noise"


def test_a_single_run_gives_no_interval():
    reference = held_out.Group("defaults", 60, [scored(40), scored(50)])
    single = held_out.Group("changed", 60, [scored(90)])

    result = held_out.compare(reference, single)

    assert single.interval() is None
    assert result.half_width is None
    assert "seed luck" in result.verdict


def test_t_quantiles_round_toward_caution():
    assert held_out.t_critical(4.9) == pytest.approx(2.776)
    assert held_out.t_critical(0.5) == pytest.approx(12.706)
    assert held_out.t_critical(10_000) == pytest.approx(1.980)


def test_evaluate_command_reports_and_writes_every_episode(tiny_config, tmp_path, capsys):
    first = make_run(tiny_config, 10, "run-j")
    second = make_run(tiny_config, 11, "run-k")
    out = tmp_path / "result.json"

    code = main([
        "evaluate", str(first), str(second),
        "--episodes", "3", "--workers", "1", "--json", str(out),
    ])

    assert code == 0
    printed = capsys.readouterr().out
    assert "(2 runs)" in printed
    assert "not comparable with HISTORY.md" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["held_out"]["episodes"] == 3
    [group] = report["groups"]
    assert [len(c["scores"]) for c in group["champions"]] == [3, 3]
    assert {c["seed"] for c in group["champions"]} == {10, 11}


# ------------------------------------------------------------------ choosing

def _tagged(value):
    return np.full(3, value, dtype=np.float32)


def _scripted(table):
    """Scores from a table keyed by a genome's first weight: (screen game, final game)."""
    screen_end = held_out.SELECTION_FIRST_SEED + held_out.SCREEN_GAMES

    def score(genomes, seeds):
        return np.array([[table[float(g[0])][0 if s < screen_end else 1] for s in seeds]
                         for g in genomes])
    return score


def test_the_pick_goes_to_the_better_average_not_the_luckier_screen():
    candidates = [
        ("stored champion", _tagged(0.0)),       # leads the screen, poor afterwards
        ("population member 0", _tagged(1.0)),   # steady
        ("population member 1", _tagged(2.0)),
    ]
    score = _scripted({0.0: (60.0, 10.0), 1.0: (40.0, 40.0), 2.0: (20.0, 20.0)})

    choice = held_out.choose_by_average(candidates, context=None, workers=1, score=score)

    assert choice.source == "population member 0"
    assert choice.mean == pytest.approx(40.0)
    assert choice.champion_mean == pytest.approx((8 * 60 + 40 * 10) / 48)


def test_the_stored_champion_is_always_measured_in_the_final():
    candidates = [
        ("stored champion", _tagged(2.0)),
        ("population member 0", _tagged(1.0)),
        ("population member 1", _tagged(0.0)),
    ]
    score = _scripted({0.0: (60.0, 10.0), 1.0: (40.0, 40.0), 2.0: (20.0, 20.0)})

    # One finalist from the screen, and it is the fluke; the champion joins it.
    choice = held_out.choose_by_average(candidates, context=None, workers=1, score=score,
                                        finalists=1)

    assert choice.champion_mean == pytest.approx(20.0)
    assert choice.source == "stored champion"


def test_selection_games_stay_clear_of_the_held_out_block():
    first = held_out.SELECTION_FIRST_SEED
    last = first + held_out.SCREEN_GAMES + held_out.FINAL_GAMES
    assert not set(range(first, last)) & set(CONFIG.held_out.seeds)


def test_candidates_cover_the_champion_the_population_and_past_champions(tiny_config):
    run = make_run(tiny_config, 12, "run-l", generations=3)
    candidates = held_out.candidates_from(latest(run))

    sources = [source for source, _ in candidates]
    assert sources[0] == "stored champion"
    assert sum(source.startswith("population member") for source in sources) >= 1
    keys = [genome.tobytes() for _, genome in candidates]
    assert len(keys) == len(set(keys)), "the same genome must not be scored twice"


def test_export_can_pick_by_average(tiny_config, monkeypatch):
    pytest.importorskip("torch")
    import evolvarium.cli as cli

    run = make_run(tiny_config, 13, "run-m")
    monkeypatch.setattr(cli, "CONFIG", tiny_config)

    code = main(["export", "--checkpoint", str(latest(run)), "--pick", "best-average",
                 "--name", "picked", "--workers", "1"])

    assert code == 0
    exports = tiny_config.paths.exports_dir
    sidecar = json.loads((exports / "picked.json").read_text(encoding="utf-8"))
    assert sidecar["selection"]["rule"] == "best average"
    assert sidecar["selection"]["candidates"] >= 1
    archived = list((exports / "archive").glob("*.pt"))
    assert archived and all(p.name.startswith("picked_gen") for p in archived)
