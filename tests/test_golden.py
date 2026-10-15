"""The shipped 81.5 model, replayed on the held-out block.

tests/data/held_out_81.json holds the model's genome, the rules it was trained
under and every held-out game it played at the baseline: score, length in moves
and how it ended. A change anywhere in the game, the encoding, the network or
the food sequence that alters even one move sends the game down a different
path, and over thousands of moves that path ends somewhere else.

Replaying all 60 games takes under a minute, so the whole block is marked slow
and runs with `pytest -m slow`. Six of the games, between them covering every
way this model dies, run with the rest of the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.games.snake.settings import (
    BoardConfig,
    EpisodeConfig,
    FeatureConfig,
    SnakeSettings,
)
from evolvarium.persistence import model_file

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "held_out_81.json").read_text(encoding="utf-8"))

# The four shortest games, plus the only starvation and the shorter of the two wall deaths.
QUICK_SEEDS = (900029, 900041, 900010, 900003, 900022, 900038)


def golden_settings() -> SnakeSettings:
    """The rules the model was trained under. Playing never reads fitness."""
    return SnakeSettings(
        board=BoardConfig(**GOLDEN["board"]),
        episode=EpisodeConfig(**GOLDEN["episode"]),
        features=FeatureConfig(**GOLDEN["features"]),
    )


def replay(seeds) -> list[dict]:
    """Play the stored genome once per seed under the stored rules, exploration off."""
    ctx = EvalContext(
        game="snake",
        settings=golden_settings(),
        architecture=GOLDEN["architecture"],
        exploration_rate=0.0,
        replay_steps=0,
    )
    runner = make_runner(ctx)
    genome = np.array(GOLDEN["genome"], dtype=np.float32)
    games = []
    for seed in seeds:
        outcome = runner.play(genome, seed, False)
        games.append(
            {"seed": seed, "score": outcome.score, "steps": outcome.steps,
             "death": outcome.end}
        )
    return games


def recorded(seeds) -> list[dict]:
    by_seed = {game["seed"]: game for game in GOLDEN["games"]}
    return [by_seed[seed] for seed in seeds]


def test_quick_games_replay_exactly():
    assert replay(QUICK_SEEDS) == recorded(QUICK_SEEDS)


@pytest.mark.slow
def test_whole_held_out_block_replays_exactly():
    seeds = [game["seed"] for game in GOLDEN["games"]]
    games = replay(seeds)

    changed = [(old, new) for old, new in zip(recorded(seeds), games, strict=True) if old != new]
    assert not changed, f"{len(changed)} of {len(seeds)} games changed, first {changed[0]}"
    assert np.mean([game["score"] for game in games]) == pytest.approx(81.45)


def test_stored_genome_is_the_shipped_model():
    torch = pytest.importorskip("torch")
    model = torch.load(ROOT / GOLDEN["model"], weights_only=False)

    shipped = np.asarray(model["genome"], dtype=np.float32)
    assert np.array_equal(shipped, np.array(GOLDEN["genome"], dtype=np.float32))


def test_shipped_model_file_holds_the_genome_and_its_rules():
    document = model_file.read((ROOT / GOLDEN["model"]).with_suffix(".json"))
    genome, architecture = model_file.to_genome(document)

    assert np.array_equal(genome, np.array(GOLDEN["genome"], dtype=np.float32))
    assert architecture.to_dict() == GOLDEN["architecture"]
    assert document["inputs"]["encoding"] == GOLDEN["features"]["spec"]
    settings = {**document["inputs"]["settings"], "spec": document["inputs"]["encoding"]}
    assert settings == GOLDEN["features"]
    assert document["game"]["board"] == GOLDEN["board"]
    assert document["game"]["episode"] == GOLDEN["episode"]
