"""Scenes, and replays recorded from what a game draws."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_contract import HERD, Herd, HerdSettings

from evolvarium.contract import TONES, Label, Layer, Scene
from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.games.snake.game import SnakeGame
from evolvarium.games.snake.settings import (
    BoardConfig,
    EpisodeConfig,
    FeatureConfig,
    SnakeSettings,
)
from evolvarium.nn.network import Architecture
from evolvarium.replays import Recorder, unpack

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "held_out_81.json").read_text(encoding="utf-8"))

BOARD = Scene(
    10, 10,
    [Layer("food", "cell", "goal"), Layer("body", "cell", "trail", fade=True),
     Layer("walls", "box", "wall")],
    [Label("score"), Label("speed", 2)],
)


def frames_back(scene, frames, limit=0):
    recorder = Recorder(scene, limit)
    for frame in frames:
        recorder.add(frame)
    packed = recorder.pack()
    return packed, unpack(scene, packed)


def same_frames(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert left.keys() == right.keys()
        for name, value in left.items():
            if isinstance(value, list):
                assert len(value) == len(right[name]), name
                for p, q in zip(value, right[name], strict=True):
                    assert p == pytest.approx(q), name
            else:
                assert value == pytest.approx(right[name]), name


# ----------------------------------------------------------------- the codec


def test_every_kind_of_change_comes_back():
    frames = [
        {"food": [(2, 3)], "body": [(5, 5), (5, 6), (5, 7)], "walls": [], "score": 0,
         "speed": 1.0},
        # Nothing moves.
        {"food": [(2, 3)], "body": [(5, 5), (5, 6), (5, 7)], "walls": [], "score": 0,
         "speed": 1.0},
        # The body moves one cell, then two at once, then grows without moving its tail.
        {"food": [(2, 3)], "body": [(5, 6), (5, 7), (5, 8)], "walls": [], "score": 0,
         "speed": 1.25},
        {"food": [(2, 3)], "body": [(5, 8), (6, 8), (7, 8)], "walls": [], "score": 0,
         "speed": 1.25},
        {"food": [(7, 9)], "body": [(5, 8), (6, 8), (7, 8), (7, 9)], "walls": [], "score": 1,
         "speed": -0.5},
        # Twelve cells at once, more than an operation byte holds.
        {"food": [(0, 0)], "body": [(i % 10, i // 10) for i in range(16)],
         "walls": [(0, 0, 10, 1)], "score": 1, "speed": -0.5},
        # Everything moves together, so every item changes in place.
        {"food": [(1, 1)], "body": [(i % 10, i // 10 + 1) for i in range(16)],
         "walls": [(0, 1, 10, 1)], "score": 1, "speed": 0.0},
        # More items than an operation byte counts, then none.
        {"food": [], "body": [(i % 10, i // 10) for i in range(80)],
         "walls": [(0, 0, 1, 1), (9, 9, 1, 1)], "score": -3, "speed": 0.0},
        {"food": [], "body": [], "walls": [], "score": -3, "speed": 0.0},
    ]
    packed, back = frames_back(BOARD, frames)
    assert packed["frames"] == len(frames) and packed["truncated"] is False
    same_frames(back, frames)


def test_random_edits_come_back():
    rng = np.random.default_rng(12)
    scene = Scene(50, 50, [Layer("things", "dot", "agent"), Layer("links", "line", "ink")],
                  ["count"], unit=0.5)
    things, frames = [], []
    for _ in range(400):
        roll = rng.random()
        if roll < 0.2:
            things = things[int(rng.integers(0, 3)):] + [
                tuple(rng.integers(-100, 100, 2) / 2) for _ in range(int(rng.integers(0, 9)))
            ]
        elif roll < 0.4:
            things = [(x + rng.integers(-3, 4) / 2, y + rng.integers(-3, 4) / 2)
                      for x, y in things]
        elif roll < 0.5:
            things = [tuple(rng.integers(-2000, 2000, 2) / 2)
                      for _ in range(int(rng.integers(0, 90)))]
        links = [(*a, *b) for a, b in zip(things, things[1:], strict=False)][:5]
        frames.append({"things": list(things), "links": links, "count": len(things)})
    _, back = frames_back(scene, frames)
    same_frames(back, frames)


def test_positions_are_kept_to_the_unit_and_labels_to_their_digits():
    scene = Scene(1, 1, [Layer("dot", "dot", "agent")], [Label("angle", 3)], unit=0.01)
    _, back = frames_back(scene, [{"dot": [(0.1234, -0.5678)], "angle": 0.12345}])
    assert back[0]["dot"][0] == pytest.approx((0.12, -0.57))
    assert back[0]["angle"] == pytest.approx(0.123)


def test_a_moving_body_costs_a_few_bytes_a_frame_whatever_its_length():
    scene = Scene(100, 100, [Layer("body", "cell", "trail")], ["score"])
    path = [(i % 100, (i // 100) * 2) for i in range(1100)]
    frames = [{"body": path[i:i + 100], "score": 5} for i in range(1000)]
    packed, back = frames_back(scene, frames)
    same_frames(back, frames)
    # One run of identical operations, and one new cell of one or two bytes a move.
    assert len(packed["ops"]) < 40
    assert len(packed["data"]) < 1000 * 2 * 4 / 3 + 500


def test_a_limited_recorder_is_full_after_its_steps():
    recorder = Recorder(BOARD, limit=3)
    frame = {"food": [], "body": [], "walls": [], "score": 0, "speed": 0}
    for _ in range(4):
        assert not recorder.full
        recorder.add(frame)
    assert recorder.full


@pytest.mark.parametrize(("frame", "message"), [
    ({"food": [], "body": []}, "gave no 'walls'"),
    ({"food": [], "body": [], "walls": [], "score": 0, "speed": 0, "lives": 3},
     "'lives', which the scene does not declare"),
    ({"food": [(1, 2, 3)], "body": [], "walls": [], "score": 0, "speed": 0},
     "'food' an item of 3 numbers; its shape takes 2"),
    ({"food": [], "body": [("a", 1)], "walls": [], "score": 0, "speed": 0},
     "'body' a value that cannot be recorded"),
    ({"food": [], "body": [], "walls": [], "score": None, "speed": 0},
     "'score' a value that cannot be recorded"),
])
def test_a_frame_that_does_not_fit_the_scene_says_why(frame, message):
    with pytest.raises(ValueError, match=message):
        Recorder(BOARD).add(frame)


@pytest.mark.parametrize(("build", "message"), [
    (lambda: Layer("x", "star"), "shape 'star'"),
    (lambda: Layer("x", tone="red"), "tone 'red'"),
    (lambda: Layer("x", size=-1), "cannot be negative"),
    (lambda: Label("x", 9), "0 to 6"),
    (lambda: Scene(0, 5, [Layer("x")]), "size above 0"),
    (lambda: Scene(5, 5, [Layer("x")], unit=0), "unit must be above 0"),
    (lambda: Scene(5, 5, []), "at least one layer"),
    (lambda: Scene(5, 5, [Layer("x")], ["x"]), "distinct names"),
])
def test_a_scene_that_cannot_be_drawn_says_why(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_a_scene_describes_itself_as_plain_data():
    described = json.loads(json.dumps(BOARD.describe()))
    assert described["width"] == 10 and described["unit"] == 1
    assert [layer["shape"] for layer in described["layers"]] == ["cell", "cell", "box"]
    assert described["labels"][1] == {"name": "speed", "digits": 2}


def test_the_dashboard_has_a_colour_for_every_tone():
    style = (ROOT / "evolvarium" / "web" / "static" / "style.css").read_text(encoding="utf-8")
    for tone in TONES:
        assert f"--tone-{tone}:" in style
        assert f".tone-{tone} {{ color: var(--tone-{tone}); }}" in style


# ---------------------------------------------------------------- the engine


def golden_runner(replay_steps=0):
    settings = SnakeSettings(
        board=BoardConfig(**GOLDEN["board"]),
        episode=EpisodeConfig(**GOLDEN["episode"]),
        features=FeatureConfig(**GOLDEN["features"]),
    )
    ctx = EvalContext(game="snake", settings=settings, architecture=GOLDEN["architecture"],
                      exploration_rate=0.0, replay_steps=replay_steps)
    return make_runner(ctx), np.array(GOLDEN["genome"], dtype=np.float32), settings


def test_recording_leaves_the_game_as_it_was():
    runner, genome, _ = golden_runner()
    for seed in (900029, 900003):
        watched = runner.play(genome, seed, True)
        unwatched = runner.play(genome, seed, False)
        assert (watched.score, watched.steps, watched.end, watched.fitness) == (
            unwatched.score, unwatched.steps, unwatched.end, unwatched.fitness)
        assert unwatched.replay is None


def test_a_snake_replay_draws_the_game_it_recorded():
    runner, genome, settings = golden_runner()
    scene = SnakeGame(settings).scene()
    outcome = runner.play(genome, 900029, True)
    frames = unpack(scene, outcome.replay)

    # One frame after the reset and one after every call to step. The move into
    # itself is not counted in steps and leaves the board as it was.
    assert outcome.end == "self"
    assert len(frames) == outcome.steps + 2 and frames[-1] == frames[-2]
    assert outcome.replay["truncated"] is False
    assert frames[-1]["score"] == outcome.score
    start = settings.board.initial_length
    previous = None
    for frame in frames:
        (head,) = frame["head"]
        cells = frame["body"] + [head]
        # Listed from the tail, every cell beside the next, as long as it has eaten.
        assert len(cells) == start + frame["score"]
        for a, b in zip(cells, cells[1:], strict=False):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        if previous is not None:
            assert abs(head[0] - previous[0]) + abs(head[1] - previous[1]) <= 1
        assert len(frame["food"]) <= 1
        previous = head


def test_a_replay_is_marked_truncated_only_when_it_was_cut():
    runner, genome, _ = golden_runner(replay_steps=20)
    outcome = runner.play(genome, 900029, True)
    assert outcome.steps > 20
    assert outcome.replay["frames"] == 21 and outcome.replay["truncated"] is True

    runner, genome, _ = golden_runner(replay_steps=outcome.steps + 1)
    whole = runner.play(genome, 900029, True)
    assert whole.replay["frames"] == outcome.steps + 2 and whole.replay["truncated"] is False


def test_the_replay_kept_is_the_best_episode():
    runner, genome, _ = golden_runner()
    # Snake scores 56, 100 and 100 on these seeds, so the best is not the first
    # and two tie for it. The earlier of the two is kept.
    result = runner.score(genome, 900003, 3, True)
    best = runner.play(genome, 900004, True)
    assert result.best_episode_score == best.score == 100
    assert result.replay == best.replay
    assert result.replay != runner.play(genome, 900005, True).replay
    # The steps shown with the replay are that episode's own, not the mean.
    assert result.best_episode_steps == best.steps == 3623
    assert result.steps != best.steps

    # Herd counts lower as better, and does best on the last of these seeds.
    game = Herd(HerdSettings())
    architecture = Architecture(game.observation.size, (6,), ("tanh",), game.actions.size)
    ctx = EvalContext(game=HERD, settings=HerdSettings(), architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    genome = np.random.default_rng(5).normal(size=architecture.genome_size).astype(np.float32)
    runner = make_runner(ctx)
    result = runner.score(genome, 10, 3, True)
    scores = [runner.play(genome, seed, False).score for seed in (10, 11, 12)]
    assert result.best_episode_score == min(scores) == scores[2]
    assert result.replay == runner.play(genome, 12, True).replay
    assert result.best_episode_steps == runner.play(genome, 12, False).steps
    assert runner.score(genome, 10, 3, False).replay is None


def test_a_game_in_continuous_space_records_its_positions():
    game = Herd(HerdSettings())
    architecture = Architecture(game.observation.size, (6,), ("tanh",), game.actions.size)
    ctx = EvalContext(game=HERD, settings=HerdSettings(), architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    genome = np.random.default_rng(5).normal(size=architecture.genome_size).astype(np.float32)
    outcome = make_runner(ctx).play(genome, 11, True)

    frames = unpack(game.scene(), outcome.replay)
    assert len(frames) == outcome.steps + 1
    assert all(len(frame["herd"]) == 2 for frame in frames)
    assert frames[-1]["spread"] == pytest.approx(outcome.score, abs=0.005)
    final = sum(abs(x) for x, _ in frames[-1]["herd"])
    assert final == pytest.approx(outcome.score, abs=0.011)


def test_a_game_without_a_scene_records_nothing():
    class Unseen(Herd):
        def scene(self):
            return None

    game = Unseen(HerdSettings())
    architecture = Architecture(game.observation.size, (6,), ("tanh",), game.actions.size)
    ctx = EvalContext(game=HERD, settings=HerdSettings(), architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    runner = make_runner(ctx)
    runner.game, runner.scene = game, game.scene()
    genome = np.zeros(architecture.genome_size, dtype=np.float32)
    assert runner.play(genome, 3, True).replay is None
