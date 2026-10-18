"""The dashboard server, asked what a page asks it."""

from __future__ import annotations

import dataclasses
import gzip
import http.client
import json

import pytest

from evolvarium.config import CONFIG
from evolvarium.games.snake.game import SnakeGame
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.persistence import export_model
from evolvarium.replays import unpack
from evolvarium.trainer import Trainer
from evolvarium.web.server import DashboardServer, accepts_gzip

SETTINGS = SnakeSettings(
    board=BoardConfig(width=8, height=8),
    episode=EpisodeConfig(starvation_limit=40, max_steps=400),
)


@pytest.fixture()
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(export_model, "preload", lambda: None)
    config = CONFIG.with_overrides(
        settings=SETTINGS,
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=1, elite_count=2,
            immigrant_count=1, workers=1,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )
    trainer = Trainer(config, run_id="served", seed=4, max_generations=2)
    trainer.start()
    trainer.join()
    server = DashboardServer(trainer, "127.0.0.1", 0)
    assert server.start()
    try:
        yield server._httpd.server_address[1]
    finally:
        server.stop()
        trainer.stop()


def get(port, path, **headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize(("header", "expected"), [
    ("gzip, deflate, br", True),
    ("deflate;q=1.0, GZIP;q=0.5", True),
    ("*", True),
    ("gzip;q=0", False),
    ("identity", False),
    ("", False),
    (None, False),
])
def test_gzip_is_used_only_where_the_browser_allows_it(header, expected):
    assert accepts_gzip(header) is expected


def test_a_generation_comes_compressed_with_a_scene_its_replays_fit(served):
    response, body = get(served, "/api/generation?n=1", **{"Accept-Encoding": "gzip"})
    assert response.status == 200
    assert response.getheader("Content-Encoding") == "gzip"
    data = json.loads(gzip.decompress(body))

    scene = SnakeGame(SETTINGS).scene()
    assert data["scene"] == json.loads(json.dumps(scene.describe()))
    assert data["agents"]
    for agent in data["agents"]:
        frames = unpack(scene, agent["replay"])
        assert len(frames) == agent["replay"]["frames"] >= 1
        assert {"food", "body", "head", "score"} == set(frames[0])

    plain, raw = get(served, "/api/generation?n=1")
    assert plain.getheader("Content-Encoding") is None
    assert json.loads(raw) == data


def test_the_status_describes_the_scene(served):
    _, body = get(served, "/api/status")
    status = json.loads(body)
    assert [layer["name"] for layer in status["scene"]["layers"]] == ["food", "body", "head"]
    assert status["scene"]["width"] == 8
