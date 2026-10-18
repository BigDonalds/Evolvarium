"""The dashboard server, asked what a page asks it."""

from __future__ import annotations

import dataclasses
import gzip
import http.client
import itertools
import json
import os
import time
import types

import numpy as np
import pytest

from evolvarium import held_out
from evolvarium.config import CONFIG
from evolvarium.games.snake.game import SnakeGame
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.persistence import export_model, model_file
from evolvarium.replays import unpack
from evolvarium.trainer import Trainer
from evolvarium.web import runs
from evolvarium.web.runs import LAB_SECTIONS, Lab, SavedRun
from evolvarium.web.server import DashboardServer, accepts_gzip

SETTINGS = SnakeSettings(
    board=BoardConfig(width=8, height=8),
    episode=EpisodeConfig(starvation_limit=40, max_steps=400),
)


@pytest.fixture()
def config(tmp_path):
    return CONFIG.with_overrides(
        settings=SETTINGS,
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=1, elite_count=2,
            immigrant_count=1, workers=1,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def train(config, run_id, seed, generations=2):
    trainer = Trainer(config, run_id=run_id, seed=seed, max_generations=generations)
    trainer.start()
    trainer.join()
    trainer.stop()
    return trainer


@pytest.fixture()
def served(config, monkeypatch):
    """A dashboard that starts runs, over a runs directory holding one finished run."""
    monkeypatch.setattr(export_model, "preload", lambda: None)
    train(config, "served", 4)
    lab = Lab(config, can_start=True)
    server = DashboardServer(lab, "127.0.0.1", 0)
    assert server.start()
    try:
        yield types.SimpleNamespace(
            port=server._httpd.server_address[1], lab=lab, server=server, config=config
        )
    finally:
        lab.close()
        server.stop()


def get(port, path, **headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


def fetch(port, path):
    response, body = get(port, path)
    assert response.status == 200, body
    return json.loads(body)


def post(port, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload or {}).encode("utf-8")
        connection.request("POST", path, body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return json.loads(response.read())
    finally:
        connection.close()


def wait_for(check, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("timed out")


def job(port, job_id):
    return fetch(port, f"/api/export/job?id={job_id}")["job"]


def finished(port, job_id):
    return wait_for(lambda: (found := job(port, job_id))["state"] != "running" and found)


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
    path = "/api/generation?run=served&n=1"
    response, body = get(served.port, path, **{"Accept-Encoding": "gzip"})
    assert response.status == 200
    assert response.getheader("Content-Encoding") == "gzip"
    data = json.loads(gzip.decompress(body))

    scene = SnakeGame(SETTINGS).scene()
    assert data["run"] == "served"
    assert data["scene"] == json.loads(json.dumps(scene.describe()))
    assert data["agents"]
    for agent in data["agents"]:
        frames = unpack(scene, agent["replay"])
        assert len(frames) == agent["replay"]["frames"] >= 1
        assert {"food", "body", "head", "score"} == set(frames[0])

    plain, raw = get(served.port, path)
    assert plain.getheader("Content-Encoding") is None
    assert json.loads(raw) == data


def test_the_status_describes_the_scene(served):
    status = fetch(served.port, "/api/status?run=served")
    assert [layer["name"] for layer in status["scene"]["layers"]] == ["food", "body", "head"]
    assert status["scene"]["width"] == 8
    assert status["direction"] == "higher"


def test_without_a_run_named_the_newest_is_shown(served):
    train(served.config, "later", 5, generations=1)
    assert fetch(served.port, "/api/status")["run_id"] == "later"
    assert fetch(served.port, "/api/generations")["current"] == 1


@pytest.mark.parametrize("name", ["missing", "..", "..%2Fruns", "served%2F..%2Fserved"])
def test_a_run_that_is_not_there_is_not_found(served, name):
    for path in ("status", "curve", "generations", "generation?n=0", "export/preview"):
        separator = "&" if "?" in path else "?"
        response, _ = get(served.port, f"/api/{path}{separator}run={name}")
        assert response.status == 404


def test_a_saved_run_reads_back_what_it_recorded(served):
    status = fetch(served.port, "/api/status?run=served")
    assert status["live"] is False and status["running"] is False
    assert status["generation"] == 2
    assert status["seed"] == 4
    assert status["population_size"] == 10
    assert status["label"] != "unknown settings"
    assert status["total_episodes"] == 20

    generations = fetch(served.port, "/api/generations?run=served")
    assert generations["available"] == [0, 1]
    curve = fetch(served.port, "/api/curve?run=served")
    assert curve["generation"] == [0, 1]


def test_the_curve_carries_only_what_any_game_reports(served):
    curve = fetch(served.port, "/api/curve?run=served")
    for key in ("best_score", "mean_score", "max_episode_score", "mutation_rate",
                "diversity", "mean_steps", "stagnation", "record"):
        assert len(curve[key]) == 2, key
    assert curve["record"][0] is True


def test_the_runs_are_listed_newest_first(served):
    train(served.config, "later", 5, generations=1)
    listing = fetch(served.port, "/api/runs")
    assert [row["run_id"] for row in listing["runs"]] == ["later", "served"]
    assert listing["live"] is None and listing["can_start"] is True
    row = listing["runs"][1]
    assert row["game"] == "snake" and row["generations"] == 2 and row["seed"] == 4
    assert row["problem"] is None and row["updated_seconds_ago"] >= 0


def test_the_form_gets_every_setting_but_the_dashboards_own(served):
    rows = fetch(served.port, "/api/settings?game=snake")["settings"]
    names = {row["name"]: row for row in rows}
    assert not [name for name in names if name.partition(".")[0] in LAB_SECTIONS]
    assert names["evolution.population_size"]["value"] == "10"
    assert names["evolution.population_size"]["input"] == "whole"
    assert names["board.width"]["group"] == "game"
    assert {"evolution", "network", "game"} <= {row["group"] for row in rows}

    assert "snake" in fetch(served.port, "/api/games")["games"]
    assert "error" in fetch(served.port, "/api/settings?game=no-such-game")


def test_every_problem_with_a_new_run_is_reported_at_once(served):
    reply = post(served.port, "/api/runs/start", {
        "game": "snake",
        "settings": {"evolution.populaton_size": "12", "paths.runs_dir": "elsewhere",
                     "evolution.elite_count": "many"},
        "generations": 0,
    })
    problems = " | ".join(reply["problems"])
    assert "did you mean evolution.population_size?" in problems
    assert "paths.runs_dir is the dashboard's to set" in problems
    assert "generations takes a whole number of 1 or more" in problems
    assert served.lab.live is None
    assert fetch(served.port, "/api/runs")["live"] is None

    unknown = post(served.port, "/api/runs/start", {"game": "no-such-game"})
    assert unknown["problems"]


def test_the_dashboard_of_one_run_starts_no_others(config):
    lab = Lab(config, can_start=False)
    reply = lab.start("snake", {})
    assert "serve command" in reply["problems"][0]
    assert lab.live is None


def test_a_run_started_on_the_page_trains_and_is_then_read_from_disk(served):
    reply = post(served.port, "/api/runs/start", {
        "game": "snake", "settings": {"evolution.population_size": "8"}, "generations": 1,
    })
    assert reply["ok"], reply
    run_id = reply["run"]
    wait_for(lambda: served.lab.live is None)

    status = fetch(served.port, f"/api/status?run={run_id}")
    assert status["live"] is False and status["generation"] == 1
    assert status["seed"] == reply["seed"]
    assert status["population_size"] == 8
    assert "evolution.population_size=8" in status["label"]

    # It ended here, so the page need not wait on another process writing to it,
    # until something does.
    rows = {row["run_id"]: row for row in fetch(served.port, "/api/runs")["runs"]}
    assert status["ended_here"] is True and rows[run_id]["ended_here"] is True
    assert rows["served"]["ended_here"] is False
    history = served.config.paths.runs_dir / run_id / "history.jsonl"
    later = time.time() + 5
    os.utime(history, (later, later))
    assert fetch(served.port, f"/api/status?run={run_id}")["ended_here"] is False

    # Every event the run published reached the page marked with its run.
    _, messages, _ = served.server.events.read(0, timeout=0)
    marked = [json.loads(m.split("data: ", 1)[1]) for m in messages]
    assert {event["kind"] for event in marked} >= {"generation", "run_finished"}
    assert all(event["run"] == run_id for event in marked)


def test_only_the_live_run_is_held_and_stopped(served):
    assert post(served.port, "/api/heartbeat")["gate"] is None
    assert "error" in post(served.port, "/api/runs/stop", {})

    run_id = post(served.port, "/api/runs/start", {"game": "snake"})["run"]
    assert post(served.port, "/api/runs/start", {"game": "snake"})["problems"]
    assert fetch(served.port, "/api/runs")["live"] == run_id
    wait_for(lambda: fetch(served.port, "/api/status")["generation"] >= 1)
    assert post(served.port, "/api/heartbeat")["gate"] is not None
    assert "error" in post(served.port, "/api/control", {"run": "served", "action": "hold"})

    assert post(served.port, "/api/runs/stop", {"run": run_id})["ok"]
    wait_for(lambda: served.lab.live is None)
    status = fetch(served.port, f"/api/status?run={run_id}")
    assert status["running"] is False and status["generation"] >= 1


def test_runs_are_set_side_by_side(served):
    train(served.config, "other", 9, generations=3)
    data = fetch(served.port, "/api/compare?runs=served,other,missing")
    assert data["missing"] == ["missing"]
    first, second = data["runs"]
    assert (first["run_id"], second["run_id"]) == ("served", "other")
    assert first["curve"]["generation"] == [0, 1]
    assert len(second["curve"]["best_score"]) == 3
    assert second["seed"] == 9 and second["episodes"] == 30


def test_the_page_exports_by_average_as_the_command_line_does(served):
    started = post(served.port, "/api/export/choose", {"run": "served"})
    assert started["ok"], started
    job_id = started["job"]["id"]
    done = finished(served.port, job_id)
    assert done["state"] == "done", done
    assert done["played"] == done["games"] > 0 and done["stage"] == "final"
    assert done["report"]["pick"] == "average"

    latest = served.config.paths.runs_dir / "served" / "checkpoints" / "latest.npz"
    expected = held_out.choose_by_average(
        held_out.candidates_from(latest), held_out.run_context(latest), workers=1
    )
    assert done["report"]["selection"] == json.loads(json.dumps(expected.to_dict()))

    written = post(served.port, "/api/export", {"run": "served", "job": job_id})
    assert written["written"], written
    path = served.config.paths.exports_dir / "served.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["source"]["selection"]["rule"] == "best average"
    assert np.array_equal(model_file.to_genome(document)[0], expected.genome)


def test_a_champion_is_exported_as_of_a_generation(served):
    preview = fetch(served.port, "/api/export/preview?run=served&n=1")
    assert preview["pick"] == "champion" and preview["resolved_generation"] == 1
    assert preview["filename"] == "served.json"
    written = post(served.port, "/api/export", {"run": "served", "generation": 1})
    assert written["written"] and written["generation"] == 1
    assert "error" in post(served.port, "/api/export", {"run": "served", "job": "41"})


def test_a_choice_can_be_cancelled_and_only_one_runs_at_once(served, monkeypatch):
    def endless(candidates, context, workers, *, progress=None, **_):
        for played in itertools.count():
            progress("screen", played, 10**6)
            time.sleep(0.01)

    monkeypatch.setattr(runs, "choose_by_average", endless)
    job_id = post(served.port, "/api/export/choose", {"run": "served"})["job"]["id"]
    assert "already choosing" in post(served.port, "/api/export/choose", {"run": "served"})["error"]
    wait_for(lambda: job(served.port, job_id)["played"] > 0)

    assert post(served.port, "/api/export/cancel", {"job": job_id})["ok"]
    assert finished(served.port, job_id)["state"] == "cancelled"
    assert "error" in post(served.port, "/api/export", {"run": "served", "job": job_id})


def test_a_saved_run_answers_with_the_fields_of_a_live_one(served):
    trainer = Trainer(served.config, run_id="fresh", seed=1, max_generations=1)
    live = served.lab.adopt(trainer)
    try:
        saved = SavedRun(served.config.paths.runs_dir / "served", served.config.history)
        assert set(live.status()) == set(saved.status())
        assert set(live.entry()) == set(saved.entry())
        assert set(live.comparison()) == set(saved.comparison())
    finally:
        served.lab.retire(live)


def test_a_run_whose_game_no_longer_loads_is_still_listed(served):
    broken = served.config.paths.runs_dir / "broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text(
        json.dumps({"run_id": "broken", "seed": 3, "config": {"game": "nowhere:Missing"}}),
        encoding="utf-8",
    )
    rows = {row["run_id"]: row for row in fetch(served.port, "/api/runs")["runs"]}
    assert rows["broken"]["problem"] and rows["served"]["problem"] is None
    status = fetch(served.port, "/api/status?run=broken")
    assert status["problem"] and status["scene"] is None
    assert "error" in fetch(served.port, "/api/export/preview?run=broken")
    assert "error" in post(served.port, "/api/export/choose", {"run": "broken"})
