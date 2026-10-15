"""Training events: the bus, the trainer's announcements and the dashboard stream."""

from __future__ import annotations

import dataclasses
import http.client
import signal
import threading
import types

import pytest

from evolvarium.config import CONFIG
from evolvarium.events import (
    CheckpointSaved,
    Event,
    EventBus,
    GenerationEvaluated,
    GenerationStarted,
    Note,
    RecordSet,
    RunFinished,
    RunStarted,
)
from evolvarium.evolution.pool import EvaluationPool
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.persistence import checkpoint, export_model
from evolvarium.trainer import Trainer
from evolvarium.web.server import DashboardServer
from evolvarium.web.stream import EventLog


def make_config(tmp_path, workers: int = 1):
    return CONFIG.with_overrides(
        settings=SnakeSettings(
            board=BoardConfig(width=8, height=8),
            episode=EpisodeConfig(starvation_limit=40, max_steps=400),
        ),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=1, elite_count=2,
            immigrant_count=1, workers=workers,
        ),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def listen(events: EventBus, *kinds: type[Event]) -> list[Event]:
    heard: list[Event] = []
    events.subscribe(heard.append, *kinds)
    return heard


# ---------------------------------------------------------------------- bus


def test_record_keepers_hear_an_event_before_views():
    events = EventBus()
    order = []
    events.subscribe(lambda e: order.append("view"))
    events.subscribe(lambda e: order.append("keeper"), keeps_record=True)
    events.publish(Note("x"))
    assert order == ["keeper", "view"]


def test_subscriptions_filter_by_kind_and_can_end():
    events = EventBus()
    notes = listen(events, Note)
    everything = []
    unsubscribe = events.subscribe(everything.append)

    events.publish(Note("a"))
    events.publish(GenerationStarted(4))
    unsubscribe()
    events.publish(Note("b"))

    assert [e.message for e in notes] == ["a", "b"]
    assert everything == [Note("a"), GenerationStarted(4)]


def test_a_failing_view_is_dropped_and_announced():
    events = EventBus()
    calls = []

    def broken(event):
        calls.append(event)
        raise ValueError("no canvas")

    events.subscribe(broken, GenerationStarted)
    notes = listen(events, Note)

    events.publish(GenerationStarted(1))
    events.publish(GenerationStarted(2))

    assert calls == [GenerationStarted(1)]
    assert len(notes) == 1
    assert "broken" in notes[0].message and "ValueError: no canvas" in notes[0].message


def test_a_failing_record_keeper_stops_the_publisher():
    events = EventBus()

    def broken(event):
        raise OSError("disk full")

    events.subscribe(broken, keeps_record=True)
    views = listen(events)
    with pytest.raises(OSError):
        events.publish(Note("x"))
    assert views == []


def test_events_serialise_with_their_kind():
    finished = RunFinished(generation=3, episodes=30, best_score=7, reason="limit")
    assert finished.to_dict() == {
        "kind": "run_finished", "generation": 3, "episodes": 30, "best_score": 7,
        "reason": "limit", "error": None,
    }


# ---------------------------------------------------------------------- trainer


def test_trainer_announces_a_run_in_order(tmp_path):
    events = EventBus()
    heard = listen(events)
    trainer = Trainer(make_config(tmp_path), run_id="order", seed=2, events=events,
                      max_generations=3)
    trainer.start()
    trainer.join()
    trainer.stop()

    kinds = [type(e) for e in heard if not isinstance(e, (RecordSet, CheckpointSaved))]
    assert kinds == [
        RunStarted,
        GenerationStarted, GenerationEvaluated,
        GenerationStarted, GenerationEvaluated,
        GenerationStarted, GenerationEvaluated,
        RunFinished,
    ]
    assert [e.generation for e in heard if isinstance(e, GenerationStarted)] == [0, 1, 2]
    # Reports carry the number of the generation that played, the population the next one.
    assert [e.report.generation for e in heard if isinstance(e, GenerationEvaluated)] == [0, 1, 2]

    # A record follows the generation that set it. The checkpoint it caused
    # comes in between: record keepers act on an event before views hear it.
    for index, event in enumerate(heard):
        if isinstance(event, RecordSet):
            assert isinstance(heard[index - 1], CheckpointSaved)
            assert heard[index - 1].trigger == "record"
            assert isinstance(heard[index - 2], GenerationEvaluated)
            assert heard[index - 2].report.record

    finished = heard[-1]
    assert finished.reason == "limit" and finished.error is None
    assert finished.generation == 3


def test_the_end_of_a_run_is_saved_and_announced_before_it_finishes(tmp_path):
    events = EventBus()
    heard = listen(events)
    trainer = Trainer(make_config(tmp_path), run_id="end", seed=3, events=events,
                      max_generations=2)
    trainer.start()
    trainer.join()
    trainer.stop()

    latest = trainer.checkpoint_dir / "latest.npz"
    assert checkpoint.inspect(latest).generation == trainer.population.generation == 2
    assert isinstance(heard[-1], RunFinished)
    saved = [e.generation for e in heard[:-1] if isinstance(e, CheckpointSaved)]
    assert saved[-1] == 2
    assert trainer.history.latest().generation == 1


def test_a_requested_stop_finishes_the_generation_and_saves_it(tmp_path):
    events = EventBus()
    trainer = Trainer(make_config(tmp_path), run_id="stop", seed=4, events=events)
    heard = listen(events)
    events.subscribe(lambda e: trainer.request_stop(), GenerationStarted)
    trainer.start()
    trainer.join()
    trainer.stop()

    finished = heard[-1]
    assert isinstance(finished, RunFinished) and finished.reason == "stopped"
    assert finished.generation == trainer.population.generation == 1
    assert checkpoint.inspect(trainer.checkpoint_dir / "latest.npz").generation == 1


def test_a_failing_view_does_not_end_the_run(tmp_path):
    events = EventBus()
    heard = listen(events)

    def broken(event):
        raise RuntimeError("chart failed")

    events.subscribe(broken, GenerationEvaluated)
    trainer = Trainer(make_config(tmp_path), run_id="view", seed=5, events=events,
                      max_generations=2)
    trainer.start()
    trainer.join()
    trainer.stop()

    assert heard[-1].reason == "limit"
    assert any(isinstance(e, Note) and "chart failed" in e.message for e in heard)


def test_a_failing_record_keeper_ends_the_run_without_a_final_save(tmp_path):
    events = EventBus()
    heard = listen(events)

    def broken(event):
        raise OSError("disk full")

    events.subscribe(broken, GenerationEvaluated, keeps_record=True)
    trainer = Trainer(make_config(tmp_path), run_id="keeper", seed=6, events=events,
                      max_generations=2)

    # Run on this thread, where the error can be caught.
    with pytest.raises(OSError):
        trainer._loop()

    finished = heard[-1]
    assert finished.reason == "error" and finished.error == "OSError: disk full"
    assert not any(isinstance(e, CheckpointSaved) for e in heard)
    assert trainer.finished


def test_a_generation_cut_short_by_stop_is_not_saved(tmp_path):
    events = EventBus()
    heard = listen(events)
    trainer = Trainer(make_config(tmp_path), run_id="cut", seed=7, events=events)
    entered = threading.Event()
    release = threading.Event()

    # Stands in for a generation whose workers are shut down under it.
    def stuck(**kwargs):
        entered.set()
        release.wait(10)
        raise RuntimeError("workers went away")

    trainer.population.run_generation = stuck
    trainer.start()
    assert entered.wait(10)
    trainer.stop(timeout=0)
    release.set()
    trainer.join()

    finished = heard[-1]
    assert finished.reason == "abandoned" and finished.error is None
    assert list(trainer.checkpoint_dir.iterdir()) == []


def sigint_ignored(task) -> bool:
    return signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_workers_leave_ctrl_c_to_the_trainer(tmp_path):
    trainer = Trainer(make_config(tmp_path), run_id="workers", seed=8)
    pool = EvaluationPool(trainer.population.context, 2)
    try:
        assert pool.map([(), ()], run=sigint_ignored) == [True, True]
    finally:
        pool.close()


# ---------------------------------------------------------------------- stream


def test_a_page_continues_from_the_last_event_it_saw():
    log = EventLog()
    cursor, lost = log.cursor_for(None)
    assert (cursor, lost) == (0, False)

    log.add(Note("one"))
    log.add(Note("two"))
    cursor, messages, lost = log.read(cursor, timeout=0)
    assert not lost
    assert [m.split("\n")[0] for m in messages] == [f"id: {log.epoch}-1", f"id: {log.epoch}-2"]
    assert messages[1].endswith('data: {"kind":"note","message":"two"}\n\n')

    log.add(Note("three"))
    resumed, lost = log.cursor_for(f"{log.epoch}-2")
    assert not lost
    _, messages, _ = log.read(resumed, timeout=0)
    assert len(messages) == 1 and '"three"' in messages[0]


@pytest.mark.parametrize("last_id", ["0-1", "garbage", "{epoch}-9", "{epoch}-1"],
                         ids=["earlier-server", "unreadable", "ahead", "evicted"])
def test_a_gap_the_log_cannot_fill_is_reported(last_id):
    log = EventLog(capacity=2)
    for message in ("a", "b", "c", "d"):
        log.add(Note(message))
    cursor, lost = log.cursor_for(last_id.format(epoch=log.epoch))
    assert lost and cursor == 4


def test_a_reader_that_falls_behind_is_told_so():
    log = EventLog(capacity=2)
    cursor, _ = log.cursor_for(None)
    for message in ("a", "b", "c"):
        log.add(Note(message))
    cursor, messages, lost = log.read(cursor, timeout=0)
    assert lost and messages == [] and cursor == 3


def test_closing_the_log_wakes_a_waiting_reader():
    log = EventLog()
    done = threading.Event()

    def wait():
        log.read(0, timeout=30)
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    log.close()
    assert done.wait(5)


def test_the_dashboard_streams_events(monkeypatch):
    monkeypatch.setattr(export_model, "preload", lambda: None)
    trainer = types.SimpleNamespace(events=EventBus())
    server = DashboardServer(trainer, "127.0.0.1", 0)
    assert server.start()
    port = server._httpd.server_address[1]

    def lines(response):
        while True:
            yield response.fp.readline().decode("utf-8").rstrip("\n")

    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/api/events")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/event-stream")
        stream = lines(response)
        assert next(stream) == "retry: 2000"

        trainer.events.publish(Note("hello"))
        received = [next(stream) for _ in range(5)]
        assert received[1:4] == [
            f"id: {server.events.epoch}-1", "event: note",
            'data: {"kind":"note","message":"hello"}',
        ]
        connection.close()

        stale = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        stale.request("GET", "/api/events", headers={"Last-Event-ID": "0-1"})
        stream = lines(stale.getresponse())
        assert [next(stream) for _ in range(4)] == ["retry: 2000", "", "event: resync", "data: {}"]
        stale.close()
    finally:
        server.stop()
