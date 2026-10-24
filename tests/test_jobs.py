"""Several runs training at once, and the runs a restart cut off carried on.

The workers a dashboard has are shared between the runs it trains, and how
they are shared must never change what a run does. A run cut off part way is
carried on from its newest checkpoint, or from its seed when it had none, and
has to end exactly as the run that never stopped.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evolvarium.config import CONFIG
from evolvarium.events import GenerationEvaluated, Note
from evolvarium.evolution.pool import WorkerBudget
from evolvarium.games.snake.settings import BoardConfig, EpisodeConfig, SnakeSettings
from evolvarium.persistence import checkpoint, jobs
from evolvarium.telemetry.history import HistoryStore, read_summaries
from evolvarium.telemetry.reporter import ConsoleReporter
from evolvarium.trainer import Trainer
from evolvarium.web.runs import Lab

ROOT = Path(__file__).resolve().parents[1]
TOTAL = 5
SPLIT = 2


def make_config(tmp_path, workers: int = 1):
    # A checkpoint every SPLIT generations gives a cut-off run one to carry on
    # from, with generations played after it.
    return CONFIG.with_overrides(
        settings=SnakeSettings(
            board=BoardConfig(width=8, height=8),
            episode=EpisodeConfig(starvation_limit=40, max_steps=400),
        ),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=workers,
        ),
        checkpoint=dataclasses.replace(CONFIG.checkpoint, save_every=SPLIT),
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )


def train(trainer: Trainer) -> Trainer:
    trainer.start()
    trainer.join()
    trainer.stop()
    return trainer


def wait_for(check, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("timed out")


def played(run_dir: Path) -> list[dict]:
    """Every generation a run wrote, less what differs from one sitting to the next.

    Ids come from a counter shared by the whole process, and how long a
    generation took depends on the machine.
    """
    summaries = read_summaries(run_dir / "history.jsonl")
    for summary in summaries:
        summary.pop("elapsed")
        if summary["best_individual"] is not None:
            summary["best_individual"].pop("id")
    return summaries


def assert_same_population(a, b) -> None:
    assert a.generation == b.generation
    assert a.total_episodes == b.total_episodes
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    for x, y in zip(a.members, b.members, strict=True):
        assert np.array_equal(x.genome, y.genome)


def cut_off(config, run_id: str, seed: int | None, generations: int | None, *,
            reached: int, saved: int) -> Path:
    """A run a dashboard started, killed at generation `reached`.

    Its newest checkpoint is from generation `saved`, or it has none when
    `saved` is 0, and the generations after that are in its history all the same.
    """
    trainer = train(Trainer(config, run_id=run_id, seed=seed, max_generations=reached))
    for found in checkpoint.list_checkpoints(trainer.checkpoint_dir):
        if found.generation > saved:
            found.path.unlink()
    jobs.begin(trainer.run_dir, generations)
    return trainer.run_dir


def started_before(older: Path, newer: Path) -> None:
    past = time.time() - 60
    os.utime(older / "run.json", (past, past))
    assert jobs.unfinished(older.parent)[:2] == [older, newer]


def lock_is_free(run_dir: Path) -> bool:
    lock = jobs.JobLock(run_dir)
    free = lock.acquire()
    lock.release()
    return free


# ---------------------------------------------------------------------- workers


def test_the_workers_are_split_evenly_with_the_earliest_taking_what_is_left():
    budget = WorkerBudget(5)
    assert budget.join("a") == 5
    assert budget.join("b") == 2
    assert budget.share("a") == 3
    budget.join("c")
    assert [budget.share(key) for key in "abc"] == [2, 2, 1]
    budget.leave("a")
    assert [budget.share(key) for key in "bc"] == [3, 2]
    assert budget.jobs == 2


def test_a_job_wanting_fewer_workers_leaves_the_rest_to_the_others():
    budget = WorkerBudget(8)
    budget.join("small", 1)
    budget.join("open")
    budget.join("middle", 3)
    assert {key: budget.share(key) for key in ("small", "open", "middle")} == {
        "small": 1, "open": 4, "middle": 3,
    }
    # Asking twice keeps what was first asked for.
    assert budget.join("small", 6) == 1


def test_more_jobs_than_workers_each_get_one():
    budget = WorkerBudget(2)
    for key in "abc":
        budget.join(key)
    assert [budget.share(key) for key in "abc"] == [1, 1, 1]
    assert budget.share("gone") == 1
    assert WorkerBudget(0).total == 1


def test_a_run_whose_share_changes_plays_as_one_that_kept_its_workers(tmp_path):
    config = make_config(tmp_path, workers=2)
    alone = train(Trainer(config, run_id="alone", seed=6, max_generations=TOTAL))

    budget = WorkerBudget(2)
    shared = Trainer(config, run_id="shared", seed=6, max_generations=TOTAL, budget=budget)
    notes = []
    shared.events.subscribe(lambda event: notes.append(event.message), Note)

    # Another job takes a worker for two generations, part way through.
    def crowd(event):
        if event.report.generation == 1:
            budget.join("other")
        elif event.report.generation == 3:
            budget.leave("other")

    shared.events.subscribe(crowd, GenerationEvaluated)
    train(shared)

    assert played(shared.run_dir) == played(alone.run_dir)
    assert_same_population(shared.population, alone.population)
    assert notes == [
        "training on 1 worker from generation 2, as the runs here share the cores",
        "training on 2 workers from generation 4, as the runs here share the cores",
    ]
    assert budget.jobs == 0


def test_a_tagged_reporter_marks_every_line_of_its_run(tmp_path, capsys):
    trainer = Trainer(make_config(tmp_path), run_id="tagged", seed=1, max_generations=2)
    ConsoleReporter(colour=False, tag="2").attach(trainer.events)
    train(trainer)
    worded = [line for line in capsys.readouterr().out.splitlines()
              if any(c.isalpha() for c in line)]
    assert len(worded) >= 7
    assert all(line.startswith("  [2] ") for line in worded), worded

    ConsoleReporter(colour=False).note("on its own")
    assert capsys.readouterr().out == "  on its own\n"


# ---------------------------------------------------------------------- jobs


def test_history_forgets_every_generation_from_a_point(tmp_path):
    config = make_config(tmp_path)
    run = train(Trainer(config, run_id="long", seed=4, max_generations=4))
    store = HistoryStore(run.run_dir, config.history)
    assert store.truncate(2) == 2
    assert [s["generation"] for s in store.summaries()] == [0, 1]
    assert [s["generation"] for s in read_summaries(run.run_dir / "history.jsonl")] == [0, 1]
    replays = sorted(path.name for path in (run.run_dir / "replays").glob("gen_*.json.gz"))
    assert replays == ["gen_000000.json.gz", "gen_000001.json.gz"]
    assert store.truncate(2) == 0


def test_a_job_records_how_its_run_ended(tmp_path):
    assert jobs.read(tmp_path) is None
    assert jobs.update(tmp_path, state="stopped") is None
    assert not (tmp_path / jobs.JOB_FILE).exists()

    jobs.begin(tmp_path, 40)
    assert jobs.read(tmp_path) == {"generations": 40, "state": "training", "restarts": []}
    jobs.update(tmp_path, state="failed", error="ValueError: no")
    job = jobs.restarted(tmp_path, 12)
    assert job["state"] == "training" and "error" not in job
    assert [restart["generation"] for restart in job["restarts"]] == [12]
    assert jobs.read(tmp_path) == job

    (tmp_path / jobs.JOB_FILE).write_text('{"state": "paused"}', encoding="utf-8")
    assert jobs.read(tmp_path) is None
    (tmp_path / jobs.JOB_FILE).write_text("{", encoding="utf-8")
    assert jobs.read(tmp_path) is None


def test_only_the_runs_still_training_are_unfinished_oldest_first(tmp_path):
    now = time.time()
    for name, state, age in [("a", "training", 100), ("b", "finished", 400),
                             ("c", "training", 300), ("d", None, 500),
                             ("e", "stopped", 200), ("f", "failed", 600)]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}", encoding="utf-8")
        os.utime(run_dir / "run.json", (now - age, now - age))
        if state is not None:
            jobs.write(run_dir, {"generations": None, "state": state, "restarts": []})
    # A job with no run beside it is nothing to carry on.
    (tmp_path / "g").mkdir()
    jobs.begin(tmp_path / "g", None)

    assert jobs.unfinished(tmp_path) == [tmp_path / "c", tmp_path / "a"]


def test_a_job_lock_has_one_holder_at_a_time(tmp_path):
    first, second = jobs.JobLock(tmp_path), jobs.JobLock(tmp_path)
    assert first.acquire() and first.held
    assert first.acquire()
    assert not second.acquire() and not second.held
    first.release()
    first.release()
    assert second.acquire()
    second.release()


HOLD = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from evolvarium.persistence.jobs import JobLock\n"
    "lock = JobLock(Path(sys.argv[1]))\n"
    "print(lock.acquire(), flush=True)\n"
    "time.sleep(120)\n"
)


def test_a_job_lock_goes_with_a_process_that_is_killed(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)], cwd=ROOT, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "True"
        assert not lock_is_free(tmp_path)
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()
    # The system lets the lock go once the process is gone, though not
    # always at the same moment.
    wait_for(lambda: lock_is_free(tmp_path), timeout=10)


# ---------------------------------------------------------------------- carrying on


def test_a_run_cut_off_carries_on_as_the_run_that_never_stopped(tmp_path):
    config = make_config(tmp_path)
    straight = train(Trainer(config, run_id="straight", seed=5, max_generations=TOTAL))
    run_dir = cut_off(config, "cut", 5, TOTAL, reached=SPLIT + 1, saved=SPLIT)
    assert len(played(run_dir)) == SPLIT + 1

    trainers, heard = [], []
    lab = Lab(config, can_start=True, announce=trainers.append)
    lab.listen(lambda run_id, event: heard.append(event))
    try:
        assert lab.carry_on() == [{"run": "cut", "state": "carried", "generation": SPLIT}]
        wait_for(lambda: not lab.live)
    finally:
        lab.close()

    # The generation played after the checkpoint was played again, the same way.
    assert played(run_dir) == played(straight.run_dir)
    assert_same_population(trainers[0].population, straight.population)
    job = jobs.read(run_dir)
    assert job["state"] == "finished"
    assert [restart["generation"] for restart in job["restarts"]] == [SPLIT]
    notes = [event.message for event in heard if isinstance(event, Note)]
    assert f"carried on from generation {SPLIT}, where its newest checkpoint left it" in notes
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 5 and manifest["resumed_from"] is None


def test_a_run_cut_off_before_its_first_checkpoint_starts_again_from_its_seed(tmp_path):
    config = make_config(tmp_path)
    straight = train(Trainer(config, run_id="straight", seed=8, max_generations=TOTAL))
    run_dir = cut_off(config, "early", 8, TOTAL, reached=1, saved=0)
    assert checkpoint.list_checkpoints(run_dir / "checkpoints") == []

    trainers, heard = [], []
    lab = Lab(config, can_start=True, announce=trainers.append)
    lab.listen(lambda run_id, event: heard.append(event))
    try:
        assert lab.carry_on() == [{"run": "early", "state": "carried", "generation": 0}]
        wait_for(lambda: not lab.live)
    finally:
        lab.close()

    assert played(run_dir) == played(straight.run_dir)
    assert_same_population(trainers[0].population, straight.population)
    assert jobs.read(run_dir)["state"] == "finished"
    notes = [event.message for event in heard if isinstance(event, Note)]
    assert "started again from its seed, as it was cut off before its first checkpoint" in notes


def test_a_run_with_nothing_to_carry_on_from_is_marked_failed(tmp_path):
    config = make_config(tmp_path)
    run_dir = cut_off(config, "lost", None, TOTAL, reached=1, saved=0)
    lab = Lab(config, can_start=True)
    [outcome] = lab.carry_on()
    assert outcome == {
        "run": "lost", "state": "failed",
        "error": "ValueError: it has no checkpoint, and no seed to start it again from",
    }
    job = jobs.read(run_dir)
    assert job["state"] == "failed" and job["error"] == outcome["error"]
    assert lab.live == [] and lab.carry_on() == []
    assert lock_is_free(run_dir)


def test_a_run_another_dashboard_holds_is_left_to_it(tmp_path):
    config = make_config(tmp_path)
    run_dir = cut_off(config, "held", 3, None, reached=1, saved=1)
    elsewhere = jobs.JobLock(run_dir)
    assert elsewhere.acquire()

    lab = Lab(config, can_start=True)
    try:
        assert lab.carry_on() == []
        elsewhere.release()
        assert [outcome["state"] for outcome in lab.carry_on()] == ["carried"]
        # Carried on once, however often it is asked.
        assert lab.carry_on() == []
        assert not lock_is_free(run_dir)
    finally:
        lab.close()

    # Closing the dashboard leaves the run to the next one.
    assert jobs.read(run_dir)["state"] == "training" and lock_is_free(run_dir)
    again = Lab(config, can_start=True)
    try:
        assert [outcome["state"] for outcome in again.carry_on()] == ["carried"]
    finally:
        again.close()
    assert len(jobs.read(run_dir)["restarts"]) == 2


def test_runs_wait_for_a_worker_in_the_order_they_started(tmp_path):
    config = make_config(tmp_path)
    older = cut_off(config, "older", 1, None, reached=1, saved=1)
    newer = cut_off(config, "newer", 2, SPLIT + 1, reached=SPLIT, saved=SPLIT)
    started_before(older, newer)

    lab = Lab(config, can_start=True)
    try:
        assert lab.carry_on() == [
            {"run": "older", "state": "carried", "generation": 1},
            {"run": "newer", "state": "waiting", "generation": SPLIT},
        ]
        assert lab.waiting == ["newer"]
        rows = {row["run_id"]: row for row in lab.runs()}
        assert rows["newer"]["waiting"] is True and rows["older"]["waiting"] is False
        assert lab.start("snake", {})["problems"] == [
            "older is still training, and the dashboard has one worker; stop it first"
        ]

        # The worker the first run gives up goes to the one waiting.
        assert lab.stop("older")["ok"]
        wait_for(lambda: not lab.live and not lab.waiting)
    finally:
        lab.close()
    assert jobs.read(older)["state"] == "stopped"
    assert jobs.read(newer)["state"] == "finished"
    assert len(played(newer)) == SPLIT + 1


def test_a_run_waiting_for_a_worker_can_be_kept_from_carrying_on(tmp_path):
    config = make_config(tmp_path)
    first = cut_off(config, "first", 1, None, reached=1, saved=1)
    second = cut_off(config, "second", 2, None, reached=1, saved=1)
    third = cut_off(config, "third", 3, None, reached=1, saved=1)
    past = time.time()
    for age, run_dir in ((30, first), (20, second), (10, third)):
        os.utime(run_dir / "run.json", (past - age, past - age))

    lab = Lab(config, can_start=True)
    try:
        lab.carry_on()
        assert lab.waiting == ["second", "third"]
        assert lab.stop("second") == {"ok": True, "run": "second"}
        assert lab.waiting == ["third"]
        assert jobs.read(second)["state"] == "stopped" and lock_is_free(second)
        assert "error" in lab.stop("second")
    finally:
        lab.close()

    # A dashboard closing leaves both the run it trained and the one waiting
    # for the next dashboard.
    for run_dir in (first, third):
        assert jobs.read(run_dir)["state"] == "training" and lock_is_free(run_dir)
    assert lab.waiting == [] and lab.carry_on() == []


def test_the_dashboard_trains_a_run_on_each_of_its_workers(tmp_path):
    config = make_config(tmp_path, workers=2)
    runs_dir = config.paths.runs_dir
    lab = Lab(config, can_start=True)
    try:
        first = lab.start("snake", {})
        second = lab.start("snake", {})
        assert first.get("ok") and second.get("ok"), (first, second)
        assert [run.run_id for run in lab.live] == [first["run"], second["run"]]
        assert lab.start("snake", {})["problems"] == [
            "the dashboard trains 2 runs at once, one on each of its workers, and has 2; "
            "stop one first"
        ]
        # With two training, the page has to name the one it means.
        assert lab.training(None) is None
        assert lab.training(second["run"]).run_id == second["run"]

        # The first run had both workers until the second came.
        wait_for(lambda: [run.trainer.workers for run in lab.live] == [1, 1])

        assert lab.stop(first["run"])["ok"]
        wait_for(lambda: len(lab.live) == 1)
        assert jobs.read(runs_dir / first["run"])["state"] == "stopped"
        third = lab.start("snake", {}, generations=1)
        assert third.get("ok"), third
        wait_for(lambda: [run.run_id for run in lab.live] == [second["run"]])
        assert jobs.read(runs_dir / third["run"])["state"] == "finished"
    finally:
        lab.close()

    assert jobs.read(runs_dir / second["run"])["state"] == "training"
    assert lab.start("snake", {})["problems"] == ["the dashboard is shutting down"]
