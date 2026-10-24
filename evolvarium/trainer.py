"""Training loop.

Owns the population and the hold gate, runs generations on its own thread so
the dashboard stays responsive, and announces what happens on `events`. History
and checkpoints subscribe as the run's record; the console and the dashboard
subscribe as views. Nothing here calls any of them directly.

Several trainers can share one machine through a `WorkerBudget`. Each takes up
its share of the workers between generations, so a run grows when another ends
and shrinks when another starts, without that changing what it evolves.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from . import games, settings
from .config import Config
from .events import (
    EventBus,
    GenerationEvaluated,
    GenerationStarted,
    Held,
    Note,
    RecordSet,
    Resumed,
    RunFinished,
    RunStarted,
)
from .evolution.pool import EvaluationPool, WorkerBudget, resolve_workers
from .evolution.population import GenerationReport, Population
from .persistence import checkpoint
from .telemetry.history import HistoryStore
from .web.control import TrainingGate


def new_run_id() -> str:
    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        run_id: str | None = None,
        seed: int | None = None,
        resume_from: Path | None = None,
        events: EventBus | None = None,
        max_generations: int | None = None,
        budget: WorkerBudget | None = None,
        carry_on: bool = False,
    ):
        """`carry_on` picks the run called `run_id` up in its own directory.

        Its manifest stands, `resume_from` is one of its own checkpoints or None
        to start it again from `seed`, and every generation after that point is
        dropped from its history to be played again.
        """
        # The game's settings are filled in first, so the manifest records the
        # rules the run actually plays under. Settings that cannot train stop
        # the run here, before it has a directory.
        config = games.resolve(config)
        settings.check(config)
        self.config = config
        if carry_on and run_id is None:
            raise ValueError("carrying a run on needs its run id")
        self.run_id = run_id or new_run_id()
        self.run_dir = config.paths.runs_dir / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if carry_on:
            manifest = _read_manifest(self.run_dir / "run.json")
            self.resumed_from: str | None = manifest.get("resumed_from")
        else:
            self._write_manifest(seed, resume_from)
            self.resumed_from = None if resume_from is None else str(resume_from)

        # Views that subscribe before this point also hear how the run was set
        # up, such as a resume that cannot continue exactly.
        self.events = events or EventBus()
        self.gate = TrainingGate(
            viewer_timeout=config.server.viewer_timeout,
            hold_enabled=config.server.hold_when_viewing,
        )

        if resume_from is not None:
            self.population = checkpoint.load(resume_from, config, seed=seed)
            if not checkpoint.restores_exactly(resume_from):
                self.events.publish(Note(
                    "checkpoint has no saved random state; this run will not "
                    "match one that never stopped"
                ))
            elif seed is not None:
                self.events.publish(
                    Note(f"seed {seed} unused: the checkpoint's random state continues")
                )
        else:
            self.population = Population(config, seed=seed)

        self.history = HistoryStore(self.run_dir, config.history)
        if carry_on:
            self.history.truncate(self.population.generation)
        self.history.attach(self.events)
        self.checkpoints = checkpoint.CheckpointWriter(
            self.population, self.checkpoint_dir, config.checkpoint
        )
        self.checkpoints.attach(self.events)

        self.budget = budget
        self.workers = resolve_workers(config.evolution.workers)
        self.pool: EvaluationPool | None = None
        # The page's thread can stop a run while its own thread resizes the pool.
        self._pool_lock = threading.Lock()

        self.max_generations = max_generations
        self.started_at = time.time()
        # Rates count from here, so a run picked up part way is not credited
        # with the generations before it.
        self._first_generation = self.population.generation
        self._first_episodes = self.population.total_episodes
        self.running = False
        self.finished = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._abandoned = False
        self._last_report: GenerationReport | None = None

    def _write_manifest(self, seed: int | None, resume_from: Path | None) -> None:
        """Record what produced this run, before it produces anything.

        Checkpoints appear only on records, every 25 generations and at the end,
        and they do not carry the seed. Two runs that differ only by seed were otherwise
        distinguishable only by launch order, which parallel launches do not
        guarantee.
        """
        manifest = {
            "run_id": self.run_id,
            "seed": seed,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "resumed_from": str(resume_from) if resume_from else None,
            "config": dataclasses.asdict(self.config),
        }
        path = self.run_dir / "run.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.budget is not None:
            self.workers = self.budget.join(self.run_id, self.config.evolution.workers)
        # Built once and kept until the run's share changes: on Windows each
        # worker is a fresh interpreter, so creating a pool per generation would
        # cost more than it saves.
        with self._pool_lock:
            if self.pool is None:
                self.pool = self._new_pool()
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="trainer", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """End the run after the generation in progress, without waiting for it."""
        self._stop.set()
        self.gate.stop()

    def stop(self, timeout: float | None = 10.0) -> None:
        """Stop, wait up to `timeout` for the loop, then shut the workers down.

        A generation still running when the workers go is abandoned, and the run
        ends without the final checkpoint.
        """
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._abandoned = self._thread.is_alive()
        with self._pool_lock:
            if self.pool is not None:
                self.pool.close()
                self.pool = None
        self._leave_budget()
        self.running = False

    def join(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.4)

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        failure: Exception | None = None
        reason = "error"
        try:
            reason = self._run()
        except Exception as error:  # surfaced rather than dying silently
            if self._abandoned:
                reason = "abandoned"
            else:
                failure = error

        population = self.population
        try:
            self.events.publish(RunFinished(
                generation=population.generation,
                episodes=population.total_episodes,
                best_score=population.best_ever_score,
                reason=reason,
                error=None if failure is None else f"{type(failure).__name__}: {failure}",
            ))
        finally:
            self._leave_budget()
            self.running = False
            self.finished = True
        if failure is not None:
            raise failure

    def _run(self) -> str:
        population = self.population
        self.events.publish(RunStarted(self.run_id, population.generation, self.resumed_from))
        while not self._stop.is_set():
            if self.max_generations is not None and population.generation >= self.max_generations:
                return "limit"
            self._fit_pool()
            self._run_one_generation()
            self.gate.wait_at_boundary(
                population.generation,
                on_hold=lambda generation: self.events.publish(Held(generation)),
                on_resume=lambda generation, why: self.events.publish(Resumed(generation, why)),
            )
        return "stopped"

    def _run_one_generation(self) -> None:
        population = self.population
        self.events.publish(GenerationStarted(population.generation))
        report = population.run_generation(record=True, pool=self.pool)
        self._last_report = report

        self.events.publish(GenerationEvaluated(report, population.best_ever_genome))
        if report.record:
            self.events.publish(RecordSet(
                report.generation, population.best_ever_score, population.best_ever_fitness
            ))

    # ------------------------------------------------------------------ workers

    def _new_pool(self) -> EvaluationPool | None:
        shared = self.budget is not None and self.budget.total > 1
        if self.workers > 1 or shared:
            return EvaluationPool(self.population.context, self.workers, own_process=shared)
        return None

    def _fit_pool(self) -> None:
        """Take up the share of the budget this run has now, between generations."""
        if self.budget is None:
            return
        share = self.budget.share(self.run_id)
        if share == self.workers:
            return
        with self._pool_lock:
            if self._stop.is_set():
                return
            if self.pool is not None:
                self.pool.close(wait=True)
                self.pool = None
            self.workers = share
            self.pool = self._new_pool()
        plural = "" if share == 1 else "s"
        self.events.publish(Note(
            f"training on {share} worker{plural} from generation "
            f"{self.population.generation}, as the runs here share the cores"
        ))

    def _leave_budget(self) -> None:
        if self.budget is not None:
            self.budget.leave(self.run_id)

    def save_latest(self) -> checkpoint.CheckpointInfo:
        return self.checkpoints.save_latest()

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        population = self.population
        report = self._last_report
        elapsed = time.time() - self.started_at
        generations = population.generation - self._first_generation
        episodes = population.total_episodes - self._first_episodes
        return {
            "run_id": self.run_id,
            "running": self.running,
            "finished": self.finished,
            "generation": population.generation,
            "population_size": len(population.members),
            "best_ever_score": population.best_ever_score,
            "best_ever_fitness": round(population.best_ever_fitness, 2),
            "best_ever_generation": population.best_ever_generation,
            "total_episodes": population.total_episodes,
            "stagnation": population.stagnation,
            "elapsed_seconds": round(elapsed, 1),
            "generations_per_minute": (
                round(generations / (elapsed / 60.0), 2) if elapsed > 1 else 0.0
            ),
            "episodes_per_second": round(episodes / elapsed, 1) if elapsed > 1 else 0.0,
            "game": self.config.game,
            "direction": population.game.direction,
            "encoding": population.game.model_inputs()["encoding"],
            "input_size": population.game.observation.size,
            "actions": list(population.game.actions.names),
            "genome_size": population.architecture.genome_size,
            "scene": population.scene,
            "resumed_from": self.resumed_from,
            "workers": self.workers,
            "last_generation": report.to_dict() if report else None,
            "gate": self.gate.state().to_dict(),
        }


def _read_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a manifest")
    return data
