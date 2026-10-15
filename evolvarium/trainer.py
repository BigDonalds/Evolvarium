"""Training loop.

Owns the population and the hold gate, runs generations on its own thread so
the dashboard stays responsive, and announces what happens on `events`. History
and checkpoints subscribe as the run's record; the console and the dashboard
subscribe as views. Nothing here calls any of them directly.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from datetime import datetime
from pathlib import Path

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
from .evolution.pool import EvaluationPool, resolve_workers
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
    ):
        self.config = config
        self.run_id = run_id or new_run_id()
        self.run_dir = config.paths.runs_dir / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(seed, resume_from)

        # Views that subscribe before this point also hear how the run was set
        # up, such as a resume that cannot continue exactly.
        self.events = events or EventBus()
        self.gate = TrainingGate(
            viewer_timeout=config.server.viewer_timeout,
            hold_enabled=config.server.hold_when_viewing,
        )

        if resume_from is not None:
            self.population = checkpoint.load(resume_from, config, seed=seed)
            self.resumed_from: str | None = str(resume_from)
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
            self.resumed_from = None

        self.history = HistoryStore(self.run_dir, config.history)
        self.history.attach(self.events)
        self.checkpoints = checkpoint.CheckpointWriter(
            self.population, self.checkpoint_dir, config.checkpoint
        )
        self.checkpoints.attach(self.events)

        self.workers = resolve_workers(config.evolution.workers)
        self.pool: EvaluationPool | None = None

        self.max_generations = max_generations
        self.started_at = time.time()
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
        # Built once and kept for the whole run: on Windows each worker is a
        # fresh interpreter, so creating a pool per generation would cost more
        # than it saves.
        if self.workers > 1 and self.pool is None:
            self.pool = EvaluationPool(self.population.context, self.workers)
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
        if self.pool is not None:
            self.pool.close()
            self.pool = None
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

    def save_latest(self) -> checkpoint.CheckpointInfo:
        return self.checkpoints.save_latest()

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        population = self.population
        report = self._last_report
        elapsed = time.time() - self.started_at
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
                round(population.generation / (elapsed / 60.0), 2) if elapsed > 1 else 0.0
            ),
            "episodes_per_second": (
                round(population.total_episodes / elapsed, 1) if elapsed > 1 else 0.0
            ),
            "feature_spec": population.feature_spec.name,
            "input_size": population.feature_spec.size,
            "genome_size": population.architecture.genome_size,
            "board": {"width": self.config.board.width, "height": self.config.board.height},
            "resumed_from": self.resumed_from,
            "workers": self.workers,
            "last_generation": report.to_dict() if report else None,
            "gate": self.gate.state().to_dict(),
        }
