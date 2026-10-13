"""Training loop.

Owns the population, writes history and checkpoints, and observes the hold gate
between generations. Runs on its own thread so the dashboard stays responsive
while evolution proceeds.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .evolution.pool import EvaluationPool, resolve_workers
from .evolution.population import GenerationReport, Population
from .persistence import checkpoint
from .telemetry.history import HistoryStore
from .telemetry.reporter import ConsoleReporter
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
        reporter: ConsoleReporter | None = None,
        max_generations: int | None = None,
    ):
        self.config = config
        self.run_id = run_id or new_run_id()
        self.run_dir = config.paths.runs_dir / self.run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(seed, resume_from)

        self.reporter = reporter or ConsoleReporter()
        self.history = HistoryStore(self.run_dir, config.history)
        self.gate = TrainingGate(
            viewer_timeout=config.server.viewer_timeout,
            hold_enabled=config.server.hold_when_viewing,
        )

        if resume_from is not None:
            self.population = checkpoint.load(resume_from, config, seed=seed)
            self.resumed_from: str | None = str(resume_from)
        else:
            self.population = Population(config, seed=seed)
            self.resumed_from = None

        self.workers = resolve_workers(config.evolution.workers)
        self.pool: EvaluationPool | None = None

        self.max_generations = max_generations
        self.started_at = time.time()
        self.running = False
        self.finished = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_report: GenerationReport | None = None

    def _write_manifest(self, seed: int | None, resume_from: Path | None) -> None:
        """Record what produced this run, before it produces anything.

        Checkpoints appear only on records and every 25 generations, and they do
        not carry the seed. Two runs that differ only by seed were otherwise
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

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self.gate.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.running = False

    def join(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.4)

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if (self.max_generations is not None
                        and self.population.generation >= self.max_generations):
                    break
                self._run_one_generation()
                self.gate.wait_at_boundary(
                    self.population.generation,
                    on_hold=self.reporter.held,
                    on_resume=self.reporter.resumed,
                )
        except Exception as error:  # surfaced rather than dying silently
            self.reporter.note(f"training stopped: {type(error).__name__}: {error}")
            raise
        finally:
            self.running = False
            self.finished = True

    def _run_one_generation(self) -> None:
        report = self.population.run_generation(record=True, pool=self.pool)
        self._last_report = report

        # Replays were packed wherever the genome was scored, so this is a cheap
        # assembly step rather than a second pass over the episodes.
        agents = []
        for member in report.evaluated:
            payload = member.summary()
            payload["replay"] = member.replay
            agents.append(payload)

        limit = self.config.history.agents_per_generation
        agents.sort(key=lambda a: a["fitness"], reverse=True)
        if limit > 0:
            agents = agents[:limit]

        best_genome = self.population.best_ever_genome
        genome_list = [float(v) for v in best_genome] if best_genome is not None else None
        self.history.append(report, agents, genome_list)

        self.reporter.generation(report)
        self._maybe_checkpoint(report)

    def _maybe_checkpoint(self, report: GenerationReport) -> None:
        every = self.config.checkpoint.save_every
        generation = self.population.generation
        if every > 0 and generation % every == 0:
            checkpoint.save(self.population, self.checkpoint_dir / f"gen_{generation:06d}.npz")
            checkpoint.prune(self.checkpoint_dir, self.config.checkpoint.keep_last)
        if report.record:
            self.save_latest()

    def save_latest(self) -> checkpoint.CheckpointInfo:
        return checkpoint.save(self.population, self.checkpoint_dir / "latest.npz")

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
