"""The runs a dashboard shows: those training in this process and those on disk.

The dashboard that `train` starts shows that run. The one `serve` starts can
also start runs of any game, as many at once as it has workers, and reads every
run in the runs directory back from what the run left there: its manifest, its
history, its replays and its newest checkpoint. A run training here and a run
read from disk answer the same questions, so the server does not need to know
which it has.

A run started here records in `job.json` that it is meant to be training. When
the dashboard is stopped or killed part way through, the next one on the same
runs directory finds those runs and carries each on from its newest checkpoint.

Choosing a model by its average plays a few thousand games. It runs as a job on
its own thread, one at a time, and the page polls it for progress.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import games, settings
from ..config import CONFIG, SECTIONS, Config, HistoryConfig
from ..contract import load_settings
from ..events import Event, EventBus, GenerationEvaluated, Note, RunFinished
from ..evolution.evaluation import EvalContext
from ..evolution.pool import WorkerBudget, resolve_workers
from ..held_out import choose_by_average, describe, gather_candidates, search_settings
from ..nn.network import Architecture
from ..persistence import checkpoint, export_model, jobs
from ..settings import SettingsError
from ..telemetry.history import HistoryStore, lineage_summaries

if TYPE_CHECKING:  # the trainer imports the web package, so it is imported late
    from ..trainer import Trainer

# Sections that belong to the dashboard rather than to a run. Every run it
# starts uses the dashboard's own, so the form does not offer them.
LAB_SECTIONS = ("server", "paths", "held_out")

# A run read from disk keeps this many generations parsed in memory. A late
# generation is several megabytes once parsed, and only the live run is
# scrubbed through quickly.
SAVED_MEMORY_WINDOW = 4
# Past runs kept open, most recently used first.
SAVED_OPEN = 8

_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _stamp(run_dir: Path) -> tuple:
    """Changes whenever a run writes something the dashboard reads from it."""
    parts = []
    for name in ("run.json", "history.jsonl", "checkpoints/latest.npz", jobs.JOB_FILE):
        try:
            info = (run_dir / name).stat()
        except OSError:
            parts.append(None)
        else:
            parts.append((info.st_mtime_ns, info.st_size))
    return tuple(parts)


def _updated_at(run_dir: Path) -> float | None:
    times = []
    for name in ("run.json", "history.jsonl"):
        try:
            times.append((run_dir / name).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _label(manifest: dict) -> str:
    """The run's configuration, named by how it differs from the defaults."""
    config = manifest.get("config")
    if not isinstance(config, dict):
        return "unknown settings"
    try:
        return describe(search_settings(config))
    except (AttributeError, KeyError, TypeError, ValueError):
        return "unknown settings"


class RunView:
    """What the dashboard asks of a run, whether it trains here or was read from disk."""

    live = False
    run_id: str
    run_dir: Path
    history: HistoryStore
    manifest: dict
    game = None
    scene: dict | None = None
    architecture: Architecture | None = None
    direction = "higher"
    problem: str | None = None

    @property
    def label(self) -> str:
        return _label(self.manifest)

    def status(self) -> dict:
        raise NotImplementedError

    def current_generation(self) -> int:
        raise NotImplementedError

    def newest_champion(self) -> tuple[int, np.ndarray | None]:
        raise NotImplementedError

    def candidates(self) -> list[tuple[str, np.ndarray]]:
        raise NotImplementedError

    def context(self) -> EvalContext:
        raise NotImplementedError

    def export_generation(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ history

    def curve(self) -> dict:
        return self.history.curve()

    def generations(self) -> dict:
        available = self.history.available()
        first, last = self.history.bounds()
        return {
            "available": available,
            "first": first,
            "last": last,
            "current": self.current_generation(),
            "retained": len(available),
        }

    def generation(self, number: int | None) -> dict:
        record = self.history.latest() if number is None else self.history.get(number)
        if record is None:
            return {"error": "generation not available", "generation": number}
        return {
            "run": self.run_id,
            "generation": record.generation,
            "summary": record.summary,
            "agents": record.agents,
            "scene": self.scene,
        }

    def checkpoints(self) -> dict:
        found = checkpoint.list_checkpoints(self.run_dir / "checkpoints")
        return {"checkpoints": [c.to_dict() for c in found]}

    def job(self) -> dict:
        """What the dashboard that started the run last wrote of it, or nothing."""
        return jobs.read(self.run_dir) or {}

    def entry(self) -> dict:
        """The run's line in the list of runs."""
        status = self.status()
        job = self.job()
        return {
            "run_id": self.run_id,
            "game": status["game"],
            "direction": self.direction,
            "label": self.label,
            "seed": self.manifest.get("seed"),
            "started_at": self.manifest.get("started_at"),
            "resumed_from": self.manifest.get("resumed_from"),
            "generations": status["generation"],
            "best": status["best_ever_score"],
            "live": self.live,
            "running": status["running"],
            "updated_at": None if self.live else _updated_at(self.run_dir),
            "problem": self.problem,
            "job_state": job.get("state"),
            "job_error": job.get("error"),
        }

    def comparison(self) -> dict:
        """The run's curves and totals, for setting beside other runs."""
        status = self.status()
        curve = self.history.curve()
        recent = curve["mean_score"][-10:]
        return {
            "run_id": self.run_id,
            "label": self.label,
            "game": status["game"],
            "direction": self.direction,
            "seed": self.manifest.get("seed"),
            "live": self.live,
            "generations": status["generation"],
            "best": status["best_ever_score"],
            "best_generation": status["best_ever_generation"],
            "episodes": status["total_episodes"],
            "seconds": status["elapsed_seconds"],
            "episodes_per_second": status["episodes_per_second"],
            "recent_mean": float(np.mean(recent)) if recent else None,
            "curve": {
                key: curve[key]
                for key in ("generation", "max_episode_score", "best_score", "mean_score")
            },
        }

    # ------------------------------------------------------------------ export

    def _fits(self, genome) -> np.ndarray | None:
        """`genome` as float32, or None when it is not one for this run's network."""
        if genome is None or self.architecture is None:
            return None
        genome = np.asarray(genome, dtype=np.float32)
        return genome if genome.size == self.architecture.genome_size else None

    def champion(self, generation: int | None) -> tuple[int, np.ndarray | None]:
        """The champion as of `generation`, or the newest, and the generation it is from."""
        if generation is None:
            return self.newest_champion()
        record = self.history.get(generation)
        if record is not None:
            genome = self._fits(record.best_genome)
            if genome is not None:
                return generation, genome
        # The most recent archived generation at or before the request, so a
        # request for an evicted generation still resolves.
        for candidate in reversed(self.history.available()):
            if candidate > generation:
                continue
            earlier = self.history.get(candidate)
            genome = None if earlier is None else self._fits(earlier.best_genome)
            if genome is not None:
                return candidate, genome
        return generation, None

    def report(self, generation: int, genome: np.ndarray | None, destination: Path) -> dict:
        return export_model.build_report(
            generation=generation,
            genome=genome,
            summaries=lineage_summaries(self.run_dir),
            game=self.game,
            architecture=self.architecture,
            run_id=self.run_id,
            destination=destination,
        )


class LiveRun(RunView):
    """A run training in this process."""

    live = True

    def __init__(self, trainer: Trainer, heard: Callable[[LiveRun, Event], None]):
        self.trainer = trainer
        self.run_id = trainer.run_id
        self.run_dir = trainer.run_dir
        self.history = trainer.history
        self.manifest = _read_json(self.run_dir / "run.json") or {}
        population = trainer.population
        self.game = population.game
        self.scene = population.scene
        self.architecture = population.architecture
        self.direction = population.game.direction
        self._heard = heard
        self._lock = threading.Lock()
        # The generation count, the genomes that played, their ages and the
        # champion, as of the last generation scored.
        self._last: tuple[int, np.ndarray, list[int], np.ndarray | None] | None = None
        self._unsubscribe = trainer.events.subscribe(self._hear)
        # The page's thread and the command line can both end a run.
        self._retiring = threading.Lock()

    def _hear(self, event: Event) -> None:
        if isinstance(event, GenerationEvaluated):
            # The population is bred again as soon as the trainer moves on, so
            # the genomes are copied for a choice made later.
            members = event.report.evaluated
            genomes = (
                np.stack([m.genome for m in members]).astype(np.float32) if members
                else np.zeros((0, self.architecture.genome_size), dtype=np.float32)
            )
            ages = [m.age for m in members]
            best = None if event.best_genome is None else np.array(
                event.best_genome, dtype=np.float32
            )
            with self._lock:
                # Counted as a checkpoint counts it: the generations finished.
                self._last = (event.report.generation + 1, genomes, ages, best)
        self._heard(self, event)

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def retire(self, timeout: float | None = 10.0) -> None:
        """Stop the trainer, waiting up to `timeout` for its generation, and stop listening."""
        with self._retiring:
            self.trainer.stop(timeout)
            self.close()

    def status(self) -> dict:
        status = self.trainer.status()
        status.update(
            live=True,
            label=self.label,
            seed=self.manifest.get("seed"),
            started_at=self.manifest.get("started_at"),
            updated_seconds_ago=0.0,
            problem=None,
        )
        return status

    def current_generation(self) -> int:
        return self.trainer.population.generation

    def newest_champion(self) -> tuple[int, np.ndarray | None]:
        population = self.trainer.population
        return population.generation, self._fits(population.best_ever_genome)

    def candidates(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            last = self._last
        if last is None:
            raise ValueError("no generation has finished yet")
        _, genomes, ages, best = last
        return gather_candidates(best, genomes, self.run_dir, ages)

    def context(self) -> EvalContext:
        return dataclasses.replace(self.trainer.population.context, exploration_rate=0.0)

    def export_generation(self) -> int:
        with self._lock:
            last = self._last
        return self.trainer.population.generation if last is None else last[0]


class SavedRun(RunView):
    """A run read back from its directory, finished or training somewhere else."""

    def __init__(self, run_dir: Path, history: HistoryConfig):
        self.run_id = run_dir.name
        self.run_dir = run_dir
        self.manifest = _read_json(run_dir / "run.json") or {}
        recorded = self.manifest.get("config")
        recorded = recorded if isinstance(recorded, dict) else {}
        self.latest = run_dir / "checkpoints" / "latest.npz"
        self.meta: dict | None = None
        if self.latest.is_file():
            try:
                self.meta = checkpoint.read_meta(self.latest)
            except (OSError, ValueError, KeyError):
                self.meta = None
        self.history = HistoryStore(run_dir, history)
        self.summaries = self.history.summaries()
        # A manifest older than the game contract has no game, and was Snake's.
        self.game_name = (
            checkpoint.saved_game(self.meta)[0] if self.meta
            else str(recorded.get("game", "snake"))
        )
        self.evolution = recorded.get("evolution") or {}
        self.settings = None
        try:
            cls = games.get(self.game_name)
            self.direction = cls.direction
            # Settings added since the run was recorded take their defaults.
            self.settings = load_settings(
                cls.Settings, checkpoint.saved_settings(self.meta or {}, recorded), strict=False
            )
            self.game = cls(self.settings)
            scene = self.game.scene()
            self.scene = None if scene is None else scene.describe()
            self.architecture = self._architecture(recorded)
        except Exception as error:  # a game that no longer loads is still listed
            self.problem = f"{type(error).__name__}: {error}"

    def _architecture(self, recorded: dict) -> Architecture | None:
        if self.meta and "architecture" in self.meta:
            return Architecture.from_dict(self.meta["architecture"])
        # A run stopped before its first checkpoint has only its manifest.
        network = recorded.get("network") or {}
        if "hidden_layers" not in network:
            return None
        return Architecture.from_dict({
            "input_size": self.game.observation.size,
            "hidden_layers": network["hidden_layers"],
            "hidden_activations": network.get("hidden_activations", ()),
            "output_size": self.game.actions.size,
        })

    def _best(self) -> tuple[float | None, float, int]:
        """The best score, its fitness and its generation."""
        meta = self.meta or {}
        if meta.get("best_ever_score") is not None:
            return (meta["best_ever_score"], float(meta.get("best_ever_fitness", 0.0)),
                    int(meta.get("best_ever_generation", 0)))
        records = [s for s in self.summaries if s.get("record") and s.get("best_individual")]
        if not records:
            return None, 0.0, 0
        last = records[-1]
        return (last["best_individual"]["best_episode_score"], float(last["best_fitness"]),
                int(last["generation"]))

    def current_generation(self) -> int:
        if self.summaries:
            return int(self.summaries[-1]["generation"]) + 1
        return int((self.meta or {}).get("generation", 0))

    def status(self) -> dict:
        summaries = self.summaries
        last = summaries[-1] if summaries else None
        episodes = sum(int(s.get("episodes", 0)) for s in summaries)
        # The time spent on generations, which leaves out holds and restarts.
        seconds = sum(float(s.get("elapsed", 0.0)) for s in summaries)
        best, fitness, best_generation = self._best()
        game = self.game
        updated = _updated_at(self.run_dir)
        return {
            "run_id": self.run_id,
            "running": False,
            "finished": True,
            "live": False,
            "generation": self.current_generation(),
            "population_size": self.evolution.get("population_size"),
            "best_ever_score": best,
            "best_ever_fitness": round(fitness, 2),
            "best_ever_generation": best_generation,
            "total_episodes": episodes,
            "stagnation": last.get("stagnation", 0) if last else 0,
            "elapsed_seconds": round(seconds, 1),
            "generations_per_minute": (
                round(len(summaries) / (seconds / 60.0), 2) if seconds > 1 else 0.0
            ),
            "episodes_per_second": round(episodes / seconds, 1) if seconds > 1 else 0.0,
            "game": self.game_name,
            "direction": self.direction,
            "encoding": None if game is None else game.model_inputs()["encoding"],
            "input_size": None if game is None else game.observation.size,
            "actions": None if game is None else list(game.actions.names),
            "genome_size": None if self.architecture is None else self.architecture.genome_size,
            "scene": self.scene,
            "resumed_from": self.manifest.get("resumed_from"),
            "workers": self.evolution.get("workers"),
            "last_generation": last,
            "gate": None,
            "label": self.label,
            "seed": self.manifest.get("seed"),
            "started_at": self.manifest.get("started_at"),
            "updated_seconds_ago": None if updated is None else round(time.time() - updated, 1),
            "problem": self.problem,
        }

    def _archive(self) -> tuple[np.ndarray | None, np.ndarray, list[int] | None]:
        """The newest checkpoint's champion, population and ages, or nothing if unreadable."""
        empty = np.zeros((0, 0), dtype=np.float32)
        if self.meta is None:
            return None, empty, None
        try:
            with np.load(self.latest, allow_pickle=False) as archive:
                champion = (
                    np.array(archive["best_genome"], dtype=np.float32)
                    if "best_genome" in archive.files else None
                )
                population = np.array(archive["genomes"], dtype=np.float32)
        except (OSError, ValueError, KeyError, EOFError):
            return None, empty, None
        ages = self.meta.get("ages")
        # Ages that don't line up with the population can't order it.
        if ages is not None and len(ages) != len(population):
            ages = None
        return champion, population, ages

    def newest_champion(self) -> tuple[int, np.ndarray | None]:
        genome = self._fits(self._archive()[0])
        if genome is not None:
            return int(self.meta.get("generation", 0)), genome
        available = self.history.available()
        if not available:
            return self.current_generation(), None
        return self.champion(available[-1])

    def candidates(self) -> list[tuple[str, np.ndarray]]:
        champion, population, ages = self._archive()
        found = gather_candidates(champion, population, self.run_dir, ages)
        return [(source, genome) for source, genome in found if self._fits(genome) is not None]

    def context(self) -> EvalContext:
        if self.game is None or self.architecture is None:
            raise ValueError(self.problem or "the run's network is not recorded")
        return EvalContext(
            game=self.game_name,
            settings=self.settings,
            architecture=self.architecture.to_dict(),
            exploration_rate=0.0,
            replay_steps=CONFIG.history.replay_steps,
        )

    def export_generation(self) -> int:
        if self.meta is not None:
            return int(self.meta.get("generation", 0))
        return self.current_generation()


class _Cancelled(Exception):
    """Raised from a choice's progress report to end it early."""


@dataclass(eq=False)
class ExportJob:
    """Choosing one run's genome by its average, on a thread of its own."""

    id: str
    run: str
    state: str = "running"   # "running", "done", "failed" or "cancelled"
    stage: str = "reading"   # "reading", "screen" or "final"
    played: int = 0
    games: int = 0
    candidates: int = 0
    error: str | None = None
    report: dict | None = None
    genome: np.ndarray | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run": self.run,
            "state": self.state,
            "stage": self.stage,
            "played": self.played,
            "games": self.games,
            "candidates": self.candidates,
            "error": self.error,
            "report": self.report,
            "seconds": round((self.finished or time.time()) - self.started, 1),
        }


@dataclass(eq=False)
class _Waiting:
    """A run a restart cut off, held until a worker comes free to carry it on."""

    run_dir: Path
    config: Config
    resume_from: Path | None
    seed: int | None
    generations: int | None
    generation: int


class Lab:
    """Every run under one runs directory, and the runs training here.

    `config` is what a new run starts from before the form changes it, and its
    paths, server and held-out sections hold for every run. `can_start` is
    whether the page may start runs. `announce(trainer)` is called for each run
    started or carried on, before it starts, so the command line can report it
    too.

    The runs started here share the workers `config` gives the dashboard, one
    run to a worker at most. Each holds its job's lock while it trains, so two
    dashboards on one runs directory never carry the same run on.
    """

    def __init__(self, config: Config, *, can_start: bool,
                 announce: Callable[[Trainer], None] | None = None):
        self.config = config
        self.can_start = can_start
        self._announce = announce
        self._lock = threading.RLock()
        self.budget = WorkerBudget(resolve_workers(config.evolution.workers))
        self._live: dict[str, LiveRun] = {}
        # Runs carried on as workers come free, oldest first.
        self._waiting: OrderedDict[str, _Waiting] = OrderedDict()
        # The job lock of every run started or carried on here, held until it ends.
        self._locks: dict[str, jobs.JobLock] = {}
        self._closing = False
        self._listeners: list[Callable[[str, Event], None]] = []
        self._open: OrderedDict[str, tuple[tuple, SavedRun]] = OrderedDict()
        self._entries: dict[str, tuple[tuple, dict]] = {}
        # When each run that trained here was let go.
        self._ended: dict[str, float] = {}
        self._job: ExportJob | None = None
        self._job_ids = itertools.count(1)
        self._saved_history = dataclasses.replace(
            config.history, memory_window=min(config.history.memory_window, SAVED_MEMORY_WINDOW)
        )

    @property
    def runs_dir(self) -> Path:
        return self.config.paths.runs_dir

    @property
    def exports_dir(self) -> Path:
        return self.config.paths.exports_dir

    @property
    def live(self) -> list[LiveRun]:
        """The runs training here, in the order they started."""
        with self._lock:
            return list(self._live.values())

    @property
    def waiting(self) -> list[str]:
        """The runs waiting for a worker, in the order they will be carried on."""
        with self._lock:
            return list(self._waiting)

    def training(self, name: str | None) -> LiveRun | None:
        """The run called `name` if it trains here. Without a name, the only one that does."""
        with self._lock:
            if name:
                return self._live.get(name)
            return next(iter(self._live.values())) if len(self._live) == 1 else None

    def config_for(self, game: str) -> Config:
        """The configuration a new run of `game` starts from."""
        if game == self.config.game:
            return self.config
        return dataclasses.replace(self.config, game=game, settings=None)

    def games(self) -> list[str]:
        known = games.names()
        return known if self.config.game in known else [*known, self.config.game]

    def settings_rows(self, game: str) -> list[dict]:
        return settings.table(games.resolve(self.config_for(game)), LAB_SECTIONS)

    # ------------------------------------------------------------------ events

    def listen(self, callback: Callable[[str, Event], None]) -> Callable[[], None]:
        """Call `callback(run_id, event)` with every event of every run trained here."""
        with self._lock:
            self._listeners.append(callback)

        def remove() -> None:
            with self._lock:
                if callback in self._listeners:
                    self._listeners.remove(callback)
        return remove

    def _heard(self, run: LiveRun, event: Event) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for callback in listeners:
            callback(run.run_id, event)
        if not isinstance(event, RunFinished):
            return
        self._ended_as(run, event)
        if self.can_start:
            # The trainer thread is the one announcing this, so it cannot be
            # joined from here.
            threading.Thread(
                target=self.retire, args=(run,), name="retire", daemon=True
            ).start()

    def _ended_as(self, run: LiveRun, event: RunFinished) -> None:
        """Write how a run started here ended.

        A run stopped from the page was written as stopped when it was asked to.
        One stopped by the dashboard's own end is left training, and that is
        what has the next dashboard carry it on.
        """
        if event.reason == "limit":
            self._write_job(run.run_id, state="finished")
        elif event.reason == "error":
            self._write_job(run.run_id, state="failed", error=event.error)

    def _write_job(self, name: str, **fields) -> None:
        with self._lock:
            if name not in self._locks:  # not a run this dashboard started
                return
            try:
                jobs.update(self.runs_dir / name, **fields)
            except OSError:
                # Left as it was. A job still training is carried on next time,
                # and a run that had reached its limit ends again at once.
                pass

    def retire(self, run: LiveRun, timeout: float | None = 10.0) -> None:
        """Stop `run` and let it go, so it is read from disk like any other."""
        run.retire(timeout)
        with self._lock:
            self._ended[run.run_id] = time.time()
            if self._live.get(run.run_id) is run:
                del self._live[run.run_id]
            lock = self._locks.pop(run.run_id, None)
        if lock is not None:
            lock.release()
        self._next()

    def ended_here(self, run: RunView) -> bool:
        """Whether `run` trained here and has not been written to since it ended."""
        if run.live:
            return False
        return self._still_ended(run.run_id, _updated_at(run.run_dir))

    def _still_ended(self, name: str, updated: float | None) -> bool:
        with self._lock:
            ended = self._ended.get(name)
        return ended is not None and (updated is None or updated <= ended)

    # ------------------------------------------------------------------ runs

    def adopt(self, trainer: Trainer) -> LiveRun:
        """Show `trainer` as training here. Called before it starts, so no event is missed."""
        run = LiveRun(trainer, self._heard)
        with self._lock:
            self._live[trainer.run_id] = run
        return run

    def _launch(self, trainer: Trainer, note: str | None = None) -> None:
        """Announce, show and start a run, with `note` the first thing it says."""
        if self._announce is not None:
            self._announce(trainer)
        self.adopt(trainer)
        if note is not None:
            trainer.events.publish(Note(note))
        trainer.start()

    def start(self, game: str, assignments: dict | list, *,
              generations: int | None = None, seed: int | None = None) -> dict:
        """Start a run of `game` with the settings in `assignments` changed.

        Returns the new run's id, or every problem that keeps it from starting.
        """
        if not self.can_start:
            return {"problems": [
                "this dashboard shows the run it was started with; "
                "the serve command opens one that starts runs"
            ]}
        pairs = list(assignments.items()) if isinstance(assignments, dict) else list(assignments)
        pairs = [(str(name), value) for name, value in pairs]
        problems = [
            f"{name} is the dashboard's to set, so every run shares it"
            for name, _ in pairs if name.partition(".")[0] in LAB_SECTIONS
        ]
        if generations is not None and (
            not _whole(generations) or generations < 1
        ):
            problems.append(f"generations takes a whole number of 1 or more, not {generations!r}")
        if seed is not None and not _whole(seed):
            problems.append(f"seed takes a whole number, not {seed!r}")
        config = None
        try:
            config = settings.apply(self.config_for(str(game)), pairs)
            settings.check(config, named=[name for name, _ in pairs], dashboard=True)
        except SettingsError as error:
            problems.extend(error.problems)
        except games.UnknownGame as error:
            problems.append(str(error))
        if problems:
            return {"problems": problems}

        from ..trainer import Trainer

        with self._lock:
            if self._closing:
                return {"problems": ["the dashboard is shutting down"]}
            busy = [*self._live, *self._waiting]
            if len(busy) >= self.budget.total:
                return {"problems": [self._full(busy)]}
            # A seed is always recorded, so any run started here can be repeated.
            seed = secrets.randbelow(2**31) if seed is None else seed
            try:
                trainer = Trainer(
                    config, run_id=self._new_id(), seed=seed, events=EventBus(),
                    max_generations=generations, budget=self.budget,
                )
            except (OSError, ValueError) as error:
                return {"problems": [f"{type(error).__name__}: {error}"]}
            lock = jobs.JobLock(trainer.run_dir)
            try:
                # The directory is new, so nothing else holds its lock.
                lock.acquire()
                jobs.begin(trainer.run_dir, generations)
            except OSError as error:
                lock.release()
                return {"problems": [f"{type(error).__name__}: {error}"]}
            self._locks[trainer.run_id] = lock
            self._launch(trainer)
        return {"ok": True, "run": trainer.run_id, "seed": seed}

    def _full(self, busy: list[str]) -> str:
        total = self.budget.total
        if total > 1:
            return (f"the dashboard trains {total} runs at once, one on each of its workers, "
                    f"and has {len(busy)}; stop one first")
        doing = "is still training" if busy[0] in self._live else "is waiting to carry on"
        return f"{busy[0]} {doing}, and the dashboard has one worker; stop it first"

    def _new_id(self) -> str:
        from ..trainer import new_run_id

        base = run_id = new_run_id()
        for number in itertools.count(2):
            if not (self.runs_dir / run_id).exists():
                return run_id
            run_id = f"{base}-{number}"
        raise AssertionError("unreachable")

    def stop(self, run_id: str | None) -> dict:
        """End a run training here after its generation, or keep a waiting one from starting."""
        with self._lock:
            if run_id in self._waiting:
                del self._waiting[run_id]
                self._write_job(run_id, state="stopped")
                self._locks.pop(run_id).release()
                return {"ok": True, "run": run_id}
            live = self.training(run_id)
            if live is None:
                return {"error": "that run is not training here", "run": run_id}
            self._write_job(live.run_id, state="stopped")
        live.trainer.request_stop()
        return {"ok": True, "run": live.run_id}

    # ------------------------------------------------------------------ carrying on

    def carry_on(self) -> list[dict]:
        """Carry on every run a restart cut off, as far as the workers go, and queue the rest.

        A run carries on from its newest checkpoint, or from its seed when it
        had none, and plays on as it would have had it never stopped. One that
        another dashboard is training is left to it. Says what became of each.
        """
        if not self.can_start:
            return []
        outcomes = []
        for run_dir in jobs.unfinished(self.runs_dir):
            name = run_dir.name
            lock = jobs.JobLock(run_dir)
            with self._lock:
                if self._closing:
                    break
                try:
                    if name in self._locks or not lock.acquire():
                        continue
                except OSError:
                    continue
                self._locks[name] = lock
                try:
                    plan = self._plan(run_dir)
                except Exception as error:  # reported, and the run is left failed
                    outcomes.append(self._fail(name, error))
                    continue
                if len(self._live) < self.budget.total:
                    outcomes.append(self._carry(plan))
                else:
                    self._waiting[name] = plan
                    outcomes.append(
                        {"run": name, "state": "waiting", "generation": plan.generation}
                    )
        return outcomes

    def _plan(self, run_dir: Path) -> _Waiting:
        manifest = _read_json(run_dir / "run.json")
        if manifest is None:
            raise ValueError("its run.json cannot be read")
        found = checkpoint.list_checkpoints(run_dir / "checkpoints")
        newest = found[-1] if found else None
        seed = manifest.get("seed")
        if newest is None and not _whole(seed):
            raise ValueError("it has no checkpoint, and no seed to start it again from")
        return _Waiting(
            run_dir=run_dir,
            config=self._recorded_config(manifest),
            resume_from=None if newest is None else newest.path,
            seed=None if newest is not None else seed,
            generations=(jobs.read(run_dir) or {}).get("generations"),
            generation=0 if newest is None else newest.generation,
        )

    def _recorded_config(self, manifest: dict) -> Config:
        """The configuration a run recorded, under this dashboard's own sections."""
        recorded = manifest.get("config")
        if not isinstance(recorded, dict):
            raise ValueError("its run.json records no configuration")
        game = str(recorded.get("game", "snake"))
        sections = {
            name: getattr(self.config, name) if name in LAB_SECTIONS
            else load_settings(type(getattr(self.config, name)), recorded.get(name), strict=False)
            for name in SECTIONS
        }
        # Settings added since the run was recorded take their defaults.
        return dataclasses.replace(
            self.config,
            game=game,
            settings=load_settings(
                games.get(game).Settings, checkpoint.saved_settings({}, recorded), strict=False
            ),
            **sections,
        )

    def _carry(self, plan: _Waiting) -> dict:
        from ..trainer import Trainer

        name = plan.run_dir.name
        try:
            trainer = Trainer(
                plan.config, run_id=name, seed=plan.seed, resume_from=plan.resume_from,
                events=EventBus(), max_generations=plan.generations, budget=self.budget,
                carry_on=True,
            )
            jobs.restarted(plan.run_dir, plan.generation)
        except Exception as error:  # reported, and the run is left failed
            return self._fail(name, error)
        if plan.resume_from is None:
            note = "started again from its seed, as it was cut off before its first checkpoint"
        else:
            note = (f"carried on from generation {plan.generation}, "
                    "where its newest checkpoint left it")
        self._launch(trainer, note)
        return {"run": name, "state": "carried", "generation": plan.generation}

    def _fail(self, name: str, error: Exception) -> dict:
        """Mark a run that cannot be carried on as failed, and let it go."""
        message = f"{type(error).__name__}: {error}"
        with self._lock:
            self._write_job(name, state="failed", error=message)
            lock = self._locks.pop(name, None)
        if lock is not None:
            lock.release()
        return {"run": name, "state": "failed", "error": message}

    def _next(self) -> None:
        """Carry on the runs waiting, as far as the workers that came free go."""
        while True:
            with self._lock:
                if self._closing or not self._waiting or len(self._live) >= self.budget.total:
                    return
                _, plan = self._waiting.popitem(last=False)
                self._carry(plan)

    def _names(self) -> list[str]:
        """Every run in the runs directory, newest first."""
        found = []
        for manifest in self.runs_dir.glob("*/run.json"):
            try:
                found.append((manifest.stat().st_mtime, manifest.parent.name))
            except OSError:
                continue
        return [name for _, name in sorted(found, reverse=True)]

    def run(self, name: str | None = None) -> RunView | None:
        """The run called `name`; without one, the newest run training here or else the newest."""
        if not name:
            live = self.live
            if live:
                return live[-1]
            names = self._names()
            return self._saved(names[0]) if names else None
        found = self.training(name)
        if found is not None:
            return found
        if not _RUN_NAME.fullmatch(name) or not (self.runs_dir / name / "run.json").is_file():
            return None
        return self._saved(name)

    def _saved(self, name: str) -> SavedRun:
        run_dir = self.runs_dir / name
        stamp = _stamp(run_dir)
        with self._lock:
            cached = self._open.get(name)
            if cached is not None and cached[0] == stamp:
                self._open.move_to_end(name)
                return cached[1]
        run = SavedRun(run_dir, self._saved_history)
        with self._lock:
            self._open[name] = (stamp, run)
            self._open.move_to_end(name)
            while len(self._open) > SAVED_OPEN:
                self._open.popitem(last=False)
        return run

    def runs(self) -> list[dict]:
        """A line for every run, newest first."""
        with self._lock:
            live = dict(self._live)
            waiting = set(self._waiting)
        rows = []
        now = time.time()
        for name in self._names():
            if name in live:
                rows.append({**live[name].entry(), "updated_seconds_ago": 0.0,
                             "ended_here": False, "waiting": False})
                continue
            stamp = _stamp(self.runs_dir / name)
            with self._lock:
                cached = self._entries.get(name)
            if cached is None or cached[0] != stamp:
                cached = (stamp, self._saved(name).entry())
                with self._lock:
                    self._entries[name] = cached
            entry = dict(cached[1])
            updated = entry.get("updated_at")
            entry["updated_seconds_ago"] = None if updated is None else round(now - updated, 1)
            entry["ended_here"] = self._still_ended(name, updated)
            entry["waiting"] = name in waiting
            rows.append(entry)
        return rows

    def compare(self, names: list[str]) -> dict:
        found, missing = [], []
        for name in dict.fromkeys(names):
            view = self.run(name)
            if view is None:
                missing.append(name)
            else:
                found.append(view.comparison())
        return {"runs": found, "missing": missing}

    # ------------------------------------------------------------------ export

    def destination(self, view: RunView) -> Path:
        return self.exports_dir / f"{view.run_id}.json"

    def preview(self, view: RunView, generation: int | None) -> dict:
        """The export report for the champion as of `generation`, writing nothing."""
        if view.architecture is None:
            return {"error": view.problem or "the run's network is not recorded"}
        resolved, genome = view.champion(generation)
        report = view.report(resolved, genome, self.destination(view))
        report["pick"] = "champion"
        report["requested_generation"] = generation
        report["resolved_generation"] = resolved
        return report

    def choose(self, view: RunView) -> dict:
        """Start choosing `view`'s genome by its average."""
        if view.architecture is None:
            return {"error": view.problem or "the run's network is not recorded"}
        with self._lock:
            if self._job is not None and self._job.state == "running":
                return {"error": f"already choosing a genome from {self._job.run}",
                        "job": self._job.to_dict()}
            job = ExportJob(id=str(next(self._job_ids)), run=view.run_id)
            self._job = job
        threading.Thread(
            target=self._choose, args=(job, view), name="choose", daemon=True
        ).start()
        return {"ok": True, "job": job.to_dict()}

    def _choose(self, job: ExportJob, view: RunView) -> None:
        def progress(stage: str, played: int, total: int) -> None:
            if job.cancel.is_set():
                raise _Cancelled
            job.stage, job.played, job.games = stage, played, total

        # The choice takes its share of the workers like a run, and the runs
        # training here give theirs up at their next generation.
        key = f"choice-{job.id}"
        workers = self.budget.join(key, self.config.evolution.workers)
        try:
            candidates = view.candidates()
            if not candidates:
                raise ValueError("the run holds no genome to choose from")
            job.candidates = len(candidates)
            choice = choose_by_average(candidates, view.context(), workers, progress=progress)
            generation = view.export_generation()
            report = view.report(generation, choice.genome, self.destination(view))
            report["pick"] = "average"
            report["selection"] = choice.to_dict()
            report["requested_generation"] = None
            report["resolved_generation"] = generation
            job.genome, job.report, job.state = choice.genome, report, "done"
        except _Cancelled:
            job.state = "cancelled"
        except Exception as error:  # reported on the page rather than lost with the thread
            job.error = f"{type(error).__name__}: {error}"
            job.state = "failed"
        finally:
            self.budget.leave(key)
            job.finished = time.time()

    def job(self, job_id: str | None) -> dict:
        with self._lock:
            job = self._job
        if job is None or (job_id and job.id != str(job_id)):
            return {"error": "no such job", "job": None}
        return {"job": job.to_dict()}

    def cancel(self, job_id: str | None) -> dict:
        with self._lock:
            job = self._job
        if job is None or (job_id and job.id != str(job_id)) or job.state != "running":
            return {"error": "no choice is running"}
        job.cancel.set()
        return {"ok": True}

    def export(self, view: RunView, payload: dict) -> dict:
        """Write `view`'s model: a finished choice's pick, or the champion as of a generation."""
        if view.architecture is None:
            return {"error": view.problem or "the run's network is not recorded"}
        if payload.get("job") is not None:
            with self._lock:
                job = self._job
            if (job is None or job.id != str(payload["job"]) or job.run != view.run_id
                    or job.state != "done"):
                return {"error": "that choice is not finished"}
            genome, report = job.genome, job.report
        else:
            requested = payload.get("generation")
            try:
                generation = None if requested is None else int(requested)
            except (TypeError, ValueError):
                return {"error": f"{requested!r} is not a generation"}
            resolved, genome = view.champion(generation)
            if genome is None:
                return {"error": "no champion genome archived for that generation",
                        "generation": resolved}
            report = self.preview(view, generation)
        return export_model.write_model(
            report=report,
            genome=genome,
            architecture=view.architecture,
            destination=self.destination(view),
            archive_dir=self.exports_dir / "archive",
        )

    def winding_down(self) -> list[Trainer]:
        """Take on no more runs, and return the trainers of those training here to stop."""
        with self._lock:
            self._closing = True
            return [run.trainer for run in self._live.values()]

    def close(self, timeout: float | None = 10.0) -> None:
        """Cancel a running choice and stop every run here, waiting up to `timeout` for each.

        The runs stop after their generations, together, and their jobs are left
        training, as are those of the runs still waiting. The next dashboard on
        the runs directory carries them all on.
        """
        with self._lock:
            self._closing = True
            job, live = self._job, list(self._live.values())
            locks = [self._locks.pop(name) for name in self._waiting]
            self._waiting.clear()
        if job is not None:
            job.cancel.set()
        for run in live:
            run.trainer.request_stop()
        for run in live:
            self.retire(run, timeout)
        for lock in locks:
            lock.release()
