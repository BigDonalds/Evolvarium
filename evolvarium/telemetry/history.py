"""Generation history.

Storage is tiered so that browsing any past generation stays possible without
memory growing with the length of the run:

  summaries  one small JSON line per generation, appended to history.jsonl and
             never evicted, so the full curve of a run is always available.
  replays    per-generation gzipped frame archives on disk, under a retention
             limit. Record-setting generations are pinned and never evicted, so
             milestones stay reviewable however long the run goes on.
  memory     a bounded ring of the most recent generations for instant
             scrubbing without touching disk.

Holding every generation's frames in memory is what makes a long run degrade:
frames are by far the largest part of a generation record, and only the most
recent handful are ever read interactively.
"""

from __future__ import annotations

import base64
import gzip
import json
import threading
from collections import OrderedDict
from pathlib import Path, PureWindowsPath

from ..config import HistoryConfig
from ..events import EventBus, GenerationEvaluated
from ..evolution.individual import Individual
from ..evolution.population import GenerationReport
from ..game.engine import EpisodeResult

# Heading index per step, matching Direction: up, right, down, left.
_DELTA_INDEX = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}


def pack_replay(result: EpisodeResult, max_steps: int) -> dict:
    """Compact a recorded episode into a reconstructable path.

    Every step is kept rather than sampled. Sampling head positions makes the
    body impossible to rebuild: consecutive kept frames are no longer adjacent
    cells, so the snake renders as scattered squares instead of a connected body.

    The path is stored as one byte per step holding the direction moved, which
    is both smaller than coordinate pairs and highly compressible, since a snake
    repeats headings. Food positions and growth points are stored as sparse
    change lists because both change only when the snake eats.
    """
    frames = result.frames[:max_steps] if max_steps > 0 else result.frames
    if not frames:
        return {"start": [0, 0], "dirs": "", "food": [], "grow": [],
                "start_length": 0, "steps": 0, "truncated": False}

    start = frames[0].head
    directions = bytearray()
    previous = start
    for frame in frames[1:]:
        delta = (frame.head[0] - previous[0], frame.head[1] - previous[1])
        directions.append(_DELTA_INDEX.get(delta, 1))
        previous = frame.head

    food: list[list[int]] = []
    grow: list[int] = []
    last_food = None
    last_score = frames[0].score
    for index, frame in enumerate(frames):
        if frame.food != last_food:
            food.append([index, frame.food[0], frame.food[1]])
            last_food = frame.food
        if frame.score != last_score:
            grow.append(index)
            last_score = frame.score

    return {
        "start": [start[0], start[1]],
        "dirs": base64.b64encode(bytes(directions)).decode("ascii"),
        "food": food,
        "grow": grow,
        "start_length": frames[0].length,
        "steps": len(frames),
        "truncated": max_steps > 0 and len(result.frames) > max_steps,
    }


class GenerationRecord:
    """One generation's summary, its agents' replays, and its champion genome.

    The genome is archived per generation so a model can be exported for any
    point in the run, not only for the generations a checkpoint happened to land
    on. With the egocentric encoding a genome is a few hundred floats, which is
    negligible next to the replay frames stored alongside it.
    """

    __slots__ = ("generation", "summary", "agents", "best_genome")

    def __init__(self, generation: int, summary: dict, agents: list[dict],
                 best_genome: list[float] | None = None):
        self.generation = generation
        self.summary = summary
        self.agents = agents
        self.best_genome = best_genome

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "summary": self.summary,
            "agents": self.agents,
            "best_genome": self.best_genome,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GenerationRecord:
        return cls(
            int(data["generation"]),
            data["summary"],
            data["agents"],
            data.get("best_genome"),
        )


class HistoryStore:
    """Reader and writer for one run's generation history."""

    def __init__(self, run_dir: Path, config: HistoryConfig):
        self.run_dir = run_dir
        self.config = config
        self.replay_dir = run_dir / "replays"
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = run_dir / "history.jsonl"

        self._lock = threading.RLock()
        self._memory: OrderedDict[int, GenerationRecord] = OrderedDict()
        self._summaries: list[dict] = []
        self._pinned: set[int] = set()
        self._load_summaries()

    # ------------------------------------------------------------------ loading

    def _load_summaries(self) -> None:
        self._summaries = read_summaries(self.summary_path)
        self._pinned = {int(e["generation"]) for e in self._summaries if e.get("record")}

    def _replay_path(self, generation: int) -> Path:
        return self.replay_dir / f"gen_{generation:06d}.json.gz"

    # ------------------------------------------------------------------ writing

    def attach(self, events: EventBus) -> None:
        events.subscribe(self.record_generation, GenerationEvaluated, keeps_record=True)

    def record_generation(self, event: GenerationEvaluated) -> None:
        report = event.report

        # Replays were packed wherever the genome was scored, so this is a cheap
        # assembly step rather than a second pass over the episodes.
        agents = []
        for member in report.evaluated:
            payload = member.summary()
            payload["replay"] = member.replay
            agents.append(payload)

        limit = self.config.agents_per_generation
        agents.sort(key=lambda a: a["fitness"], reverse=True)
        if limit > 0:
            agents = agents[:limit]

        genome = event.best_genome
        self.append(report, agents, None if genome is None else [float(v) for v in genome])

    def append(self, report: GenerationReport, agents: list[dict],
               best_genome: list[float] | None = None) -> None:
        summary = report.to_dict()
        record = GenerationRecord(report.generation, summary, agents, best_genome)

        with self._lock:
            self._summaries.append(summary)
            with self.summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, separators=(",", ":")) + "\n")

            if report.record:
                self._pinned.add(report.generation)

            payload = json.dumps(record.to_dict(), separators=(",", ":")).encode("utf-8")
            with gzip.open(self._replay_path(report.generation), "wb", compresslevel=6) as handle:
                handle.write(payload)

            self._memory[report.generation] = record
            while len(self._memory) > self.config.memory_window:
                self._memory.popitem(last=False)

            self._evict_replays()

    def _evict_replays(self) -> None:
        keep = self.config.replay_retention
        if keep <= 0:
            return
        files = sorted(self.replay_dir.glob("gen_*.json.gz"))
        removable = [
            p for p in files
            if int(p.stem.split("_")[1].split(".")[0]) not in self._pinned
        ]
        excess = len(files) - keep
        for path in removable[: max(excess, 0)]:
            try:
                path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------ reading

    def summaries(self, start: int = 0, limit: int | None = None) -> list[dict]:
        with self._lock:
            window = [s for s in self._summaries if s["generation"] >= start]
        return window[-limit:] if limit else window

    def curve(self) -> dict[str, list]:
        """Compact series for the dashboard charts."""
        with self._lock:
            return {
                "generation": [s["generation"] for s in self._summaries],
                "best_score": [s["best_score"] for s in self._summaries],
                "mean_score": [s["mean_score"] for s in self._summaries],
                "max_episode_score": [s.get("max_episode_score", 0) for s in self._summaries],
                "best_fitness": [s["best_fitness"] for s in self._summaries],
                "mean_fitness": [s["mean_fitness"] for s in self._summaries],
                "diversity": [s.get("diversity", 0) for s in self._summaries],
                "scoring_fraction": [s.get("scoring_fraction", 0) for s in self._summaries],
            }

    def available(self) -> list[int]:
        """Generations whose replay can still be loaded."""
        with self._lock:
            on_disk = {
                int(p.stem.split("_")[1].split(".")[0])
                for p in self.replay_dir.glob("gen_*.json.gz")
            }
            return sorted(on_disk | set(self._memory))

    def get(self, generation: int) -> GenerationRecord | None:
        with self._lock:
            cached = self._memory.get(generation)
            if cached is not None:
                return cached
        path = self._replay_path(generation)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as handle:
                record = GenerationRecord.from_dict(json.loads(handle.read().decode("utf-8")))
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        with self._lock:
            self._memory[generation] = record
            while len(self._memory) > self.config.memory_window:
                self._memory.popitem(last=False)
        return record

    def latest(self) -> GenerationRecord | None:
        with self._lock:
            if self._memory:
                return next(reversed(self._memory.values()))
        available = self.available()
        return self.get(available[-1]) if available else None

    def summary_for(self, generation: int) -> dict | None:
        with self._lock:
            for entry in reversed(self._summaries):
                if entry["generation"] == generation:
                    return entry
        return None

    def bounds(self) -> tuple[int, int]:
        with self._lock:
            if not self._summaries:
                return (0, 0)
            return (self._summaries[0]["generation"], self._summaries[-1]["generation"])


def read_summaries(path: Path) -> list[dict]:
    """Parse a history.jsonl, skipping blank and damaged lines."""
    if not path.exists():
        return []
    summaries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                summaries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return summaries


def lineage_summaries(run_dir: Path) -> list[dict]:
    """Summaries for a run and every run it was resumed from, oldest first.

    Resuming starts a new run directory whose history.jsonl begins at the
    checkpoint's generation, so the generations before it live in the parent
    runs. Where a parent ran past the point its child resumed from, the child's
    entries win.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    current: Path | None = run_dir
    while current is not None and current.name not in seen:
        seen.add(current.name)
        own = read_summaries(current / "history.jsonl")
        if merged:
            own = [s for s in own if s["generation"] < merged[0]["generation"]]
        merged = own + merged
        current = _parent_run(current)
    return merged


def _parent_run(run_dir: Path) -> Path | None:
    """The run a resumed run started from, looked up next to it by name.

    `resumed_from` is stored relative to wherever training was launched, and
    may use Windows separators, so only the run's directory name is trusted.
    """
    try:
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source = manifest.get("resumed_from")
    if not source:
        return None
    # runs/<run>/checkpoints/<file>.npz
    name = PureWindowsPath(source).parent.parent.name
    parent = run_dir.parent / name
    return parent if name and parent.is_dir() else None


def agent_payload(individual: Individual, result: EpisodeResult, max_steps: int) -> dict:
    """Serialisable per-agent record for a generation archive."""
    payload = individual.summary()
    payload["replay"] = pack_replay(result, max_steps)
    return payload
