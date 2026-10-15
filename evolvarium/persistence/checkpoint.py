"""Checkpoint save and load.

Checkpoints are numpy .npz archives. Nothing here imports torch: the training
loop should not carry a deep-learning framework just to persist a few float32
arrays, and .npz keeps checkpoints readable by any numpy install.

Exporting a standalone model is a separate, explicit action that lives in
`export_model` and is triggered from the dashboard or the command line.

Every checkpoint records the feature spec and architecture it was trained under.
Loading refuses a mismatch rather than reshaping the genome to fit, because a
resized genome is not the model that was saved.

From format 3 a checkpoint also holds everything the next generation reads that
is not a setting: the random generator's state, the species and the members'
ids and parents. A run resumed from it continues exactly as the uninterrupted
run would have. Format 2 checkpoints still load, but the resumed run draws new
random numbers from the seed it is given and goes its own way.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import CheckpointConfig, Config
from ..events import (
    CheckpointSaved,
    Event,
    EventBus,
    GenerationEvaluated,
    RecordSet,
    RunFinished,
)
from ..evolution import speciation
from ..evolution.individual import Individual, id_number, reserve_ids
from ..evolution.population import Population
from ..nn.network import Architecture, random_genome

FORMAT_VERSION = 3

# The dashboard lists and inspects checkpoints on its own thread while the
# trainer writes them. Windows refuses to replace a file that still has an open
# handle, so readers and the writer take turns rather than racing.
_FILE_LOCK = threading.RLock()


class CheckpointMismatch(RuntimeError):
    """Raised when a checkpoint cannot be loaded into the current setup."""


def _replace_with_retry(source: Path, target: Path, attempts: int = 5) -> None:
    """Move `source` onto `target`, tolerating a transient Windows lock.

    Indexers and virus scanners open newly written files briefly, which surfaces
    as PermissionError on an otherwise valid replace.
    """
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


@dataclass
class CheckpointInfo:
    path: Path
    generation: int
    best_score: int
    best_fitness: float
    population_size: int
    feature_spec: str
    saved_at: str
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "name": self.path.name,
            "generation": self.generation,
            "best_score": self.best_score,
            "best_fitness": round(self.best_fitness, 2),
            "population_size": self.population_size,
            "feature_spec": self.feature_spec,
            "saved_at": self.saved_at,
            "size_bytes": self.size_bytes,
        }


def save(population: Population, path: Path) -> CheckpointInfo:
    """Write the full population state to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)

    genomes = np.stack([m.genome for m in population.members]).astype(np.float32)
    meta = {
        "format_version": FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation": population.generation,
        "total_episodes": population.total_episodes,
        "best_ever_score": population.best_ever_score,
        "best_ever_fitness": population.best_ever_fitness,
        "best_ever_generation": population.best_ever_generation,
        "stagnation": population.stagnation,
        "feature_spec": population.feature_spec.name,
        "architecture": population.architecture.to_dict(),
        # Recorded so a finished run can say what produced it. Comparing two runs
        # by launch order alone is a good way to attribute a result to the wrong
        # configuration.
        "fitness": dataclasses.asdict(population.config.fitness),
        "evolution": dataclasses.asdict(population.config.evolution),
        "origins": [m.origin for m in population.members],
        "ages": [m.age for m in population.members],
        "ids": [m.id for m in population.members],
        "parents": [list(m.parents) for m in population.members],
        # PCG64 state is two 128-bit integers. JSON keeps integers exact at any
        # size, so the state survives the round trip unchanged.
        "rng": population.rng.bit_generator.state,
        "compatibility_threshold": population.compatibility_threshold,
        "next_species_id": population._next_species_id,
        "species": [
            {"id": s.id, "best_fitness": s.best_fitness, "stagnation": s.stagnation}
            for s in population.species
        ],
    }

    best = population.best_ever_genome
    arrays = {
        "genomes": genomes,
        "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    if best is not None:
        arrays["best_genome"] = best.astype(np.float32)
    if population.species:
        arrays["species_representatives"] = np.stack(
            [s.representative for s in population.species]
        ).astype(np.float32)

    # Write beside the target and move into place so an interrupted save cannot
    # leave a half-written checkpoint where a valid one used to be.
    staging = path.with_name(path.name + ".partial")
    with _FILE_LOCK:
        # savez_compressed appends .npz when handed a path, so pass a file
        # object to keep the staging name exactly as written.
        with staging.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        _replace_with_retry(staging, path)

    return CheckpointInfo(
        path=path,
        generation=population.generation,
        best_score=population.best_ever_score,
        best_fitness=population.best_ever_fitness,
        population_size=len(population.members),
        feature_spec=population.feature_spec.name,
        saved_at=meta["saved_at"],
        size_bytes=path.stat().st_size,
    )


def read_meta(path: Path) -> dict:
    with _FILE_LOCK, np.load(path, allow_pickle=False) as archive:
        return json.loads(bytes(archive["meta"]).decode("utf-8"))


def inspect(path: Path) -> CheckpointInfo:
    meta = read_meta(path)
    return CheckpointInfo(
        path=path,
        generation=int(meta.get("generation", 0)),
        best_score=int(meta.get("best_ever_score", 0)),
        best_fitness=float(meta.get("best_ever_fitness", 0.0)),
        population_size=len(meta.get("origins", [])),
        feature_spec=str(meta.get("feature_spec", "unknown")),
        saved_at=str(meta.get("saved_at", "")),
        size_bytes=path.stat().st_size,
    )


def restores_exactly(path: Path) -> bool:
    """Whether resuming from `path` continues the run it was saved from."""
    return "rng" in read_meta(path)


def load(path: Path, config: Config, seed: int | None = None) -> Population:
    """Rebuild a population from a checkpoint, or raise CheckpointMismatch.

    `seed` is used only when the checkpoint carries no random state. Otherwise
    the saved state wins, since reseeding is exactly what makes a resumed run
    drift from the original.
    """
    with _FILE_LOCK, np.load(path, allow_pickle=False) as archive:
        meta = json.loads(bytes(archive["meta"]).decode("utf-8"))
        genomes = np.array(archive["genomes"], dtype=np.float32)
        best = (
            np.array(archive["best_genome"], dtype=np.float32)
            if "best_genome" in archive else None
        )
        representatives = (
            np.array(archive["species_representatives"], dtype=np.float32)
            if "species_representatives" in archive else None
        )

    spec = meta.get("feature_spec")
    if spec != config.features.spec:
        raise CheckpointMismatch(
            f"checkpoint was trained with feature spec {spec!r}, "
            f"current configuration uses {config.features.spec!r}"
        )

    saved_arch = Architecture.from_dict(meta["architecture"])
    population = Population(config, seed=seed)
    if saved_arch != population.architecture:
        raise CheckpointMismatch(
            f"checkpoint architecture {saved_arch.to_dict()} does not match "
            f"current {population.architecture.to_dict()}"
        )

    if "rng" in meta:
        population.rng.bit_generator.state = meta["rng"]

    origins = meta.get("origins", [])
    ages = meta.get("ages", [])
    ids = meta.get("ids", [])
    parents = meta.get("parents", [])
    if ids:
        reserve_ids(max(id_number(i) for i in ids))
    population.members = []
    for i in range(len(genomes)):
        member = Individual(
            genome=genomes[i].copy(),
            generation=int(meta.get("generation", 0)),
            age=int(ages[i]) if i < len(ages) else 0,
            origin=str(origins[i]) if i < len(origins) else "restored",
            parents=tuple(parents[i]) if i < len(parents) else (),
        )
        if i < len(ids):
            member.id = str(ids[i])
        population.members.append(member)

    population.compatibility_threshold = float(
        meta.get("compatibility_threshold", population.compatibility_threshold)
    )
    population._next_species_id = int(meta.get("next_species_id", 0))
    population.species = []
    for i, saved in enumerate(meta.get("species", [])):
        group = speciation.Species(int(saved["id"]), representatives[i])
        group.best_fitness = float(saved["best_fitness"])
        group.stagnation = int(saved["stagnation"])
        population.species.append(group)

    # Top up or trim if the configured population size has changed since saving.
    target = config.evolution.population_size
    while len(population.members) < target:
        population.members.append(
            Individual(
                genome=random_genome(
                    population.architecture, population.rng,
                    config.network.weight_init_scale, config.network.output_init_scale,
                ),
                generation=int(meta.get("generation", 0)),
                origin="immigrant",
            )
        )
    population.members = population.members[:target]

    population.generation = int(meta.get("generation", 0))
    population.total_episodes = int(meta.get("total_episodes", 0))
    population.best_ever_score = int(meta.get("best_ever_score", 0))
    population.best_ever_fitness = float(meta.get("best_ever_fitness", 0.0))
    population.best_ever_generation = int(meta.get("best_ever_generation", 0))
    population.stagnation = int(meta.get("stagnation", 0))
    population.best_ever_genome = best
    return population


def list_checkpoints(directory: Path) -> list[CheckpointInfo]:
    if not directory.is_dir():
        return []
    found: list[CheckpointInfo] = []
    for path in sorted(directory.glob("*.npz")):
        try:
            found.append(inspect(path))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            # A checkpoint being rewritten right now is skipped, not fatal.
            continue
    return sorted(found, key=lambda c: c.generation)


def prune(directory: Path, keep_last: int) -> int:
    """Drop the oldest periodic checkpoints, leaving `latest.npz` alone."""
    if keep_last <= 0:
        return 0
    periodic = sorted(
        (p for p in directory.glob("gen_*.npz")),
        key=lambda p: p.stat().st_mtime,
    )
    removed = 0
    with _FILE_LOCK:
        for path in periodic[:-keep_last] if len(periodic) > keep_last else []:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


class CheckpointWriter:
    """Saves checkpoints as the trainer announces generations.

    A numbered checkpoint every `save_every` generations, pruned to the newest
    `keep_last`, and `latest.npz` on each record and once more when the run ends.
    All of it happens on the trainer thread between generations, so a checkpoint
    never catches a population part way through one.
    """

    def __init__(self, population: Population, directory: Path, config: CheckpointConfig):
        self.population = population
        self.directory = directory
        self.config = config
        self._events: EventBus | None = None
        self._latest_generation: int | None = None

    def attach(self, events: EventBus) -> None:
        self._events = events
        events.subscribe(
            self.handle, GenerationEvaluated, RecordSet, RunFinished, keeps_record=True
        )

    def handle(self, event: Event) -> None:
        match event:
            case GenerationEvaluated():
                every = self.config.save_every
                generation = self.population.generation
                if every > 0 and generation % every == 0:
                    self._save(f"gen_{generation:06d}.npz", "periodic")
                    prune(self.directory, self.config.keep_last)
            case RecordSet():
                self.save_latest("record")
            case RunFinished(reason="limit" | "stopped"):
                # A run that set a record in its last generation already has it.
                if self._latest_generation != self.population.generation:
                    self.save_latest("final")

    def save_latest(self, trigger: str = "final") -> CheckpointInfo:
        info = self._save("latest.npz", trigger)
        self._latest_generation = info.generation
        return info

    def _save(self, name: str, trigger: str) -> CheckpointInfo:
        info = save(self.population, self.directory / name)
        if self._events is not None:
            self._events.publish(CheckpointSaved(str(info.path), info.generation, trigger))
        return info
