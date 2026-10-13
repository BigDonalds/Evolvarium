"""Checkpoint save and load.

Checkpoints are numpy .npz archives. Nothing here imports torch: the training
loop should not carry a deep-learning framework just to persist a few float32
arrays, and .npz keeps checkpoints readable by any numpy install.

Exporting a standalone .pt model is a separate, explicit action that lives in
`export_torch` and is triggered from the dashboard.

Every checkpoint records the feature spec and architecture it was trained under.
Loading refuses a mismatch rather than reshaping the genome to fit, because a
resized genome is not the model that was saved.
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

from ..config import Config
from ..evolution.individual import Individual
from ..evolution.population import Population
from ..nn.network import Architecture

FORMAT_VERSION = 2

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
    }

    best = population.best_ever_genome
    arrays = {
        "genomes": genomes,
        "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    if best is not None:
        arrays["best_genome"] = best.astype(np.float32)

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


def load(path: Path, config: Config, seed: int | None = None) -> Population:
    """Rebuild a population from a checkpoint, or raise CheckpointMismatch."""
    with _FILE_LOCK, np.load(path, allow_pickle=False) as archive:
        meta = json.loads(bytes(archive["meta"]).decode("utf-8"))
        genomes = np.array(archive["genomes"], dtype=np.float32)
        best = (
            np.array(archive["best_genome"], dtype=np.float32)
            if "best_genome" in archive else None
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

    origins = meta.get("origins", [])
    ages = meta.get("ages", [])
    population.members = [
        Individual(
            genome=genomes[i].copy(),
            generation=int(meta.get("generation", 0)),
            age=int(ages[i]) if i < len(ages) else 0,
            origin=str(origins[i]) if i < len(origins) else "restored",
        )
        for i in range(len(genomes))
    ]

    # Top up or trim if the configured population size has changed since saving.
    target = config.evolution.population_size
    while len(population.members) < target:
        from ..nn.network import random_genome

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
