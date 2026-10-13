"""Standalone .pt model export.

Training never produces a .pt file. Checkpoints are .npz and torch is not
imported anywhere in the training path; it is pulled in only when an export is
actually requested, so a plain training run has no framework dependency.

Export is deliberately two-phase. `build_report` gathers the statistics,
achievements, timings and size estimate for a chosen generation without touching
the filesystem, so the dashboard can show exactly what is about to be written.
`write_model` then produces the file.

The exported archive is self-contained: it carries the weights as named tensors,
the architecture, the input encoding and its column names, and the action space.
Running the model afterwards needs numpy or torch, not this project.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..features import FeatureSpec
from ..nn.network import Architecture, PolicyNetwork

FORMAT = "evolvarium-model"
FORMAT_VERSION = 1

USAGE = (
    "Load with torch.load(path, weights_only=False). Feed the 11 inputs named in "
    "feature_spec.inputs, in order, through the weight_i/bias_i pairs applying the "
    "listed hidden activations; argmax of the 3 outputs selects turn "
    "left / straight / right relative to the snake's current heading."
)


class TorchUnavailable(RuntimeError):
    """Raised when a .pt export is requested without torch installed."""


_availability: bool | None = None
_preload_started = False


def torch_available() -> bool:
    """Whether torch could be imported, without importing it.

    Importing torch costs a couple of seconds on an idle machine but far longer
    while the trainer's tight Python loop is holding the GIL, so an availability
    check must never trigger the import. Locating the module is enough and is
    effectively free.
    """
    global _availability
    if _availability is None:
        _availability = importlib.util.find_spec("torch") is not None
    return _availability


def preload() -> None:
    """Import torch in the background so a later export does not wait on it.

    Started when the dashboard comes up. The import competes with training for
    the GIL and may take a while, but nothing blocks on it: an export requested
    before it finishes simply performs the import itself.
    """
    global _preload_started
    if _preload_started or not torch_available():
        return
    _preload_started = True

    def load() -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            global _availability
            _availability = False

    threading.Thread(target=load, name="torch-preload", daemon=True).start()


@dataclass
class Achievement:
    key: str
    label: str
    detail: str
    generation: int | None
    unlocked: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "detail": self.detail,
            "generation": self.generation,
            "unlocked": self.unlocked,
        }


_SCORE_TIERS = (
    (1, "First bite", "An agent ate at least once"),
    (5, "Five course", "Reached a score of 5"),
    (10, "Double digits", "Reached a score of 10"),
    (20, "Twenty", "Reached a score of 20"),
    (35, "Coiled", "Reached a score of 35"),
    (50, "Half century", "Reached a score of 50"),
)


def _first_generation_where(summaries: list[dict], predicate) -> int | None:
    for entry in summaries:
        try:
            if predicate(entry):
                return int(entry["generation"])
        except (KeyError, TypeError):
            continue
    return None


def compute_achievements(summaries: list[dict]) -> list[Achievement]:
    """Milestones reached at or before the last summary supplied."""
    achievements: list[Achievement] = []

    for threshold, label, detail in _SCORE_TIERS:
        gen = _first_generation_where(
            summaries, lambda e, t=threshold: e.get("max_episode_score", 0) >= t
        )
        achievements.append(
            Achievement(f"score_{threshold}", label, detail, gen, gen is not None)
        )

    gen = _first_generation_where(summaries, lambda e: e.get("scoring_fraction", 0) >= 0.5)
    achievements.append(
        Achievement("majority_feeding", "Majority feeding",
                    "Over half the population ate in one generation", gen, gen is not None)
    )

    gen = _first_generation_where(summaries, lambda e: e.get("scoring_fraction", 0) >= 0.9)
    achievements.append(
        Achievement("population_feeding", "Whole population feeding",
                    "Nine in ten agents ate in one generation", gen, gen is not None)
    )

    gen = _first_generation_where(summaries, lambda e: e.get("mean_steps", 0) >= 100)
    achievements.append(
        Achievement("endurance", "Endurance",
                    "Mean survival passed 100 steps", gen, gen is not None)
    )

    gen = _first_generation_where(
        summaries,
        lambda e: e.get("deaths", {}).get("wall", 1) == 0 and sum(e.get("deaths", {}).values()) > 0,
    )
    achievements.append(
        Achievement("wall_wise", "Wall wise",
                    "A generation passed with no wall deaths", gen, gen is not None)
    )

    gen = _first_generation_where(summaries, lambda e: e.get("mean_score", 0) >= 1.0)
    achievements.append(
        Achievement("consistent", "Consistent",
                    "Mean score across the population reached 1.0", gen, gen is not None)
    )

    return achievements


def summarise_training(summaries: list[dict], generation: int) -> dict:
    """Aggregate run statistics up to and including `generation`."""
    window = [s for s in summaries if s["generation"] <= generation]
    if not window:
        return {
            "generations": 0, "episodes": 0, "compute_seconds": 0.0,
            "best_score": 0, "best_fitness": 0.0, "best_generation": None,
            "mean_score_final": 0.0, "scoring_fraction_final": 0.0,
            "records_set": 0, "deaths": {},
        }

    best_entry = max(window, key=lambda e: e.get("max_episode_score", 0))
    deaths: dict[str, int] = {}
    for entry in window:
        for cause, count in entry.get("deaths", {}).items():
            deaths[cause] = deaths.get(cause, 0) + count

    return {
        "generations": len(window),
        "episodes": int(sum(e.get("episodes", 0) for e in window)),
        "compute_seconds": round(sum(e.get("elapsed", 0.0) for e in window), 2),
        "best_score": int(best_entry.get("max_episode_score", 0)),
        "best_fitness": float(max(e.get("best_fitness", 0.0) for e in window)),
        "best_generation": int(best_entry["generation"]),
        "mean_score_final": float(window[-1].get("mean_score", 0.0)),
        "scoring_fraction_final": float(window[-1].get("scoring_fraction", 0.0)),
        "mean_steps_final": float(window[-1].get("mean_steps", 0.0)),
        "records_set": sum(1 for e in window if e.get("record")),
        "deaths": deaths,
    }


def estimate_size(architecture: Architecture) -> int:
    """Approximate exported file size: float32 parameters plus metadata."""
    return architecture.genome_size * 4 * 2 + 6144


def build_report(
    *,
    generation: int,
    genome: np.ndarray | None,
    summaries: list[dict],
    config: Config,
    architecture: Architecture,
    feature_spec: FeatureSpec,
    run_id: str,
    destination: Path,
) -> dict:
    """Everything the dashboard shows before an export is confirmed.

    Touches no files.
    """
    achievements = compute_achievements([s for s in summaries if s["generation"] <= generation])
    training = summarise_training(summaries, generation)
    unlocked = [a for a in achievements if a.unlocked]

    return {
        "ready": genome is not None and torch_available(),
        "generation": generation,
        "run_id": run_id,
        "torch_available": torch_available(),
        "genome_available": genome is not None,
        "destination": str(destination),
        "filename": destination.name,
        "estimated_size_bytes": estimate_size(architecture),
        "architecture": architecture.to_dict(),
        "parameter_count": architecture.genome_size,
        "feature_spec": {
            "name": feature_spec.name,
            "size": feature_spec.size,
            "inputs": list(feature_spec.names),
        },
        "board": {"width": config.board.width, "height": config.board.height},
        "action_space": ["left", "straight", "right"],
        "training": training,
        "achievements": [a.to_dict() for a in achievements],
        "achievements_unlocked": len(unlocked),
        "achievements_total": len(achievements),
    }


def write_model(
    *,
    report: dict,
    genome: np.ndarray,
    architecture: Architecture,
    destination: Path,
    archive_dir: Path | None = None,
) -> dict:
    """Write the .pt archive plus a readable JSON sidecar."""
    try:
        import torch
    except ImportError as exc:
        raise TorchUnavailable(
            "exporting a .pt model requires torch; install it with "
            "'pip install evolvarium[export]' or 'pip install torch'"
        ) from exc

    network = PolicyNetwork(architecture).load(genome)
    parameters = {
        name: torch.from_numpy(array) for name, array in network.layer_arrays().items()
    }

    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": report["run_id"],
        "generation": report["generation"],
        "architecture": report["architecture"],
        "feature_spec": report["feature_spec"],
        "action_space": report["action_space"],
        "board": report["board"],
        "genome": torch.from_numpy(np.asarray(genome, dtype=np.float32)),
        "parameters": parameters,
        "training": report["training"],
        "achievements": report["achievements"],
        "usage": USAGE,
    }
    # How the genome was chosen, when it was not simply the stored champion.
    if report.get("selection"):
        payload["selection"] = report["selection"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, staging)
    staging.replace(destination)

    sidecar = destination.with_suffix(".json")
    readable = {k: v for k, v in payload.items() if k not in ("genome", "parameters")}
    readable["parameter_count"] = report["parameter_count"]
    sidecar.write_text(json.dumps(readable, indent=2), encoding="utf-8")

    archived: str | None = None
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        copy = archive_dir / f"{destination.stem}_gen{report['generation']:06d}_{stamp}.pt"
        copy.write_bytes(destination.read_bytes())
        archived = str(copy)

    size = destination.stat().st_size
    return {
        "written": True,
        "path": str(destination),
        "sidecar": str(sidecar),
        "archived": archived,
        "size_bytes": size,
        "generation": report["generation"],
        "parameter_count": report["parameter_count"],
        "achievements_unlocked": report["achievements_unlocked"],
    }
