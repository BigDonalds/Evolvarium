"""Standalone model export.

An export is a JSON model file (see `evolvarium.player` for the format), which
runs with the standard library alone, and a .pt copy of it when torch is
installed. Training never produces either. Checkpoints are .npz and torch is not
imported anywhere in the training path; it is pulled in only when an export is
actually written, so a plain training run has no framework dependency.

Export is deliberately two-phase. `build_report` gathers the statistics,
achievements, timings and size for a chosen generation without touching the
filesystem, so the dashboard can show exactly what is about to be written.
`write_model` then produces the files.
"""

from __future__ import annotations

import importlib.util
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import games
from ..contract import Game
from ..nn.network import Architecture, PolicyNetwork
from . import model_file

USAGE = (
    "Load with torch.load(path). parameters holds weight_i and bias_i for each layer, "
    "weight_i shaped (outputs, inputs) as torch.nn.Linear keeps it. Everything else is "
    "the JSON model exported with this file; evolvarium/player.py describes the format."
)

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


def summarise_training(summaries: list[dict], generation: int,
                       direction: str = "higher") -> dict:
    """Aggregate run statistics up to and including `generation`."""
    window = [s for s in summaries if s["generation"] <= generation]
    if not window:
        return {
            "generations": 0, "episodes": 0, "compute_seconds": 0.0,
            "best_score": None, "best_fitness": 0.0, "best_generation": None,
            "mean_score_final": 0.0, "scoring_fraction_final": 0.0,
            "mean_steps_final": 0.0, "records_set": 0, "ends": {},
        }

    scored = [e for e in window if e.get("max_episode_score") is not None] or window
    pick = max if direction == "higher" else min
    best_entry = pick(scored, key=lambda e: e.get("max_episode_score", 0))
    ends: dict[str, int] = {}
    for entry in window:
        # Summaries written before games named their own endings called them deaths.
        for cause, count in entry.get("ends", entry.get("deaths", {})).items():
            ends[cause] = ends.get(cause, 0) + count

    return {
        "generations": len(window),
        "episodes": int(sum(e.get("episodes", 0) for e in window)),
        "compute_seconds": round(sum(e.get("elapsed", 0.0) for e in window), 2),
        "best_score": best_entry.get("max_episode_score", 0),
        "best_fitness": float(max(e.get("best_fitness", 0.0) for e in window)),
        "best_generation": int(best_entry["generation"]),
        "mean_score_final": float(window[-1].get("mean_score", 0.0)),
        "scoring_fraction_final": float(window[-1].get("scoring_fraction", 0.0)),
        "mean_steps_final": float(window[-1].get("mean_steps", 0.0)),
        "records_set": sum(1 for e in window if e.get("record")),
        "ends": ends,
    }


def estimate_size(architecture: Architecture) -> int:
    """Approximate size of the JSON model when there is no genome to measure."""
    return architecture.genome_size * 24 + 3072


def build_report(
    *,
    generation: int,
    genome: np.ndarray | None,
    summaries: list[dict],
    game: Game,
    architecture: Architecture,
    run_id: str,
    destination: Path,
) -> dict:
    """Everything the dashboard shows before an export is confirmed.

    `game` is built with the settings the genome was trained under; the model
    file records its rules. Touches no files.
    """
    achievements = game.milestones([s for s in summaries if s["generation"] <= generation])
    training = summarise_training(summaries, generation, game.direction)
    unlocked = [a for a in achievements if a.reached]
    inputs = game.model_inputs()

    report = {
        "ready": genome is not None,
        "generation": generation,
        "run_id": run_id,
        "torch_available": torch_available(),
        "genome_available": genome is not None,
        "destination": str(destination),
        "filename": destination.name,
        "estimated_size_bytes": estimate_size(architecture),
        "architecture": architecture.to_dict(),
        "parameter_count": architecture.genome_size,
        "inputs": {
            "encoding": inputs["encoding"],
            "size": game.observation.size,
            "names": list(game.observation.names),
            "settings": inputs["settings"],
        },
        "game": {"name": games.reference(game), **game.model_rules()},
        "actions": game.actions.describe(),
        "training": training,
        "achievements": [a.to_dict() for a in achievements],
        "achievements_unlocked": len(unlocked),
        "achievements_total": len(achievements),
    }
    if report["ready"]:
        report["estimated_size_bytes"] = len(
            model_file.dumps(_document(report, genome, architecture)).encode("utf-8")
        )
    return report


def _document(report: dict, genome: np.ndarray, architecture: Architecture) -> dict:
    inputs = report["inputs"]
    source = {"run": report["run_id"], "generation": report["generation"]}
    # How the genome was chosen, when it was not simply the stored champion.
    if report.get("selection"):
        source["selection"] = report["selection"]
    return model_file.build(
        genome=genome,
        architecture=architecture,
        input_names=inputs["names"],
        encoding=inputs["encoding"],
        settings=inputs["settings"],
        actions=report["actions"],
        game=report["game"],
        source=source,
        report={"training": report["training"], "achievements": report["achievements"]},
    )


def write_model(
    *,
    report: dict,
    genome: np.ndarray,
    architecture: Architecture,
    destination: Path,
    archive_dir: Path | None = None,
) -> dict:
    """Write the JSON model to `destination`, and a .pt copy beside it when torch is installed."""
    if destination.suffix != ".json":
        raise ValueError(f"the model file is JSON; got destination {destination}")
    document = _document(report, genome, architecture)
    size = model_file.write(document, destination)
    written = [destination]

    torch_copy = destination.with_suffix(".pt")
    if torch_available() and _write_torch(document, genome, architecture, torch_copy):
        written.append(torch_copy)
    else:
        torch_copy = None

    archived = []
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in written:
            copy = archive_dir / f"{path.stem}_gen{report['generation']:06d}_{stamp}{path.suffix}"
            copy.write_bytes(path.read_bytes())
            archived.append(str(copy))

    return {
        "written": True,
        "path": str(destination),
        "pt": None if torch_copy is None else str(torch_copy),
        "archived": archived,
        "size_bytes": size,
        "generation": report["generation"],
        "parameter_count": report["parameter_count"],
        "achievements_unlocked": report["achievements_unlocked"],
    }


def _write_torch(document: dict, genome: np.ndarray, architecture: Architecture,
                 path: Path) -> bool:
    """The .pt copy: the JSON model plus the weights as tensors."""
    try:
        import torch
    except ImportError:
        return False

    genome = np.asarray(genome, dtype=np.float32)
    parameters = {}
    for name, array in PolicyNetwork(architecture).load(genome).layer_arrays().items():
        if name.startswith("weight_"):
            array = np.ascontiguousarray(array.T)
        parameters[name] = torch.from_numpy(array)

    payload = {
        **document,
        "usage": USAGE,
        "genome": torch.from_numpy(genome.copy()),
        "parameters": parameters,
    }
    staging = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, staging)
    staging.replace(path)
    return True
