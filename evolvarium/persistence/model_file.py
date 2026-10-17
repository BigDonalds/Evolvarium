"""The engine's side of the JSON model file.

`evolvarium.player` defines the format and runs a model with the standard
library alone. This module builds a model file from a genome, writes it, reads
it back and turns it into a genome and an architecture again.

The round trip is exact. Every weight is written as the float64 value of its
float32, which any JSON reader recovers exactly and which converts back to the
same float32.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..nn.network import Architecture, PolicyNetwork
from ..player import FORMAT, VERSION, ModelError, check, read

__all__ = ["ModelError", "build", "dumps", "read", "to_genome", "write"]

# A list of numbers, which json.dumps(indent=...) spreads one to a line.
_NUMBER_LIST = re.compile(r"\[\n\s*([-+.\deE]+(?:,\n\s*[-+.\deE]+)*)\n\s*\]")


def build(
    *,
    genome: np.ndarray,
    architecture: Architecture,
    input_names: Sequence[str],
    encoding: str,
    settings: dict,
    actions: Sequence[dict],
    game: dict,
    source: dict,
    report: dict | None = None,
    exported_at: str | None = None,
) -> dict:
    """The model file for `genome`, as a dict ready for `write`.

    `actions` is the game's action parts as `contract.Actions.describe` gives
    them.
    """
    network = PolicyNetwork(architecture).load(np.asarray(genome, dtype=np.float32))
    arrays = network.layer_arrays()
    layers = []
    for index, activation in enumerate((*architecture.hidden_activations, "linear")):
        weight = arrays[f"weight_{index}"]
        layers.append({
            "inputs": int(weight.shape[0]),
            "outputs": int(weight.shape[1]),
            "activation": activation,
            # The engine holds (inputs, outputs); the file has one row per output.
            "weights": weight.T.astype(np.float64).tolist(),
            "biases": arrays[f"bias_{index}"].astype(np.float64).tolist(),
        })

    document = {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": exported_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "game": game,
        "inputs": {"names": list(input_names), "encoding": encoding, "settings": settings},
        "actions": {"parts": [dict(part) for part in actions]},
        "network": {"kind": "dense", "layers": layers},
    }
    if report is not None:
        document["report"] = report
    check(document)
    return document


def dumps(document: dict) -> str:
    """The document as JSON, with each row of weights on one line."""
    check(document)
    text = json.dumps(document, indent=2, allow_nan=False)
    return _NUMBER_LIST.sub(
        lambda match: "[" + ", ".join(v.strip() for v in match.group(1).split(",")) + "]", text
    ) + "\n"


def write(document: dict, path: Path) -> int:
    """Write the document, replacing any file at `path` only once it is complete.

    Returns the size in bytes.
    """
    data = dumps(document).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_bytes(data)
    staging.replace(path)
    return len(data)


def to_genome(document: dict) -> tuple[np.ndarray, Architecture]:
    """The genome and architecture that run the model in the engine again."""
    check(document)
    layers = document["network"]["layers"]
    if layers[-1]["activation"] != "linear":
        raise ModelError(
            "the engine's networks end in a linear layer; this one ends in "
            f"{layers[-1]['activation']!r}"
        )
    architecture = Architecture(
        input_size=layers[0]["inputs"],
        hidden_layers=tuple(layer["outputs"] for layer in layers[:-1]),
        hidden_activations=tuple(layer["activation"] for layer in layers[:-1]),
        output_size=layers[-1]["outputs"],
    )
    parts = []
    for layer in layers:
        parts.append(np.asarray(layer["weights"], dtype=np.float32).T.ravel())
        parts.append(np.asarray(layer["biases"], dtype=np.float32))
    return np.concatenate(parts), architecture
