"""Run an exported Evolvarium model with the Python standard library alone.

This file imports nothing outside the standard library and nothing from the rest
of the package, so it can be copied next to a game that never installs
Evolvarium:

    from player import Player

    player = Player.load("model.json")
    move = player.action(inputs)

`inputs` is a sequence of numbers in the order of `player.input_names`, and the
result is one of `player.action_names`. From a shell,
`python player.py model.json` describes a model and
`python player.py model.json 0.0 1.0 ...` prints the action it takes.

The model file, version 1
-------------------------
A JSON object. Keys other than these, such as the training report the engine
adds, are ignored here.

format    "evolvarium-model"
version   1
inputs    {"names": [...]}, one name per input, in the order the network reads
          them. The engine also records how it computed them, under "encoding"
          and "settings".
actions   {"names": [...], "choice": "argmax"}, one name per output. The action
          taken is the one with the highest output, the earliest on a tie.
network   {"kind": "dense", "layers": [...]}. Each layer is
          {"inputs": n, "outputs": m, "activation": name,
           "weights": m rows of n numbers, "biases": m numbers}
          and gives, for each output k,
          activation(biases[k] + weights[k][0] * x[0] + ... + weights[k][n-1] * x[n-1]).
          The first layer reads the inputs and each later layer the one before
          it. The last layer's outputs are the actions'.

The activations are relu, leaky_relu (slope 0.01 below zero), tanh, sigmoid
(its argument clipped to [-60, 60]) and linear. A file of any other version is
refused rather than guessed at.

The engine computes in float32 and writes each weight as the exact value of its
float32. The player starts from the same numbers and computes in float64, so
the two can differ only where two outputs come within rounding of each other.
"""

from __future__ import annotations

import json
import math
import sys

FORMAT = "evolvarium-model"
VERSION = 1


class ModelError(ValueError):
    """The file is not a model this player can run."""


def _relu(z: float) -> float:
    return z if z > 0.0 else 0.0


def _leaky_relu(z: float) -> float:
    return z if z > 0.0 else 0.01 * z


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(z, -60.0), 60.0)))


def _linear(z: float) -> float:
    return z


ACTIVATIONS = {
    "relu": _relu,
    "leaky_relu": _leaky_relu,
    "tanh": math.tanh,
    "sigmoid": _sigmoid,
    "linear": _linear,
}


def check(document: object) -> None:
    """Raise ModelError naming the first thing wrong with `document`."""
    if not isinstance(document, dict):
        raise ModelError("a model file holds a JSON object")
    if document.get("format") != FORMAT:
        raise ModelError(f"not an Evolvarium model: format is {document.get('format')!r}")
    if document.get("version") != VERSION:
        raise ModelError(
            f"this player reads version {VERSION}; the file is version "
            f"{document.get('version')!r}"
        )

    input_names = _names(document, "inputs")
    action_names = _names(document, "actions")
    choice = document["actions"].get("choice")
    if choice != "argmax":
        raise ModelError(f"actions are chosen by {choice!r}; this player takes the highest output")

    network = document.get("network")
    kind = network.get("kind") if isinstance(network, dict) else None
    if kind != "dense":
        raise ModelError(f"network kind is {kind!r}; this player runs 'dense' networks")
    layers = network.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ModelError("network.layers must be a list with at least one layer")

    width = len(input_names)
    for index, layer in enumerate(layers):
        where = f"layer {index}"
        if not isinstance(layer, dict):
            raise ModelError(f"{where} is not an object")
        inputs, outputs = layer.get("inputs"), layer.get("outputs")
        if not (_count(inputs) and _count(outputs)):
            raise ModelError(f"{where} needs whole numbers of inputs and outputs above zero")
        if inputs != width:
            given = "the model names" if index == 0 else f"layer {index - 1} gives"
            raise ModelError(f"{where} takes {inputs} inputs, but {given} {width}")
        activation = layer.get("activation")
        if activation not in ACTIVATIONS:
            raise ModelError(
                f"{where} has unknown activation {activation!r}; known: {', '.join(ACTIVATIONS)}"
            )
        weights = layer.get("weights")
        if not isinstance(weights, list) or len(weights) != outputs:
            raise ModelError(f"{where} needs {outputs} rows of weights, one per output")
        for row, values in enumerate(weights):
            if not isinstance(values, list) or len(values) != inputs:
                raise ModelError(f"{where}, weight row {row} needs {inputs} numbers, one per input")
            _numbers(values, f"{where}, weight row {row}")
        biases = layer.get("biases")
        if not isinstance(biases, list) or len(biases) != outputs:
            raise ModelError(f"{where} needs {outputs} biases, one per output")
        _numbers(biases, f"{where} biases")
        width = outputs

    if width != len(action_names):
        raise ModelError(
            f"the last layer gives {width} outputs, but there are {len(action_names)} actions"
        )


def _names(document: dict, key: str) -> list:
    section = document.get(key)
    names = section.get("names") if isinstance(section, dict) else None
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ModelError(f"{key}.names must be a list of at least one string")
    return names


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _numbers(values: list, where: str) -> None:
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ModelError(f"{where} holds {value!r}, which is not a finite number")


def read(path) -> dict:
    """Load a model file and check it."""
    with open(path, encoding="utf-8") as file:
        try:
            document = json.load(file)
        except json.JSONDecodeError as error:
            raise ModelError(f"{path} is not valid JSON: {error}") from None
    check(document)
    return document


class Player:
    """A loaded model. It keeps no state between calls."""

    def __init__(self, document: dict):
        check(document)
        self.document = document
        self.input_names = tuple(document["inputs"]["names"])
        self.action_names = tuple(document["actions"]["names"])
        self._layers = [
            (
                [[float(w) for w in row] for row in layer["weights"]],
                [float(b) for b in layer["biases"]],
                ACTIVATIONS[layer["activation"]],
            )
            for layer in document["network"]["layers"]
        ]

    @classmethod
    def load(cls, path) -> Player:
        return cls(read(path))

    @property
    def parameter_count(self) -> int:
        return sum(len(biases) * (len(weights[0]) + 1) for weights, biases, _ in self._layers)

    def outputs(self, inputs) -> list[float]:
        """The last layer's values for one set of inputs."""
        values = [float(x) for x in inputs]
        if len(values) != len(self.input_names):
            raise ValueError(f"the model takes {len(self.input_names)} inputs, got {len(values)}")
        for weights, biases, activation in self._layers:
            layer = []
            for row, bias in zip(weights, biases, strict=True):
                total = bias
                for weight, value in zip(row, values, strict=True):
                    total += weight * value
                layer.append(activation(total))
            values = layer
        return values

    def choose(self, inputs) -> int:
        """Index of the action taken: the highest output, the earliest on a tie."""
        outputs = self.outputs(inputs)
        return max(range(len(outputs)), key=outputs.__getitem__)

    def action(self, inputs) -> str:
        return self.action_names[self.choose(inputs)]

    def describe(self) -> str:
        layers = self.document["network"]["layers"]
        shape = " -> ".join(
            [str(layers[0]["inputs"])]
            + [f"{layer['outputs']} {layer['activation']}" for layer in layers]
        )
        return "\n".join([
            f"{len(self.input_names)} inputs: {', '.join(self.input_names)}",
            f"{len(self.action_names)} actions: {', '.join(self.action_names)}",
            f"network: {shape}, {self.parameter_count} parameters",
        ])


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python player.py MODEL.json [INPUT ...]")
        return 0 if argv else 2
    try:
        player = Player.load(argv[0])
    except (OSError, ModelError) as error:
        print(error, file=sys.stderr)
        return 1
    if len(argv) == 1:
        print(player.describe())
        return 0
    try:
        print(player.action(float(value) for value in argv[1:]))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
