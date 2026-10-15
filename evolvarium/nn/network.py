"""Feed-forward policy network.

Weights are evolved, never trained by gradient descent, so there is no backward
pass, no optimiser state and no dropout or normalisation layers.

Inputs are consumed exactly as the game's observation encodes them. Rescaling the
input vector against its own mean and standard deviation would couple every
feature to every other one: an unchanged danger flag would read differently
purely because the food moved, which removes the fixed meaning the network
needs in order to learn a rule about that flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import NetworkConfig
from .activations import resolve


@dataclass(frozen=True)
class Architecture:
    """Shape of a network, sufficient to rebuild it from a checkpoint."""

    input_size: int
    hidden_layers: tuple[int, ...]
    hidden_activations: tuple[str, ...]
    output_size: int

    @classmethod
    def from_config(cls, config: NetworkConfig, input_size: int,
                    output_size: int) -> Architecture:
        """The hidden layers come from the configuration; the two ends from the game."""
        if len(config.hidden_layers) != len(config.hidden_activations):
            raise ValueError("hidden_layers and hidden_activations must be the same length")
        return cls(
            input_size=input_size,
            hidden_layers=tuple(config.hidden_layers),
            hidden_activations=tuple(config.hidden_activations),
            output_size=output_size,
        )

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.input_size, *self.hidden_layers, self.output_size)

    @property
    def genome_size(self) -> int:
        sizes = self.layer_sizes
        return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_layers": list(self.hidden_layers),
            "hidden_activations": list(self.hidden_activations),
            "output_size": self.output_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Architecture:
        return cls(
            input_size=int(data["input_size"]),
            hidden_layers=tuple(int(n) for n in data["hidden_layers"]),
            hidden_activations=tuple(str(a) for a in data["hidden_activations"]),
            output_size=int(data["output_size"]),
        )


class PolicyNetwork:
    """Dense network whose parameters live in a single flat genome vector.

    One instance is reused across an entire population: `load` rebinds it to a
    different genome in place, which avoids allocating a network per individual.
    """

    __slots__ = ("architecture", "_weights", "_biases", "_activations")

    def __init__(self, architecture: Architecture):
        self.architecture = architecture
        sizes = architecture.layer_sizes
        self._weights = [
            np.zeros((sizes[i], sizes[i + 1]), dtype=np.float32) for i in range(len(sizes) - 1)
        ]
        self._biases = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(sizes) - 1)]
        self._activations = [resolve(name) for name in architecture.hidden_activations]

    def load(self, genome: np.ndarray) -> PolicyNetwork:
        """Unpack a flat genome into the layer matrices."""
        expected = self.architecture.genome_size
        if genome.size != expected:
            raise ValueError(f"genome has {genome.size} parameters, architecture needs {expected}")
        offset = 0
        for i, (weight, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            count = weight.size
            self._weights[i] = genome[offset : offset + count].reshape(weight.shape)
            offset += count
            count = bias.size
            self._biases[i] = genome[offset : offset + count]
            offset += count
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Logits for one state. Input is used as given."""
        current = x
        last = len(self._weights) - 1
        for i, (weight, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            current = current @ weight + bias
            if i < last:
                current = self._activations[i](current)
        return current

    def act(self, x: np.ndarray) -> int:
        """Index of the highest-scoring action."""
        return int(np.argmax(self.forward(x)))

    def layer_arrays(self) -> dict[str, np.ndarray]:
        """Named copies of the parameters, for standalone model exports."""
        arrays: dict[str, np.ndarray] = {}
        for i, (weight, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            arrays[f"weight_{i}"] = np.array(weight, dtype=np.float32)
            arrays[f"bias_{i}"] = np.array(bias, dtype=np.float32)
        return arrays


def random_genome(architecture: Architecture, rng: np.random.Generator, scale: float = 1.0,
                  output_scale: float = 0.5) -> np.ndarray:
    """Layer-wise scaled initialisation, flattened into a genome.

    Hidden layers use a fan-in scaling so activations neither saturate nor
    vanish at depth; the output layer starts smaller so early policies are not
    locked into one action before selection has any signal to work with.
    """
    sizes = architecture.layer_sizes
    parts: list[np.ndarray] = []
    last = len(sizes) - 2
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        spread = (output_scale if i == last else scale) * np.sqrt(2.0 / fan_in)
        parts.append((rng.standard_normal(fan_in * fan_out) * spread).astype(np.float32))
        parts.append(np.zeros(fan_out, dtype=np.float32))
    return np.concatenate(parts)
