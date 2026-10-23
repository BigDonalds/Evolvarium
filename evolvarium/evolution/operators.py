"""Crossover and mutation.

Measured on this network, blending two unaligned genomes preserves markedly less
of either parent's behaviour than mutation does. Two hidden units that compute
the same thing in different parents sit at different indices, so averaging them
produces a unit that matches neither. Crossover therefore stays a minority
operator and mutation carries most of the search.
"""

from __future__ import annotations

import numpy as np

BLENDS = ("weighted_average", "uniform", "single_point", "layer_uniform")
MUTATIONS = ("gaussian", "polynomial")


def blend(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator,
          method: str = "weighted_average",
          layer_sizes: tuple[int, ...] | None = None) -> np.ndarray:
    """Combine two genomes.

    `layer_sizes` is the network's, inputs first, and only `layer_uniform`
    needs it.
    """
    if method == "weighted_average":
        weight = rng.beta(2.0, 2.0)
        return (weight * parent_a + (1.0 - weight) * parent_b).astype(np.float32)

    if method == "uniform":
        mask = rng.random(parent_a.size) < 0.5
        return np.where(mask, parent_a, parent_b).astype(np.float32)

    if method == "single_point":
        cut = int(rng.integers(1, parent_a.size))
        return np.concatenate([parent_a[:cut], parent_b[cut:]]).astype(np.float32)

    if method == "layer_uniform":
        # Each unit takes all of its input weights and its bias from one parent,
        # which is less disruptive than mixing individual connections.
        mask = _unit_mask(layer_sizes, parent_a.size, rng)
        return np.where(mask, parent_a, parent_b).astype(np.float32)

    raise ValueError(f"unknown crossover method {method!r}")


def _unit_mask(layer_sizes: tuple[int, ...] | None, size: int,
               rng: np.random.Generator) -> np.ndarray:
    """True for every gene of a unit that comes from the first parent.

    The genome holds each layer as its weights, one row per input, then its
    biases, as the network unpacks it. A unit's input weights are a column of
    that block, so its choice repeats once per row and once more for its bias.
    """
    if layer_sizes is None:
        raise ValueError("layer_uniform crossover needs the network's layer sizes")
    parts: list[np.ndarray] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
        first = rng.random(fan_out) < 0.5
        parts.append(np.tile(first, fan_in))
        parts.append(first)
    mask = np.concatenate(parts)
    if mask.size != size:
        raise ValueError(f"layer sizes {layer_sizes} need {mask.size} genes, genome has {size}")
    return mask


def mutate(genome: np.ndarray, rng: np.random.Generator, rate: float, strength: float,
           method: str = "gaussian", eta: float = 20.0) -> np.ndarray:
    """Perturb a fraction of the genes."""
    child = genome.copy()
    if rate <= 0.0 or strength <= 0.0:
        return child

    mask = rng.random(genome.size) < rate
    selected = int(mask.sum())
    if selected == 0:
        return child

    if method == "gaussian":
        child[mask] += (rng.standard_normal(selected) * strength).astype(np.float32)
    elif method == "polynomial":
        u = rng.random(selected)
        delta = np.where(
            u < 0.5,
            np.power(2.0 * u, 1.0 / (eta + 1.0)) - 1.0,
            1.0 - np.power(2.0 * (1.0 - u), 1.0 / (eta + 1.0)),
        )
        child[mask] += (delta * strength).astype(np.float32)
    else:
        raise ValueError(f"unknown mutation method {method!r}")

    return child.astype(np.float32)
