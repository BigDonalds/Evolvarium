"""Activation functions available to evolved networks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0.0, z, 0.01 * z)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def linear(z: np.ndarray) -> np.ndarray:
    return z


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "linear": linear,
}


def resolve(name: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown activation {name!r}; available: {sorted(ACTIVATIONS)}"
        ) from None
