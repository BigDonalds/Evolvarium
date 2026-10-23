"""Crossover operators."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from evolvarium.config import CONFIG
from evolvarium.evolution import operators
from evolvarium.evolution.population import Population
from evolvarium.nn.network import Architecture, PolicyNetwork

ARCHITECTURE = Architecture(
    input_size=5, hidden_layers=(4, 3), hidden_activations=("tanh", "tanh"), output_size=2,
)


def _units(architecture: Architecture, genome: np.ndarray) -> list[np.ndarray]:
    """Each unit's input weights and bias, as the network reads them from a genome."""
    arrays = PolicyNetwork(architecture).load(genome).layer_arrays()
    sizes = architecture.layer_sizes
    return [
        np.append(arrays[f"weight_{k}"][:, j], arrays[f"bias_{k}"][j])
        for k in range(len(sizes) - 1)
        for j in range(sizes[k + 1])
    ]


def _parents(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Two genomes with no gene value used twice, in either one or across both."""
    first = np.arange(1, size + 1, dtype=np.float32)
    return first, -first


def _from_first(child: np.ndarray, first: np.ndarray, second: np.ndarray) -> list[bool]:
    """Which parent each unit of the child came from, failing on a unit from both."""
    taken = []
    for unit, a, b in zip(_units(ARCHITECTURE, child), _units(ARCHITECTURE, first),
                          _units(ARCHITECTURE, second), strict=True):
        assert np.array_equal(unit, a) or np.array_equal(unit, b)
        taken.append(bool(np.array_equal(unit, a)))
    return taken


def test_layer_uniform_takes_each_unit_whole_from_one_parent():
    first, second = _parents(ARCHITECTURE.genome_size)
    rng = np.random.default_rng(3)

    draws = [
        _from_first(
            operators.blend(first, second, rng, "layer_uniform", ARCHITECTURE.layer_sizes),
            first, second,
        )
        for _ in range(400)
    ]

    share = np.mean(draws, axis=0)
    assert np.all((share > 0.4) & (share < 0.6))


def test_uniform_still_mixes_the_genes_of_a_unit():
    first, second = _parents(ARCHITECTURE.genome_size)
    child = operators.blend(first, second, np.random.default_rng(3), "uniform")

    with pytest.raises(AssertionError):
        _from_first(child, first, second)


def test_layer_uniform_needs_layer_sizes_that_fit_the_genome():
    first, second = _parents(ARCHITECTURE.genome_size)
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="layer sizes"):
        operators.blend(first, second, rng, "layer_uniform")
    with pytest.raises(ValueError, match="genome has"):
        operators.blend(first, second, rng, "layer_uniform", (5, 4, 2))


def test_a_population_breeds_layer_uniform_children_from_whole_units():
    config = CONFIG.with_overrides(
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=12, evaluation_episodes=1, elite_count=2,
            immigrant_count=1, crossover_rate=1.0, crossover_method="layer_uniform",
            mutation_rate=0.0, stagnation_step=0.0,
        )
    )
    population = Population(config, seed=4)
    genomes = {m.id: m.genome.copy() for m in population.members}
    architecture = population.architecture

    population.run_generation()

    children = [m for m in population.members if m.origin == "crossover"]
    assert len(children) == 9
    sources = set()
    for child in children:
        parents = [_units(architecture, genomes[p]) for p in child.parents]
        for k, unit in enumerate(_units(architecture, child.genome)):
            matches = [np.array_equal(unit, units[k]) for units in parents]
            assert any(matches)
            if len(set(child.parents)) == 2 and matches.count(True) == 1:
                sources.add(matches.index(True))
    assert sources == {0, 1}
