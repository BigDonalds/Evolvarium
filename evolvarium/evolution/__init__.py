"""Genetic algorithm: individuals, operators, selection, population."""

from .individual import Individual
from .population import GenerationReport, Population

__all__ = ["Individual", "Population", "GenerationReport"]
