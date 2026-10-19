"""Generational loop: evaluate, rank, reproduce.

Two invariants hold the whole thing together and both are enforced here:

1. Every individual is scored only on episodes played in the current generation.
   `reset_scores` runs across the population before evaluation begins, so an
   elite that has survived twenty generations enters selection with exactly the
   same clean slate as a newborn.

2. A genome's score is the MEAN of its evaluation episodes, never the best.
   Taking the best of several trials rewards a favourable episode rather than a
   better policy, and the advantage compounds for whichever individuals happen
   to be evaluated most often.

Fitness is always maximised. A game that wants its score low says so, and its
goal turns the score round before it reaches selection. Scores themselves keep
the game's direction, so "best" and "record" below follow it.
"""

from __future__ import annotations

import time

import numpy as np

from .. import games
from ..config import Config
from ..nn.network import Architecture, PolicyNetwork, random_genome
from . import operators, selection, speciation
from .evaluation import EvalContext, EvalResult, make_runner
from .individual import Individual


class GenerationReport:
    """Everything measured about one finished generation."""

    __slots__ = (
        "generation", "population", "best", "best_fitness", "best_score",
        "mean_fitness", "mean_score", "max_episode_score", "scoring_fraction",
        "mean_steps", "ends", "diversity", "species_count", "mutation_rate",
        "stagnation", "elapsed", "episodes", "record", "evaluated",
    )

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": round(self.best_fitness, 2),
            "best_score": round(self.best_score, 3),
            "max_episode_score": self.max_episode_score,
            "mean_fitness": round(self.mean_fitness, 2),
            "mean_score": round(self.mean_score, 4),
            "scoring_fraction": round(self.scoring_fraction, 4),
            "mean_steps": round(self.mean_steps, 1),
            "ends": dict(self.ends),
            "diversity": round(self.diversity, 4),
            "species_count": self.species_count,
            "mutation_rate": round(self.mutation_rate, 4),
            "stagnation": self.stagnation,
            "elapsed": round(self.elapsed, 3),
            "episodes": self.episodes,
            "record": self.record,
            "best_individual": self.best.summary() if self.best else None,
        }


class Population:
    """Owns the genomes and drives one generation at a time."""

    def __init__(self, config: Config, seed: int | None = None):
        config = games.resolve(config)
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.game = games.create(config.game, config.settings)
        self.higher = self.game.direction == "higher"
        # Described once, as the dashboard reads it with every generation.
        scene = self.game.scene()
        self.scene = None if scene is None else scene.describe()
        self.architecture = Architecture.from_config(
            config.network, self.game.observation.size, self.game.actions.size
        )
        # One network instance is rebound per individual rather than allocating
        # a network per genome.
        self.network = PolicyNetwork(self.architecture)

        # Scoring lives in `evaluation` so the serial and parallel paths run the
        # same code and cannot drift apart.
        self.context = EvalContext(
            game=config.game,
            settings=config.settings,
            architecture=self.architecture.to_dict(),
            exploration_rate=config.evolution.exploration_rate,
            replay_steps=config.history.replay_steps,
        )
        self._runner = make_runner(self.context)

        self.generation = 0
        self.total_episodes = 0
        # None until the first generation is scored: any first score is a
        # record, whichever way the game counts.
        self.best_ever_score: float | None = None
        self.best_ever_fitness = 0.0
        self.best_ever_genome: np.ndarray | None = None
        self.best_ever_generation = 0
        self.stagnation = 0
        self.started_at = time.time()

        self.species: list[speciation.Species] = []
        self._next_species_id = 0
        self.compatibility_threshold = config.evolution.compatibility_threshold

        self.members: list[Individual] = [
            Individual(
                genome=random_genome(
                    self.architecture,
                    self.rng,
                    config.network.weight_init_scale,
                    config.network.output_init_scale,
                ),
                generation=0,
                origin="seed",
            )
            for _ in range(config.evolution.population_size)
        ]

    # ------------------------------------------------------------------ evaluation

    def _apply(self, individual: Individual, outcome: EvalResult) -> None:
        """Copy a scoring result onto the individual it belongs to."""
        individual.fitness = outcome.fitness
        individual.score = outcome.score
        individual.best_episode_score = outcome.best_episode_score
        individual.best_episode_steps = outcome.best_episode_steps
        individual.steps = outcome.steps
        individual.terms = outcome.terms
        individual.signals = outcome.signals
        individual.end = outcome.end
        individual.replay = outcome.replay
        individual.evaluated = True
        self.total_episodes += outcome.episodes

    def evaluate_individual(self, individual: Individual, base_seed: int,
                            record: bool = False) -> EvalResult:
        """Play the configured number of episodes and store the mean outcome."""
        trials = max(1, self.config.evolution.evaluation_episodes)
        outcome = self._runner.score(individual.genome, base_seed, trials, record)
        self._apply(individual, outcome)
        return outcome

    # ------------------------------------------------------------------ generation

    def run_generation(self, record: bool = False, pool=None) -> GenerationReport:
        """Evaluate every member, record the outcome, then breed the next set.

        `pool` may be an active EvaluationPool. Results are identical either way:
        the seed sequence is fixed here before any work is handed out, and
        results are applied in the members' own order rather than in whatever
        order workers happen to finish.
        """
        started = time.time()
        episodes_before = self.total_episodes

        # Invariant 1: nothing survives from an earlier generation's scoring.
        for member in self.members:
            member.reset_scores()

        trials = max(1, self.config.evolution.evaluation_episodes)
        base_seed = int(self.rng.integers(1 << 30))

        if pool is not None and pool.active:
            tasks = [(m.genome, base_seed, trials, record) for m in self.members]
            for member, outcome in zip(self.members, pool.map(tasks), strict=True):
                self._apply(member, outcome)
        else:
            for member in self.members:
                self.evaluate_individual(member, base_seed, record=record)

        ranked = sorted(self.members, key=lambda i: i.fitness, reverse=True)
        best = ranked[0]

        record = self.best_ever_score is None or self.better(
            best.best_episode_score, self.best_ever_score
        )
        if record:
            self.best_ever_score = best.best_episode_score
            self.best_ever_fitness = best.fitness
            self.best_ever_genome = best.genome.copy()
            self.best_ever_generation = self.generation
            self.stagnation = 0
        else:
            self.stagnation += 1

        ends: dict[str, int] = {}
        for member in self.members:
            key = member.end or "unknown"
            ends[key] = ends.get(key, 0) + 1
        episode_scores = [m.best_episode_score for m in self.members]

        mutation_rate = self._current_mutation_rate()
        report = GenerationReport(
            # The members that actually played, before breeding replaces them.
            evaluated=list(self.members),
            generation=self.generation,
            population=len(self.members),
            best=best,
            best_fitness=best.fitness,
            best_score=best.score,
            # The best single episode, which is the lowest when lower is better.
            max_episode_score=max(episode_scores) if self.higher else min(episode_scores),
            mean_fitness=float(np.mean([m.fitness for m in self.members])),
            mean_score=float(np.mean([m.score for m in self.members])),
            scoring_fraction=float(np.mean([m.best_episode_score > 0 for m in self.members])),
            mean_steps=float(np.mean([m.steps for m in self.members])),
            ends=ends,
            diversity=self._diversity(),
            species_count=len(self.species),
            mutation_rate=mutation_rate,
            stagnation=self.stagnation,
            elapsed=time.time() - started,
            episodes=self.total_episodes - episodes_before,
            record=record,
        )

        self.members = self._breed(ranked, mutation_rate)
        self.generation += 1
        return report

    def better(self, score: float, than: float) -> bool:
        """Whether `score` beats `than` in the game's direction."""
        return score > than if self.higher else score < than

    # ------------------------------------------------------------------ reproduction

    def _current_mutation_rate(self) -> float:
        evo = self.config.evolution
        widened = evo.mutation_rate + self.stagnation * evo.stagnation_step
        return float(min(widened, evo.max_mutation_rate))

    def _breed(self, ranked: list[Individual], mutation_rate: float) -> list[Individual]:
        evo = self.config.evolution
        next_generation = self.generation + 1
        offspring: list[Individual] = []

        pool = ranked
        if evo.use_speciation:
            self.species, self._next_species_id = speciation.assign(
                ranked, self.species, self.compatibility_threshold, self._next_species_id
            )
            self.compatibility_threshold = speciation.adjust_threshold(
                self.compatibility_threshold, len(self.species), evo.target_species
            )
            shared = speciation.apply_sharing(self.species)
            # Selection reads shared fitness while the reports keep raw values.
            pool = sorted(ranked, key=lambda i: shared.get(i.id, i.fitness), reverse=True)

        # Elites carry their genome forward unchanged; the copy is a fresh
        # individual so it has to earn its ranking again next generation.
        for parent in ranked[: evo.elite_count]:
            offspring.append(parent.descendant(parent.genome.copy(), "elite", next_generation))

        target = evo.population_size - evo.immigrant_count
        while len(offspring) < target:
            first = selection.select(pool, self.rng, evo.selection, evo.tournament_size)
            if self.rng.random() < evo.crossover_rate:
                second = selection.select(pool, self.rng, evo.selection, evo.tournament_size)
                genome = operators.blend(
                    first.genome, second.genome, self.rng, evo.crossover_method
                )
                parents = (first.id, second.id)
                origin = "crossover"
            else:
                genome = first.genome.copy()
                parents = (first.id,)
                origin = "mutation"

            genome = operators.mutate(
                genome, self.rng, mutation_rate, evo.mutation_strength,
                evo.mutation_method, evo.polynomial_eta,
            )
            offspring.append(first.descendant(genome, origin, next_generation, parents))

        while len(offspring) < evo.population_size:
            offspring.append(
                Individual(
                    genome=random_genome(
                        self.architecture, self.rng,
                        self.config.network.weight_init_scale,
                        self.config.network.output_init_scale,
                    ),
                    generation=next_generation,
                    origin="immigrant",
                )
            )
        return offspring

    # ------------------------------------------------------------------ diagnostics

    def _diversity(self, sample: int = 24) -> float:
        """Mean pairwise genome distance over a random sample, per parameter."""
        if len(self.members) < 2:
            return 0.0
        count = min(sample, len(self.members))
        picks = self.rng.choice(len(self.members), size=count, replace=False)
        genomes = np.stack([self.members[int(i)].genome for i in picks])
        total = 0.0
        pairs = 0
        for i in range(count):
            for j in range(i + 1, count):
                total += float(np.linalg.norm(genomes[i] - genomes[j]))
                pairs += 1
        return total / max(pairs, 1) / np.sqrt(self.architecture.genome_size)

    def replay_best(self, seed: int | None = None) -> tuple[Individual, EvalResult]:
        """Score one recorded episode with the all-time best genome."""
        genome = self.best_ever_genome
        if genome is None:
            genome = max(self.members, key=lambda i: i.fitness).genome
        individual = Individual(
            genome=genome, generation=self.best_ever_generation, origin="champion"
        )
        seed = seed if seed is not None else int(self.rng.integers(1 << 30))
        outcome = self._runner.score(genome, seed, trials=1, record=True)
        self._apply(individual, outcome)
        return individual, outcome


__all__ = ["Population", "GenerationReport"]
