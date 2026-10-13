"""Scoring a genome, in this process or another one.

Evaluation is the only expensive part of a generation and genomes never interact
during it, so it parallelises cleanly. The whole design here exists to make the
parallel path produce *identical* results to the serial one, because a speedup
that quietly changes selection is worse than no speedup.

Three rules keep them identical:

* Seeds are decided by the parent before any work is dispatched, and passed in.
  If a worker chose its own, genomes would be scored on different food sequences
  and selection would reward luck instead of policy.
* No worker touches the population RNG. Exploration, when enabled, draws from a
  generator seeded off the episode seed, so it is a function of the task rather
  than of the order tasks happen to run in.
* Workers return data, never objects the parent mutates. The parent applies
  results in the original order after every task completes.

Replays are packed inside the worker. A recorded episode holds thousands of frame
objects; packed it is a few kilobytes, and that difference is the difference
between a speedup and a serialisation bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import BoardConfig, EpisodeConfig, FeatureConfig, FitnessConfig
from ..features import FeatureSpec
from ..game.board import Turn
from ..game.engine import DeathCause, Episode
from ..nn.network import Architecture, PolicyNetwork
from .fitness import evaluate


@dataclass(frozen=True)
class EvalContext:
    """Everything a worker needs to score a genome. Picklable by construction."""

    spec_name: str
    architecture: dict
    board: BoardConfig
    episode: EpisodeConfig
    fitness: FitnessConfig
    features: FeatureConfig
    exploration_rate: float
    replay_steps: int


@dataclass
class EvalResult:
    """What scoring produces. Plain data, safe to send between processes."""

    fitness: float
    score: float
    best_episode_score: int
    steps: float
    terms: dict[str, float]
    death: DeathCause
    death_cell: tuple[int, int]
    episodes: int
    replay: dict | None = None


class _Runner:
    """Holds the spec and network so they are built once, not once per genome."""

    def __init__(self, ctx: EvalContext):
        self.ctx = ctx
        self.spec = FeatureSpec(ctx.spec_name, ctx.features)
        self.network = PolicyNetwork(Architecture.from_dict(ctx.architecture))

    def play(self, genome: np.ndarray, seed: int, record: bool) -> Any:
        ctx = self.ctx
        episode = Episode(ctx.board, ctx.episode, seed, record=record)
        network = self.network.load(genome)
        rate = ctx.exploration_rate
        # Seeded from the episode, so exploration is reproducible and identical
        # whether this runs here or in a worker process.
        rng = np.random.default_rng(seed ^ 0x5EED) if rate > 0.0 else None

        while not episode.finished:
            if rng is not None and rng.random() < rate:
                action = int(rng.integers(3))
            else:
                action = network.act(self.spec(episode))
            episode.step(Turn(action))
        return episode.result

    def score(self, genome: np.ndarray, base_seed: int, trials: int, record: bool) -> EvalResult:
        ctx = self.ctx
        results = []
        fitnesses = []
        totals: dict[str, float] = {}

        for trial in range(trials):
            # Every genome in a generation meets the same seed sequence, so
            # differences between them reflect policy rather than food luck.
            result = self.play(genome, base_seed + trial, record and trial == 0)
            fitness, terms = evaluate(result, ctx.fitness, ctx.board)
            results.append(result)
            fitnesses.append(fitness)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value

        worst = min(results, key=lambda r: r.score)
        replay = None
        if record:
            from ..telemetry.history import pack_replay

            replay = pack_replay(results[0], ctx.replay_steps)

        return EvalResult(
            fitness=float(np.mean(fitnesses)),
            score=float(np.mean([r.score for r in results])),
            best_episode_score=max(r.score for r in results),
            steps=float(np.mean([r.steps for r in results])),
            terms={k: v / trials for k, v in totals.items()},
            death=worst.death,
            death_cell=worst.death_cell,
            episodes=trials,
            replay=replay,
        )


# ---------------------------------------------------------------- worker side

_runner: _Runner | None = None


def init_worker(ctx: EvalContext) -> None:
    """Process initialiser: build the spec and network once per worker."""
    global _runner
    _runner = _Runner(ctx)


def run_task(task: tuple[np.ndarray, int, int, bool]) -> EvalResult:
    """Score one genome. Runs in a worker process."""
    genome, base_seed, trials, record = task
    assert _runner is not None, "worker was not initialised"
    return _runner.score(genome, base_seed, trials, record)


def make_runner(ctx: EvalContext) -> _Runner:
    """A runner for in-process use, so the serial path shares this exact code."""
    return _Runner(ctx)
