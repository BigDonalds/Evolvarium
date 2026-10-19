"""Scoring a genome, in this process or another one.

Evaluation is the only expensive part of a generation and genomes never interact
during it, so it parallelises cleanly. The whole design here exists to make the
parallel path produce *identical* results to the serial one, because a speedup
that quietly changes selection is worse than no speedup.

Three rules keep them identical:

* Seeds are decided by the parent before any work is dispatched, and passed in.
  If a worker chose its own, genomes would be scored on different episodes and
  selection would reward luck instead of policy.
* No worker touches the population RNG. Exploration, when enabled, draws from a
  generator seeded off the episode seed, so it is a function of the task rather
  than of the order tasks happen to run in.
* Workers return data, never objects the parent mutates. The parent applies
  results in the original order after every task completes.

Replays are packed inside the worker as the episode plays. A recorded episode
can run to thousands of frames; packed it is a few kilobytes, and that
difference is the difference between a speedup and a serialisation bottleneck.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import games
from ..contract import Outcome
from ..nn.network import Architecture, PolicyNetwork
from ..replays import Recorder


@dataclass(frozen=True)
class EvalContext:
    """Everything a worker needs to score a genome. Picklable by construction.

    `game` is a name the registry resolves, never a game object, so a worker
    builds its own copy and nothing about a live episode crosses processes.
    """

    game: str
    settings: Any
    architecture: dict
    exploration_rate: float
    replay_steps: int


@dataclass
class EvalResult:
    """What scoring produces. Plain data, safe to send between processes."""

    fitness: float
    score: float
    best_episode_score: float
    # Steps of the best episode, the one whose replay is kept. `steps` is the
    # mean over every episode.
    best_episode_steps: int
    steps: float
    terms: dict[str, float]
    signals: dict[str, float]
    # How the worst episode ended, which is the informative failure.
    end: str
    episodes: int
    replay: dict | None = None


class _Runner:
    """Holds the game and network so they are built once, not once per genome."""

    def __init__(self, ctx: EvalContext):
        self.ctx = ctx
        self.game = games.create(ctx.game, ctx.settings)
        self.network = PolicyNetwork(Architecture.from_dict(ctx.architecture))
        self.higher = self.game.direction == "higher"
        self.scene = self.game.scene()

    def play(self, genome: np.ndarray, seed: int, record: bool) -> Outcome:
        game = self.game
        game.reset(seed)
        recorder = None
        if record and self.scene is not None:
            recorder = Recorder(self.scene, self.ctx.replay_steps)
            recorder.add(game.draw())
        forward = self.network.load(genome).forward
        rate = self.ctx.exploration_rate
        # Seeded from the episode, so exploration is reproducible and identical
        # whether this runs here or in a worker process.
        rng = np.random.default_rng(seed ^ 0x5EED) if rate > 0.0 else None

        agents = game.agents
        observe = game.observe
        encode = game.observation.encode
        decode = game.actions.decode
        explore = game.actions.explore
        step = game.step

        while not game.over:
            actions = {}
            for agent in agents:
                if rng is not None and rng.random() < rate:
                    actions[agent] = explore(rng)
                else:
                    actions[agent] = decode(forward(encode(observe(agent))))
            step(actions)
            if recorder is not None:
                if recorder.full:
                    recorder.truncated = True
                else:
                    recorder.add(game.draw())

        fitness, terms = game.goal()
        return Outcome(
            score=game.score,
            steps=game.steps,
            end=game.end,
            fitness=fitness,
            terms=terms,
            signals=game.signals(),
            replay=None if recorder is None else recorder.pack(),
        )

    def score(self, genome: np.ndarray, base_seed: int, trials: int, record: bool) -> EvalResult:
        # Every genome in a generation meets the same seed sequence, so
        # differences between them reflect policy rather than luck.
        outcomes = [self.play(genome, base_seed + trial, record) for trial in range(trials)]
        terms: dict[str, float] = {}
        signals: dict[str, float] = {}
        for outcome in outcomes:
            for key, value in outcome.terms.items():
                terms[key] = terms.get(key, 0.0) + value
            for key, value in outcome.signals.items():
                signals[key] = signals.get(key, 0.0) + value

        scores = [o.score for o in outcomes]
        if self.higher:
            best = max(scores)
            worst = min(outcomes, key=lambda o: o.score)
        else:
            best = min(scores)
            worst = max(outcomes, key=lambda o: o.score)
        # The replay kept is the best episode's, the one whose score is shown
        # with it. Of episodes that tie, the earliest.
        shown = next(o for o in outcomes if o.score == best)

        return EvalResult(
            fitness=float(np.mean([o.fitness for o in outcomes])),
            score=float(np.mean(scores)),
            best_episode_score=best,
            best_episode_steps=shown.steps,
            steps=float(np.mean([o.steps for o in outcomes])),
            terms={k: v / trials for k, v in terms.items()},
            signals={k: v / trials for k, v in signals.items()},
            end=worst.end,
            episodes=trials,
            replay=shown.replay,
        )


# ---------------------------------------------------------------- worker side

_runner: _Runner | None = None


def init_worker(ctx: EvalContext) -> None:
    """Process initialiser: build the game and network once per worker.

    Workers ignore Ctrl+C. The console delivers it to every process attached to
    it, and a worker that died on it would take the generation in progress with
    it, where the trainer wants to finish that generation and save.
    """
    global _runner
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _runner = _Runner(ctx)


def run_task(task: tuple[np.ndarray, int, int, bool]) -> EvalResult:
    """Score one genome. Runs in a worker process."""
    genome, base_seed, trials, record = task
    assert _runner is not None, "worker was not initialised"
    return _runner.score(genome, base_seed, trials, record)


def make_runner(ctx: EvalContext) -> _Runner:
    """A runner for in-process use, so the serial path shares this exact code."""
    return _Runner(ctx)
