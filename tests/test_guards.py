"""Limits on a game's code, and workers that die or go quiet while scoring.

`Trouble` is a game that misbehaves on purpose. Workers import it from this
module, so the pool tests play it in other processes the same way they play a
built-in game.
"""

from __future__ import annotations

import dataclasses
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from evolvarium.config import CONFIG, LimitsConfig
from evolvarium.contract import Actions, Choice, Game, Observation
from evolvarium.events import EventBus, Note, RunFinished
from evolvarium.evolution import pool as pool_module
from evolvarium.evolution.evaluation import EvalContext, GameFault, TimeLimit, make_runner
from evolvarium.evolution.pool import EvaluationPool, WorkerLost
from evolvarium.nn.network import Architecture
from evolvarium.trainer import Trainer


@dataclass(frozen=True)
class TroubleSettings:
    # What goes wrong: "raise", "start", "wide", "hang", "exit", "slow" or
    # "endless", or "" for nothing.
    trouble: str = ""
    # The step it goes wrong on.
    at: int = 3
    # A file made the first time the trouble comes, after which it does not.
    once: str = ""


class Trouble(Game):
    """Six steps that do nothing, and whatever trouble its settings name."""

    name = "trouble"
    Settings = TroubleSettings
    observation = Observation(["turn"])
    actions = Actions(Choice("move", ("left", "right")))

    def reset(self, seed: int) -> None:
        self.turn = 0
        if self._due("start"):
            raise RuntimeError("no floor to stand on")

    def observe(self, agent: str):
        return [self.turn / 10, 0.0] if self._due("wide") else [self.turn / 10]

    def step(self, actions: dict) -> None:
        self.turn += 1
        if self.turn != self.settings.at:
            return
        if self._due("raise"):
            raise ValueError("no move called sideways")
        if self._due("hang"):
            time.sleep(3600)
        if self._due("exit"):
            os._exit(3)
        if self._due("slow"):
            time.sleep(0.5)

    def _due(self, trouble: str) -> bool:
        if self.settings.trouble != trouble:
            return False
        if not self.settings.once:
            return True
        marker = Path(self.settings.once)
        if marker.exists():
            return False
        marker.touch()
        return True

    @property
    def over(self) -> bool:
        return self.turn >= 6 and self.settings.trouble != "endless"

    @property
    def end(self) -> str:
        return "done"

    @property
    def score(self) -> int:
        return self.turn

    @property
    def steps(self) -> int:
        return self.turn


ARCHITECTURE = Architecture(1, (4,), ("tanh",), Trouble.actions.size)
GENOME = np.random.default_rng(1).normal(size=ARCHITECTURE.genome_size).astype(np.float32)
# Small enough that a test of them is quick, with room for a slow machine.
QUICK = LimitsConfig(step_seconds=0.2, episode_seconds=0.5)


def context(limits: LimitsConfig | None = None, **trouble) -> EvalContext:
    return EvalContext(game="test_guards:Trouble", settings=TroubleSettings(**trouble),
                       architecture=ARCHITECTURE.to_dict(), exploration_rate=0.0,
                       replay_steps=0, limits=limits or LimitsConfig())


def task(seed: int) -> tuple:
    return (GENOME, seed, 2, False)


# ------------------------------------------------------------- in this process


@pytest.mark.parametrize(("trouble", "message"), [
    ("raise", r"Trouble\.step raised ValueError: no move called sideways "
              r"on step 3 of the episode with seed 7 \(test_guards\.py, line \d+\)"),
    ("start", r"Trouble\.reset raised RuntimeError: no floor to stand on "
              r"while starting the episode with seed 7 \(test_guards\.py, line \d+\)"),
    ("wide", r"reading Trouble\.observe's output as inputs raised ValueError: observation "
             r"has shape \(2,\); the game declares 1 inputs on step 1 of the episode with "
             r"seed 7 \(contract\.py, line \d+\)"),
])
def test_a_fault_names_the_call_the_step_and_the_seed(trouble, message):
    runner = make_runner(context(trouble=trouble))
    with pytest.raises(GameFault) as raised:
        runner.score(GENOME, 7, 2, False)
    assert re.fullmatch(message, str(raised.value))
    assert raised.value.__cause__ is not None


def test_a_step_that_runs_over_once_is_played_again(tmp_path):
    runner = make_runner(context(QUICK, trouble="slow", once=str(tmp_path / "slow")))
    result = runner.score(GENOME, 7, 2, False)
    assert result.episodes == 2 and result.score == 6
    [note] = result.notes
    assert re.fullmatch(
        r"step 3 of the episode with seed 7 took \d+ ms, more than limits\.step_seconds "
        r"allows \(0\.2 s\); played again, it finished in time", note)


def test_a_step_that_runs_over_twice_ends_the_job():
    runner = make_runner(context(QUICK, trouble="slow"))
    with pytest.raises(TimeLimit) as raised:
        runner.score(GENOME, 7, 1, False)
    assert not raised.value.endless
    assert str(raised.value).endswith("It ran over the first time it was played as well")


def test_an_episode_that_never_ends_is_ended_with_a_hint():
    runner = make_runner(context(QUICK, trouble="endless"))
    started = time.perf_counter()
    with pytest.raises(TimeLimit) as raised:
        runner.score(GENOME, 7, 1, False)
    assert time.perf_counter() - started < 5
    assert raised.value.endless
    assert re.fullmatch(
        r"the episode with seed 7 was still going after \d+ ms and [\d,]+ steps, more than "
        r"limits\.episode_seconds allows \(0\.5 s\)\. It ran over the first time it was "
        r"played as well\. A game whose episodes can go on forever needs a rule that ends "
        r"one after so many steps, or probe\.enabled=true to work one out", str(raised.value))


def test_the_trainer_passes_notes_on(tmp_path):
    config = CONFIG.with_overrides(
        game="test_guards:Trouble",
        settings=TroubleSettings(trouble="slow", once=str(tmp_path / "slow")),
        network=dataclasses.replace(CONFIG.network, hidden_layers=(4,),
                                    hidden_activations=("tanh",)),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=6, evaluation_episodes=1, elite_count=1,
            immigrant_count=1, workers=1,
        ),
        limits=QUICK,
        paths=dataclasses.replace(
            CONFIG.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
    )
    events = EventBus()
    heard = []
    events.subscribe(heard.append, Note, RunFinished)
    trainer = Trainer(config, run_id="notes", seed=3, events=events, max_generations=1)
    trainer.start()
    trainer.join()
    trainer.stop()

    notes = [e.message for e in heard if isinstance(e, Note)]
    assert any(re.fullmatch(r"generation 0: step 3 of the episode with seed \d+ took .*; "
                            r"played again, it finished in time", note) for note in notes)
    assert heard[-1].reason == "limit"


# ------------------------------------------------------------------- in workers


def test_a_fault_in_a_worker_reads_as_it_would_here():
    serial = make_runner(context(trouble="raise"))
    with pytest.raises(GameFault) as here:
        serial.score(GENOME, 10, 2, False)

    pool = EvaluationPool(context(trouble="raise"), 1, own_process=True)
    try:
        with pytest.raises(GameFault) as there:
            pool.map([task(10)])
        assert str(there.value) == str(here.value)
        # Ending early closes the pool, which then refuses more work.
        assert not pool.active
        with pytest.raises(RuntimeError, match="the pool is closed"):
            pool.map([task(10)])
    finally:
        pool.close()


def test_a_worker_that_dies_is_replaced_and_its_genome_played_again(tmp_path):
    expected = [make_runner(context()).score(*task(seed)).fitness for seed in (10, 20)]

    with EvaluationPool(context(trouble="exit", once=str(tmp_path / "exit")), 1,
                        own_process=True) as pool:
        results = pool.map([task(10), task(20)])
        assert [r.fitness for r in results] == expected
        [note] = pool.take_notes()
        assert re.fullmatch(
            r"a worker stopped with exit code 3 while scoring a genome, last heard from at "
            r"step \d of the episode with seed 10, so the genome was played again", note)
        assert pool.take_notes() == []
        assert pool.active


def test_a_genome_that_kills_every_worker_ends_the_job():
    with EvaluationPool(context(trouble="exit"), 1, own_process=True) as pool:
        with pytest.raises(WorkerLost) as raised:
            pool.map([task(10)])
        assert not pool.active
    lost = ("a worker stopped with exit code 3 while scoring a genome, "
            "last heard from at step 1 of the episode with seed 10")
    assert str(raised.value) == (
        f"{lost}. It was the second worker lost on that genome; the first time, {lost}")


def test_a_worker_stuck_in_a_step_is_ended(monkeypatch):
    monkeypatch.setattr(pool_module, "SLACK", 0.2)
    started = time.perf_counter()
    with EvaluationPool(context(QUICK, trouble="hang"), 1, own_process=True) as pool:
        with pytest.raises(WorkerLost) as raised:
            pool.map([task(10)])
    assert time.perf_counter() - started < 30
    assert re.fullmatch(
        r"a worker went \d\.\d s without a word, longer than limits\.step_seconds allows "
        r"\(0\.2 s\) while scoring a genome, last heard from at step 1 of the episode with "
        r"seed 10\. It was the second worker lost on that genome; the first time, .*",
        str(raised.value))
