"""The probe: what it finds in a game's play and what it does about it.

Each game here gets one thing wrong with it. The broken versions of the built-in
games are probed once each and the reports kept, on budgets a good deal smaller
than the default, so most tests only read a report.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import random
import re

import numpy as np
import pytest
from test_broken_games import EVERY, every
from test_guards import QUICK, TroubleSettings

from evolvarium import games, probe
from evolvarium.cli import main
from evolvarium.config import CONFIG, LimitsConfig, ProbeConfig
from evolvarium.contract import Actions, Choice, Game, Observation
from evolvarium.evolution.evaluation import GameFault
from evolvarium.evolution.pool import WorkerError
from evolvarium.fixes import FixedGame, with_fixes
from evolvarium.games import broken


class Plain(Game):
    """Forty steps with nothing wrong with them. The games below each change one thing."""

    name = "plain"
    observation = Observation(["turn", "wave"])
    actions = Actions(Choice("move", ("left", "right")))
    length = 40

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.turn = 0
        self.lefts = 0

    def observe(self, agent: str):
        return [self.turn / 50, math.sin(self.seed + self.turn)]

    def step(self, actions: dict) -> None:
        self.turn += 1
        self.lefts += actions["agent"] == "left"

    @property
    def over(self) -> bool:
        return self.turn >= self.length

    @property
    def end(self) -> str:
        return "done"

    @property
    def score(self) -> int:
        return self.lefts

    @property
    def steps(self) -> int:
        return self.turn


class Coded(Plain):
    """A kind of thing, given as one of four whole numbers, beside the wave."""

    name = "coded"
    observation = Observation(["kind", "wave"])

    def observe(self, agent: str):
        return [float((self.seed + self.turn) % 4), math.sin(self.seed + self.turn)]


class Counted(Coded):
    """A count that goes up every step: whole numbers, but far too many to be a code."""

    def observe(self, agent: str):
        return [float(self.turn), math.sin(self.seed + self.turn)]


class Quarters(Coded):
    """Four levels of something, as a share: few values, but not whole numbers."""

    def observe(self, agent: str):
        return [((self.seed + self.turn) % 4) / 3, math.sin(self.seed + self.turn)]


class Flag(Coded):
    """A yes or a no, which reads well enough as a number."""

    def observe(self, agent: str):
        return [float((self.seed + self.turn) % 2), math.sin(self.seed + self.turn)]


class NoGoal(Plain):
    @property
    def score(self) -> int:
        return 0


class RareGoal(Plain):
    """Gives anything for only one seed in all it plays."""

    @property
    def score(self) -> int:
        return 1 if self.seed == probe.SEED else 0


class OneStep(Plain):
    length = 1


class Unstoppable(Plain):
    """Never ends, and has no results until it does."""

    @property
    def over(self) -> bool:
        return False

    def goal(self):
        raise RuntimeError("no result yet")


class BadStop(Unstoppable):
    def stop(self, stalled: bool) -> None:
        raise ValueError("cannot stop here")


class BadGoal(Plain):
    def goal(self):
        raise ZeroDivisionError("no moves to divide by")


class CannotBuild(Plain):
    def __init__(self, settings=None):
        raise RuntimeError("no parts")


class NotANumber(Plain):
    observation = Observation(["turn", "ratio"])

    def observe(self, agent: str):
        return [self.turn / 50, math.nan if self.turn == 7 else math.sin(self.turn)]


class Infinite(NotANumber):
    def observe(self, agent: str):
        return [self.turn / 50, math.inf if self.turn == 7 else math.sin(self.turn)]


class Holds(Plain):
    """Holds a generator nothing seeds, the mistake the probe can put right."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self._chance = random.Random()


class InputSplits(Holds):
    def observe(self, agent: str):
        return [self.turn / 50, self._chance.random() if self.turn >= 5 else 0.0]


class ScoreSplits(Holds):
    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.bonus = 0.0

    def step(self, actions: dict) -> None:
        super().step(actions)
        if self.turn == 5:
            self.bonus = self._chance.random()

    @property
    def score(self) -> float:
        return self.lefts + self.bonus


class LengthSplits(Holds):
    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.length = 30 + self._chance.randrange(1000)


class OwnGenerator(Plain):
    """Builds a generator inside `reset`, where nothing outside can reach it."""

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.chance = random.Random()

    def observe(self, agent: str):
        return [self.turn / 50, self.chance.random()]


class Carries(Plain):
    """Carries a count from one episode to the next."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.played = 0

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.played += 1

    def observe(self, agent: str):
        return [self.turn / 50, self.played / 1000]


class FirstDiffers(Plain):
    """Its first episode differs from all the others, and those from one another not at all."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.started = 0

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.first = self.started == 0
        self.started += 1

    def observe(self, agent: str):
        return [self.turn / 50, 1.0 if self.first else 0.5]


class Shy(Plain):
    """An input that moves only in an episode past the fifth, once the moves have changed.

    Nothing else moves, so a network keeps to one move and only held actions set it off.
    """

    observation = Observation(["alarm", "flat"])

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.last = None
        self.alarm = 0.0

    def observe(self, agent: str):
        return [self.alarm, 1.0]

    def step(self, actions: dict) -> None:
        super().step(actions)
        move = actions["agent"]
        if self.last not in (None, move) and self.seed >= probe.SEED + 5:
            self.alarm = 1.0
        self.last = move


class Late(Plain):
    """Never ends. Its score stands still from step 50 and an input first moves at step 300."""

    observation = Observation(["turn", "late"])

    @property
    def over(self) -> bool:
        return False

    def observe(self, agent: str):
        return [self.turn / 1000, 1.0 if self.turn >= 300 else 0.0]

    @property
    def score(self) -> int:
        return min(self.turn, 50) // 2


class Flicker(Plain):
    """Ends when one move is chosen five times running, and its input is far out of scale.

    A network saturates on that input and repeats one move. Scaled, it can read the
    input flip and move with it for as long as anyone lets it.
    """

    observation = Observation(["flicker"])

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.last = None
        self.running = 0

    def observe(self, agent: str):
        return [100_000.0 + (100.0 if self.turn % 2 else -100.0)]

    def step(self, actions: dict) -> None:
        super().step(actions)
        move = actions["agent"]
        self.running = self.running + 1 if move == self.last else 1
        self.last = move

    @property
    def over(self) -> bool:
        return self.running >= 5


def config_of(game: str | type, settings=None, **budget):
    """A config for probing `game`, in as long as it takes unless `seconds` says otherwise."""
    reference = game if isinstance(game, str) else games.reference(game)
    limits = budget.pop("limits", CONFIG.limits)
    budget = {"seconds": 600.0, **budget}
    return CONFIG.with_overrides(game=reference, settings=settings, limits=limits,
                                 probe=ProbeConfig(enabled=True, **budget))


def cheaply(game: type, **budget) -> probe.Report:
    return probe.run(config_of(game, **{"episodes": 5, "search_steps": 0, **budget}))


# Enough for each game's rare moments to come up, and no more.
BUDGETS = {
    "snake": {"episodes": 5, "search_steps": 2000, "ceiling": 1000},
    "cartpole": {},
    "arena": {"episodes": 5, "search_steps": 10_000, "ceiling": 1000},
}


@functools.cache
def report_of(game: str, cls: type | None = None) -> probe.Report:
    budget = dict(BUDGETS[game])
    # Every held episode of an endless game is cut, and three of them end the
    # search for a fight, so a low ceiling sees too little to go looking for one.
    if cls is broken.EndlessArena:
        del budget["ceiling"]
    return probe.run(config_of(cls or game, **budget))


def fates(report: probe.Report) -> dict[str, str]:
    return {row.name: row.fate for row in report.inputs}


def found(report: probe.Report, kind: str) -> list[str]:
    return [finding.found for finding in report.findings if finding.kind == kind]


def did(report: probe.Report, kind: str) -> list[str]:
    return [finding.did for finding in report.findings if finding.kind == kind]


def play_out(game, seed: int, moves: int = 50_000) -> tuple[list[np.ndarray], list[float]]:
    """Every input a game gives on one seed, and its score at each step, with seeded moves."""
    rng = np.random.default_rng(seed)
    game.reset(seed)
    seen, scores = [], []
    for _ in range(moves):
        if game.over:
            break
        actions = {}
        for agent in game.agents:
            seen.append(game.observation.encode(game.observe(agent)))
            actions[agent] = game.actions.explore(rng)
        game.step(actions)
        scores.append(float(game.score))
    return seen, scores


# ------------------------------------------------------------------- games left alone


@pytest.mark.parametrize("game", games.names())
def test_a_clean_game_is_played_as_it_was_written(game):
    report = report_of(game)
    assert report.fixes.empty
    assert report.kinds() <= {"opposites"}
    assert {finding.outcome for finding in report.findings} <= {"noted"}
    assert set(fates(report).values()) == {"kept"}
    played = games.create(game)
    assert with_fixes(played, report.fixes) is played
    assert not report.out_of_time


def test_a_game_with_nothing_to_find_says_so():
    report = cheaply(Plain)
    assert not report.findings and report.fixes.empty
    assert "Found nothing to change." in report.lines()
    assert report.notes() == [f"probe: played 20 episodes in {report.seconds:.1f} s"]


def test_two_inputs_that_only_look_alike_are_both_kept():
    report = report_of("snake")
    assert found(report, "opposites")
    for finding in report.findings:
        assert finding.did == "Kept both."
        assert all(name in fates(report) for name in finding.inputs)
        assert all(fates(report)[name] == "kept" for name in finding.inputs)


def test_the_same_game_gets_the_same_report():
    def timeless(report):
        return dataclasses.replace(report, seconds=0.0, steps_per_second=0.0)

    config = config_of("cartpole", episodes=5, search_steps=5000)
    assert timeless(probe.run(config)) == timeless(probe.run(config))


def test_a_report_survives_being_written_down():
    for report in (report_of("snake"), report_of("snake", broken.VARIANTS["snake"]["unscaled"]),
                   report_of("cartpole", broken.VARIANTS["cartpole"]["endless"])):
        assert probe.Report.from_dict(json.loads(json.dumps(report.to_dict()))) == report


# ------------------------------------------------------------------- planted problems


@pytest.mark.parametrize(("game", "fault", "cls"), [
    pytest.param(game, fault, cls, id=f"{game}-{fault}") for game, fault, cls in EVERY
])
def test_every_planted_problem_is_found_and_named(game, fault, cls):
    report = report_of(game, cls)
    kinds = report.kinds() - {"opposites"}
    assert fault in kinds
    assert not report.fixes.empty or fault == "unseeded"
    if fault == "unseeded":
        return
    # Its limit of a billion steps makes one of Snake's inputs tiny as well.
    assert kinds == ({"endless", "unscaled"} if (game, fault) == ("snake", "endless") else {fault})


@pytest.mark.parametrize(("game", "cls"), every("dead"))
def test_dead_inputs_are_dropped_and_real_ones_kept(game, cls):
    report = report_of(game, cls)
    names = cls().observation.names
    assert report.fixes.keep == tuple(
        i for i, name in enumerate(names) if not name.startswith("constant "))
    for name, fate in fates(report).items():
        assert fate == ("never changes" if name.startswith("constant ") else "kept")
    assert "never changed" in found(report, "dead")[0]
    assert did(report, "dead") == [f"Dropped them. The network reads {len(names) - 4} inputs "
                                   f"instead of {len(names)}."]


@pytest.mark.parametrize(("game", "cls"), every("duplicate"))
def test_one_of_each_pair_of_duplicates_is_dropped(game, cls):
    report = report_of(game, cls)
    names = cls().observation.names
    copies = [name for name in names if name.startswith("copy of ")]
    assert len(copies) == 3
    kept = {names[i] for i in report.fixes.keep}
    assert len(kept) == len(names) - 3
    for copy in copies:
        assert (copy in kept) != (copy.removeprefix("copy of ") in kept)
    assert "only repeats" in found(report, "duplicate")[0]
    assert report.fixes.scale is None


@pytest.mark.parametrize(("game", "cls"), every("unscaled"))
def test_inputs_out_of_scale_are_scaled_and_those_in_scale_kept(game, cls):
    report = report_of(game, cls)
    factors = np.resize(broken._SCALES, len(report.inputs))
    assert report.fixes.keep is None and report.fixes.scale is not None
    for factor, row in zip(factors, report.inputs, strict=True):
        if factor == 1.0:
            assert row.fate == "kept", row
        elif factor != 30.0 and row.low != row.high:
            assert row.fate == "scaled", row
        if row.fate == "scaled":
            assert abs(row.after_low) <= 2.0 and abs(row.after_high) <= 2.0, row


def test_a_fix_for_scale_makes_the_same_sum_the_report_says():
    cls = broken.VARIANTS["cartpole"]["unscaled"]
    report = report_of("cartpole", cls)
    played = with_fixes(cls(), report.fixes)
    raw, _ = play_out(cls(), 5)
    fixed, _ = play_out(played, 5)
    offset, scale = np.array(report.fixes.offset), np.array(report.fixes.scale)
    np.testing.assert_allclose(fixed, (np.array(raw) + offset) * scale, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("game", "cls"), every("endless"))
def test_episodes_that_might_never_end_get_a_limit(game, cls):
    report = report_of(game, cls)
    assert report.fixes.step_cap is not None
    assert "still going" in found(report, "endless")[0]
    assert any(row.cut for row in report.players)

    played = FixedGame(cls(), report.fixes)
    seen, _ = play_out(played, 3, moves=report.fixes.step_cap + 5)
    assert played.over and played.steps <= report.fixes.step_cap
    assert len(seen) <= report.fixes.step_cap * len(played.agents)
    assert played.end
    played.goal()
    played.signals()


def test_a_game_that_only_seems_to_end_needs_the_search_to_show_it():
    cls = broken.VARIANTS["cartpole"]["endless"]
    assert "endless" in report_of("cartpole", cls).kinds()
    assert "endless" not in cheaply(cls).kinds()


@pytest.mark.parametrize(("game", "cls"), every("unseeded"))
def test_a_game_that_ignores_its_seed_repeats_once_seeded(game, cls):
    report = report_of(game, cls)
    assert report.fixes.reseed
    assert "The seed does not decide the episode." in found(report, "unseeded")[0]
    assert did(report, "unseeded")[0].startswith("Seeded the random generators")

    played = with_fixes(cls(), report.fixes)
    first = play_out(played, 11, moves=300)
    second = play_out(played, 11, moves=300)
    fresh = play_out(with_fixes(cls(), report.fixes), 11, moves=300)
    for other in (second, fresh):
        np.testing.assert_array_equal(first[0], other[0])
        assert first[1] == other[1]

    # Played as written, each episode is drawn at random, and two short ones can
    # match by chance: a snake that dies at once has only its food to differ by.
    # Five in a row cannot.
    bare = cls()
    plays = [play_out(bare, 11, moves=300)[0] for _ in range(5)]
    assert any(not np.array_equal(plays[0], other) for other in plays[1:])


# ------------------------------------------------------------------- what a seed shows


@pytest.mark.parametrize(("cls", "what"), [
    (InputSplits, r"input wave differed at step 5"),
    (ScoreSplits, r"the score differed at step 5"),
    (LengthSplits, r"when the episode ended differed at step \d+"),
])
def test_the_report_says_where_two_plays_of_a_seed_part(cls, what):
    report = cheaply(cls)
    assert report.fixes.reseed
    message = found(report, "unseeded")[0]
    assert re.fullmatch(rf"Seed \d+ played with the same moves went differently the second "
                        rf"time: {what}\. The seed does not decide the episode\.", message)
    assert did(report, "unseeded")[0].startswith("Seeded the random generators")


@pytest.mark.parametrize(("cls", "what"), [
    (OwnGenerator, "input wave differed at step 0."),
    (Carries, "input wave differed at step 0."),
    (FirstDiffers, "input wave differed at step 0, between the game that had played before "
                   "and a fresh copy of it."),
])
def test_a_seed_problem_seeding_cannot_cure_is_reported_and_left(cls, what):
    report = cheaply(cls)
    assert not report.fixes.reseed
    assert what in found(report, "unseeded")[0]
    assert did(report, "unseeded")[0].startswith("Seeding the generators the game holds did not")


def test_a_clean_game_never_has_its_generators_touched():
    state = random.getstate(), np.random.get_state()[1].tolist()
    report = cheaply(Plain)
    assert not report.fixes.reseed
    assert (random.getstate(), np.random.get_state()[1].tolist()) == state


# ------------------------------------------------------------------- how far to trust it


def test_rewards_nearly_nothing_ever_gets_are_flagged_and_left():
    report = cheaply(RareGoal, episodes=60)
    assert found(report, "rare") == [
        "Only 4 of 240 episodes finished with a goal other than 0, "
        "too rare for selection to work with."
    ]
    assert report.fixes.empty


def test_a_reward_nothing_ever_gets_is_flagged_and_left():
    report = cheaply(NoGoal)
    assert found(report, "rare") == [
        "All 20 episodes finished with a goal of 0, so nothing tells one player from another."
    ]
    assert report.fixes.empty


def test_too_few_steps_leaves_the_inputs_alone():
    report = cheaply(OneStep)
    assert found(report, "few_steps") == [
        "Saw only 20 steps' worth of inputs, too few to judge them by."
    ]
    assert report.fixes.empty


@pytest.mark.parametrize("cls", [NotANumber, Infinite])
def test_inputs_that_are_not_numbers_are_named_and_left(cls):
    report = cheaply(cls)
    assert found(report, "bad_values") == [
        "Some inputs gave values that are not numbers or are infinite: ratio (20 times)."
    ]
    assert report.fixes.empty
    row = report.inputs[1]
    assert row.fate == "kept" and -1.0 <= row.low < row.high <= 1.0


def test_an_input_is_not_dead_for_being_rare():
    report = cheaply(Shy, search_steps=1000)
    assert fates(report) == {"alarm": "kept", "flat": "never changes"}
    assert report.fixes.keep == (0,)
    assert did(report, "dead") == ["Dropped it. The network reads 1 inputs instead of 2."]


def test_without_any_input_left_to_read_all_of_them_stay():
    report = cheaply(Shy)
    assert found(report, "dead") == ["No input carries anything of its own in 800 steps of play."]
    assert did(report, "dead")[0].startswith("Kept them all")
    assert report.fixes.empty
    assert set(fates(report).values()) == {"kept"}


def test_an_input_seen_moving_before_a_limit_came_in_is_not_dead():
    report = cheaply(Late, ceiling=1000)
    assert report.fixes.stall_cap == 100
    assert report.fixes.step_cap == 2000
    assert report.fixes.keep is None
    assert fates(report)["late"] == "kept"
    assert "dead" not in report.kinds()
    assert did(report, "endless") == [
        "An episode now ends after 100 steps with the score standing still, and the longest "
        "wait seen was 2. Episodes now end after 2,000 steps, far past anything played, so "
        "that limit is only there as a backstop."
    ]


def test_a_game_may_only_turn_endless_once_its_inputs_are_fixed():
    report = cheaply(Flicker, episodes=20, ceiling=1000)
    assert report.fixes.scale is not None
    assert report.fixes.step_cap == 1000
    assert found(report, "endless") == [
        "With the inputs fixed, some episodes were still going after 1,000 steps, "
        "so nothing in the game is sure to end one."
    ]


def test_results_that_cannot_be_read_until_the_end_are_reported():
    report = cheaply(Unstoppable, ceiling=100)
    assert report.kinds() == {"needs_stop"}
    assert {finding.outcome for finding in report.findings} == {"left"}
    assert report.fixes.step_cap is None
    assert "cannot be read before an episode is over" in found(report, "needs_stop")[0]


def test_slow_steps_are_reported_and_nothing_changed():
    limits = LimitsConfig(step_seconds=1e-9, episode_seconds=1e-9)
    report = cheaply(Plain, limits=limits)
    assert report.kinds() == {"slow"}
    assert {finding.outcome for finding in report.findings} == {"left"}
    assert report.fixes.empty
    first, second = found(report, "slow")
    assert first.startswith("One step took ") and "limits.step_seconds" in first
    assert second.startswith("At ") and "limits.episode_seconds" in second


def test_no_time_at_all_still_gives_a_report():
    report = cheaply(Plain, seconds=0.0)
    assert report.out_of_time and report.kinds() == {"few_steps"}
    assert {finding.outcome for finding in report.findings} == {"left"}
    assert report.episodes == 0 and report.fixes.empty
    assert all(row.low is None and row.high is None for row in report.inputs)
    assert not report.players
    assert any(line.startswith("Ran out of time") for line in report.lines())
    assert "probe: ran out of time, so some of this rests on fewer episodes" in report.notes()


def test_running_out_of_time_part_way_gives_a_report():
    report = probe.run(config_of("arena", seconds=0.05))
    assert report.out_of_time
    assert report.lines()
    assert all(row.low is None or row.low <= row.high for row in report.inputs)


# ------------------------------------------------------------------- faults in a game


def test_a_fault_while_being_probed_names_the_call_step_and_seed():
    config = config_of("test_guards:Trouble", TroubleSettings(trouble="raise"))
    with pytest.raises(GameFault) as raised:
        probe.run(config)
    assert re.fullmatch(
        r"Trouble\.step raised ValueError: no move called sideways on step 3 of the episode "
        r"with seed 600000 \(test_guards\.py, line \d+\)", str(raised.value))


@pytest.mark.parametrize(("cls", "message"), [
    (BadStop, r"BadStop\.stop raised ValueError: cannot stop here"),
    (BadGoal, r"BadGoal\.goal raised ZeroDivisionError: no moves to divide by"),
])
def test_a_fault_at_the_end_of_an_episode_is_a_fault_too(cls, message):
    with pytest.raises(GameFault, match=message):
        cheaply(cls, ceiling=100)


# ------------------------------------------------------------------- in its own process


def guarded(trouble: str = "", **limits) -> probe.Report:
    return probe.run_guarded(config_of("test_guards:Trouble", TroubleSettings(trouble=trouble),
                                       search_steps=0, **limits))


def test_the_guarded_form_gives_the_same_report():
    def timeless(report):
        return dataclasses.replace(report, seconds=0.0, steps_per_second=0.0)

    config = config_of("test_guards:Trouble", TroubleSettings(), search_steps=0)
    report = probe.run_guarded(config)
    assert report.steps > 0 and report.kinds() == {"rare"}
    assert {finding.outcome for finding in report.findings} == {"left"}
    assert timeless(report) == timeless(probe.run(config))


def test_the_guarded_form_passes_on_a_fault():
    with pytest.raises(GameFault, match=r"Trouble\.step raised ValueError"):
        guarded("raise")


def test_a_step_that_never_returns_ends_the_probe(monkeypatch):
    monkeypatch.setattr(probe, "SLACK", 0.2)
    with pytest.raises(GameFault) as raised:
        guarded("hang", limits=QUICK)
    assert re.match(r"the probe went [\d.]+ s without a word, longer than limits\.step_seconds "
                    r"allows \(0\.2 s\), ", str(raised.value))


def test_a_game_that_kills_its_process_ends_the_probe():
    with pytest.raises(GameFault, match="the probe's process stopped with exit code 3"):
        guarded("exit")


def test_a_failure_outside_the_game_s_play_is_not_called_a_fault():
    with pytest.raises(WorkerError, match="the probe failed: RuntimeError: no parts"):
        probe.run_guarded(config_of(CannotBuild))


# ------------------------------------------------------------------- what it prints


def test_the_report_reads_as_text():
    report = report_of("cartpole", broken.VARIANTS["cartpole"]["dead"])
    text = "\n".join(report.lines())
    assert text.startswith(f"Played {report.episodes:,} episodes of ")
    assert "what became of it" in text
    assert "never changes" in text
    assert "dead: constant 1, constant 2, constant 3 and constant 4 never changed in" in text
    assert "  Dropped them. The network reads 4 inputs instead of 8." in text
    assert any(note.startswith("probe: constant 1, ") for note in report.notes())


def test_the_command_prints_the_report(capsys):
    assert main(["probe", "--game", games.reference(broken.VARIANTS["cartpole"]["dead"])]) == 0
    printed = capsys.readouterr().out
    assert "what became of it" in printed
    assert "never changed in" in printed


@pytest.mark.parametrize(("argv", "problem"), [
    (["--game", "snak"], "did you mean 'snake'?"),
    (["--set", "probe.episodes=0"], "probe.episodes must be at least 1, not 0"),
    (["--game", "test_guards:Trouble", "--set", "trouble=raise"], "Trouble.step raised ValueError"),
])
def test_the_command_refuses_what_it_cannot_probe(argv, problem, capsys):
    assert main(["probe", *argv]) == 1
    printed = capsys.readouterr().out
    assert "cannot probe:" in printed
    assert problem in printed


# ------------------------------------------------------------------- the whole budget


@pytest.mark.slow
@pytest.mark.parametrize(("game", "cls"), [(game, None) for game in games.names()] + [
    pytest.param(game, cls, id=f"{game}-{fault}") for game, fault, cls in EVERY
])
def test_the_default_budget_finds_only_what_was_planted(game, cls):
    fault = None if cls is None else next(f for g, f, c in EVERY if c is cls)
    report = probe.run(CONFIG.with_overrides(
        game=games.reference(cls) if cls else game, settings=None,
        probe=ProbeConfig(enabled=True, seconds=600.0)))
    expected = {fault} if fault else set()
    if (game, fault) == ("snake", "endless"):
        expected = {"endless", "unscaled"}
    assert report.kinds() - {"opposites"} == expected
    assert report.fixes.empty == (fault is None)
    assert not report.out_of_time


@pytest.mark.slow
@pytest.mark.parametrize("seed", [601_000, 602_000, 603_000, 604_000, 605_000])
def test_a_clean_arena_keeps_the_inputs_only_a_fight_moves(seed, monkeypatch):
    monkeypatch.setattr(probe, "SEED", seed)
    report = probe.run(config_of("arena"))
    assert report.fixes.empty
    assert set(fates(report).values()) == {"kept"}


# ------------------------------------------------------------------- outcomes


@pytest.mark.parametrize("fault", ["unscaled", "dead", "duplicate", "endless", "unseeded"])
def test_a_fault_the_probe_fixes_is_marked_fixed(fault):
    report = report_of("cartpole", broken.VARIANTS["cartpole"][fault])
    assert fault in report.kinds()
    assert {f.outcome for f in report.findings if f.kind == fault} == {"fixed"}


def test_a_report_from_before_findings_had_outcomes_reads_them_as_left():
    data = report_of("cartpole", broken.VARIANTS["cartpole"]["dead"]).to_dict()
    for finding in data["findings"]:
        del finding["outcome"]
    assert {f.outcome for f in probe.Report.from_dict(data).findings} == {"left"}


# ------------------------------------------------------------------- codes


def test_a_code_given_as_a_number_is_read_as_one():
    report = cheaply(Coded)
    assert report.kinds() == {"code"}
    assert report.fixes.codes == ((0.0, 1.0, 2.0, 3.0), None)
    assert report.fixes.keep is None and report.fixes.scale is None
    assert found(report, "code")[0].startswith("kind only read 0, 1, 2 and 3.")
    assert did(report, "code")[0].startswith("Gave it one input per value")
    assert fates(report) == {"kind": "a code of 4 values", "wave": "kept"}
    row = next(row for row in report.inputs if row.name == "kind")
    assert (row.after_low, row.after_high) == (0.0, 1.0)
    assert {f.outcome for f in report.findings} == {"fixed"}


@pytest.mark.parametrize("cls", [Counted, Quarters, Flag])
def test_an_amount_or_a_yes_or_no_is_not_a_code(cls):
    report = cheaply(cls)
    assert "code" not in report.kinds()
    assert report.fixes.codes is None
