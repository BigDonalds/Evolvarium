"""A game's fixes, from a probe's report to the saved model that plays by them.

What the probe decides is tested in test_probe.py. Most of the fixes here are
written by hand, so each test says which one it is about, and cart-pole plays them
because it is quick.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import random

import numpy as np
import pytest
from test_cartpole import cartpole_config
from test_guards import GENOME, TroubleSettings, context
from test_jobs import assert_same_population, cut_off, train, wait_for
from test_probe import Coded, NoGoal, Plain

from evolvarium import games, held_out, probe
from evolvarium.cli import main
from evolvarium.config import CONFIG, LimitsConfig, ProbeConfig
from evolvarium.events import EventBus, Note
from evolvarium.evolution.evaluation import EvalContext, GameFault, make_runner
from evolvarium.evolution.pool import EvaluationPool
from evolvarium.evolution.population import Population
from evolvarium.fixes import FixedGame, Fixes, seed_generators, with_fixes
from evolvarium.games import broken
from evolvarium.games.cartpole import CartPoleGame
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import checkpoint, export_model, model_file
from evolvarium.player import Player
from evolvarium.trainer import Trainer
from evolvarium.web.runs import Lab, SavedRun

LEFT = {"agent": "left"}
# Three of cart-pole's four inputs, moved and scaled.
INPUTS = Fixes(keep=(0, 2, 3), offset=(0.0, 0.1, -0.1), scale=(1.0, 2.0, 0.5))
EVERYTHING = dataclasses.replace(INPUTS, step_cap=30, stall_cap=10, reseed=True)
# Coded's kind read as a code, which is how the probe fixes it.
CODES = Fixes(codes=((0.0, 1.0, 2.0, 3.0), None))
# A budget that sees enough of a cart-pole on a sixty-step limit, in about a second.
QUICKLY = ProbeConfig(enabled=True, episodes=5, search_steps=0, seconds=600.0)
DEAD = games.reference(broken.DeadInputCartPole)
TOTAL = 4
SPLIT = 2


def cartpole_in(tmp_path, **changes):
    """Cart-pole to train on, with what it writes kept under `tmp_path`."""
    config = cartpole_config()
    return dataclasses.replace(
        config,
        checkpoint=dataclasses.replace(config.checkpoint, save_every=SPLIT),
        paths=dataclasses.replace(
            config.paths, runs_dir=tmp_path / "runs", exports_dir=tmp_path / "exports"
        ),
        **changes,
    )


def report_with(fixes: Fixes) -> probe.Report:
    """A report that found nothing but says to make `fixes`."""
    return probe.Report(game="cartpole", episodes=1, steps=1, seconds=0.0, steps_per_second=1.0,
                        ceiling=5000, inputs=[], players=[], findings=[], fixes=fixes)


def trained(config, fixes: Fixes, run_id: str = "fixed", generations: int = SPLIT) -> Trainer:
    """A finished run of `config` that plays by `fixes`."""
    return train(Trainer(config, run_id=run_id, seed=3, max_generations=generations,
                         probed=report_with(fixes)))


def newest(trainer: Trainer):
    return checkpoint.list_checkpoints(trainer.checkpoint_dir)[-1].path


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def not_again(config):
    raise AssertionError("a run that has been probed is not probed again")


def scores_of(context, genome, seeds) -> list[float]:
    runner = make_runner(dataclasses.replace(context, exploration_rate=0.0))
    return [runner.score(genome, seed, 1, False).score for seed in seeds]


def exported(population, path, genome=None):
    """`population`'s first member, written to `path`. Gives the genome and the document."""
    genome = population.members[0].genome if genome is None else genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="run-fixed", destination=path,
    )
    assert report["ready"]
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=path)
    return genome, model_file.read(path)


# ------------------------------------------------------------------ the record


def test_no_fixes_leave_the_game_itself():
    game = Plain()
    assert with_fixes(game, None) is game
    assert with_fixes(game, Fixes()) is game
    population = Population(cartpole_config(), seed=1, fixes=Fixes())
    assert population.fixes is None and type(population.game) is CartPoleGame


@pytest.mark.parametrize("fixes", [EVERYTHING, CODES])
def test_fixes_survive_being_written_and_sent(fixes):
    assert Fixes.from_dict(json.loads(json.dumps(fixes.to_dict()))) == fixes
    assert pickle.loads(pickle.dumps(fixes)) == fixes


@pytest.mark.parametrize("record", [None, {}])
def test_a_record_written_without_fixes_has_none(record):
    assert Fixes.from_dict(record) is None


def test_a_fixed_game_can_be_pickled_part_way_through():
    game = FixedGame(Plain(), Fixes(step_cap=3))
    game.reset(1)
    game.step(LEFT)
    copy = pickle.loads(pickle.dumps(game))
    copy.step(LEFT)
    copy.step(LEFT)
    assert copy.over and copy.end == "step limit" and copy.score == 3


# --------------------------------------------------------------------- inputs


@pytest.mark.parametrize(("fixes", "names", "expected"), [
    (Fixes(keep=(1,)), ["wave"], lambda turn, wave: [wave]),
    (Fixes(keep=(1, 0)), ["wave", "turn"], lambda turn, wave: [wave, turn]),
    (Fixes(offset=(1.0, -1.0), scale=(2.0, 0.5)), ["turn", "wave"],
     lambda turn, wave: [(turn + 1) * 2, (wave - 1) / 2]),
    (Fixes(keep=(1,), offset=(1.0,), scale=(2.0,)), ["wave"], lambda turn, wave: [(wave + 1) * 2]),
])
def test_a_network_reads_the_inputs_kept_moved_and_then_scaled(fixes, names, expected):
    game, raw = FixedGame(Plain(), fixes), Plain()
    for each in (game, raw):
        each.reset(3)
        each.step(LEFT)
    turn, wave = raw.observation.encode(raw.observe("agent")).tolist()
    assert list(game.observation.names) == names
    values = game.observation.encode(game.observe("agent"))
    assert values.dtype == np.float32
    assert values.tolist() == pytest.approx(expected(turn, wave), rel=1e-6)


def kind_and_wave(fixes: Fixes, seed: int) -> list[tuple[float, list[float]]]:
    """Coded's kind at each step, and what the network reads then with `fixes`."""
    game, raw = FixedGame(Coded(), fixes), Coded()
    seen = []
    for each in (game, raw):
        each.reset(seed)
    while not raw.over:
        kind = raw.observation.encode(raw.observe("agent"))[0]
        seen.append((float(kind), game.observation.encode(game.observe("agent")).tolist()))
        game.step(LEFT)
        raw.step(LEFT)
    return seen


def test_a_code_gives_the_network_one_input_per_value():
    assert list(FixedGame(Coded(), CODES).observation.names) == [
        "kind=0", "kind=1", "kind=2", "kind=3", "wave"]
    seen = kind_and_wave(CODES, 3)
    for kind, values in seen:
        assert values[:4] == [1.0 if k == kind else 0.0 for k in range(4)]
    assert {kind for kind, _ in seen} == {0.0, 1.0, 2.0, 3.0}


def test_a_value_play_never_showed_reads_as_the_nearest_the_earlier_on_a_tie():
    fixes = Fixes(codes=((0.0, 2.0, 5.0), None))
    nearest = {0.0: 0, 1.0: 0, 2.0: 1, 3.0: 1}
    for kind, values in kind_and_wave(fixes, 3):
        assert values[:3] == [1.0 if k == nearest[kind] else 0.0 for k in range(3)]


def test_a_code_follows_the_inputs_kept_and_scaled():
    fixes = Fixes(keep=(1, 0), offset=(1.0, 0.0), scale=(2.0, 1.0),
                  codes=(None, (0.0, 1.0, 2.0, 3.0)))
    assert list(FixedGame(Coded(), fixes).observation.names) == [
        "wave", "kind=0", "kind=1", "kind=2", "kind=3"]
    game, raw = FixedGame(Coded(), fixes), Coded()
    for each in (game, raw):
        each.reset(4)
    kind, wave = raw.observation.encode(raw.observe("agent")).tolist()
    values = game.observation.encode(game.observe("agent")).tolist()
    assert values[0] == pytest.approx((wave + 1) * 2, rel=1e-6)
    assert values[1:] == [1.0 if k == kind else 0.0 for k in range(4)]


def test_limits_alone_leave_the_inputs_to_the_game():
    plain = Plain()
    assert FixedGame(plain, Fixes(step_cap=3)).observation is plain.observation


def test_a_network_is_as_wide_as_the_inputs_kept():
    population = Population(cartpole_config(), seed=1, fixes=INPUTS)
    assert population.architecture.input_size == len(INPUTS.keep)


# --------------------------------------------------------------------- limits


@pytest.mark.parametrize("name", games.names())
def test_a_cap_ends_an_episode_of_every_game_with_its_results_readable(name):
    game = FixedGame(games.create(name), Fixes(step_cap=3))
    idle = game.actions.decode(np.zeros(game.actions.size, dtype=np.float32))
    for seed in (1, 2):
        game.reset(seed)
        for _ in range(3):
            assert not game.over
            game.step({agent: idle for agent in game.agents})
        assert game.over and game.base.over and game.steps == 3
        assert isinstance(game.end, str)
        game.goal()
        game.signals()


def test_a_game_that_needs_no_stop_is_ended_by_the_cap():
    game = FixedGame(Plain(), Fixes(step_cap=3))
    game.reset(1)
    while not game.over:
        game.step(LEFT)
    assert game.steps == 3 and game.end == "step limit"
    game.reset(2)
    assert not game.over and game.steps == 0


def test_a_cap_the_game_ends_before_changes_nothing():
    game = FixedGame(Plain(), Fixes(step_cap=100))
    game.reset(1)
    while not game.over:
        game.step(LEFT)
    assert game.steps == Plain.length and game.end == "done"


def test_an_episode_ends_once_its_score_has_stood_still_long_enough():
    game = FixedGame(NoGoal(), Fixes(stall_cap=5))
    game.reset(1)
    while not game.over:
        game.step(LEFT)
    # The first step gives the score to compare the others with.
    assert game.steps == 6 and game.end == "stalled"


def test_a_score_still_moving_never_stalls():
    game = FixedGame(Plain(), Fixes(stall_cap=2))
    game.reset(1)
    while not game.over:
        game.step(LEFT)
    assert game.steps == Plain.length and game.end == "done"


class Told(NoGoal):
    stalled = None

    def stop(self, stalled: bool) -> None:
        self.stalled = stalled


@pytest.mark.parametrize(("fixes", "stalled"), [
    (Fixes(stall_cap=2), True),
    (Fixes(step_cap=2), False),
])
def test_a_game_is_told_which_limit_ended_its_episode(fixes, stalled):
    game = FixedGame(Told(), fixes)
    game.reset(1)
    while not game.over:
        game.step(LEFT)
    assert game.base.stalled is stalled


# ----------------------------------------------------------------- generators


class Slotted:
    __slots__ = ("chance",)

    def __init__(self):
        self.chance = np.random.RandomState()


class Hoard(Plain):
    """Keeps a generator everywhere a game might."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.named = {"dice": random.Random()}
        self.listed = [np.random.default_rng()]
        self.slotted = Slotted()

    def draws(self) -> tuple[float, float, float]:
        return (self.named["dice"].random(), float(self.listed[0].random()),
                float(self.slotted.chance.random_sample()))


def test_generators_are_seeded_wherever_a_game_keeps_them():
    first, second = Hoard(), Hoard()
    seed_generators(first, 7)
    seed_generators(second, 7)
    drawn = first.draws()
    assert drawn == second.draws()
    seed_generators(second, 8)
    # All three differ, so none of them was missed.
    assert all(a != b for a, b in zip(drawn, second.draws(), strict=True))


def test_a_generator_s_seed_does_not_depend_on_what_else_the_game_holds():
    plain, more = Hoard(), Hoard()
    more.another = random.Random()
    more.named["coin"] = np.random.default_rng()
    seed_generators(plain, 7)
    seed_generators(more, 7)
    assert plain.draws() == more.draws()


def test_two_generators_in_one_game_draw_different_numbers():
    game = Plain()
    game.a, game.b = random.Random(), random.Random()
    seed_generators(game, 7)
    assert game.a.random() != game.b.random()


class SharedChance(Plain):
    """Draws from the generators everything shares."""

    def observe(self, agent: str):
        return [random.random(), float(np.random.random())]


def test_reseeding_repeats_a_game_that_draws_from_the_shared_generators():
    game = FixedGame(SharedChance(), Fixes(reseed=True))
    game.reset(5)
    first = game.observation.encode(game.observe("agent"))
    game.reset(5)
    assert np.array_equal(first, game.observation.encode(game.observe("agent")))
    game.reset(6)
    assert not np.array_equal(first, game.observation.encode(game.observe("agent")))


# --------------------------------------------------------------------- faults


@pytest.mark.parametrize("fixes", [
    Fixes(step_cap=100),
    Fixes(stall_cap=100),
    Fixes(keep=(0,), offset=(0.0,), scale=(2.0,)),
])
def test_a_fault_is_blamed_on_the_game_and_not_on_the_fixes(fixes):
    runner = make_runner(dataclasses.replace(context(trouble="raise"), fixes=fixes))
    with pytest.raises(
        GameFault,
        match=r"^Trouble\.step raised ValueError: no move called sideways on step 3 of the "
              r"episode with seed 7 \(test_guards\.py, line \d+\)$",
    ):
        runner.score(GENOME, 7, 2, False)


class BadScore(Plain):
    """Fails on reading the score, which a stall limit does on every step."""

    @property
    def score(self) -> int:
        raise ValueError("no score to read")


class BadOver(Plain):
    @property
    def over(self) -> bool:
        raise ValueError("cannot tell")


class CannotStop(NoGoal):
    def stop(self, stalled: bool) -> None:
        raise ValueError("no way to stop")


@pytest.mark.parametrize(("cls", "fixes", "call"), [
    (BadScore, Fixes(stall_cap=5), "BadScore.score"),
    (BadOver, Fixes(step_cap=5), "BadOver.over"),
    (CannotStop, Fixes(stall_cap=2), "CannotStop.stop"),
    (CannotStop, Fixes(step_cap=2), "CannotStop.stop"),
])
def test_a_fault_names_the_game_s_call_the_fixes_made_for_it(cls, fixes, call):
    architecture = Architecture(2, (4,), ("tanh",), Plain.actions.size)
    runner = make_runner(EvalContext(
        game=games.reference(cls), settings=None, architecture=architecture.to_dict(),
        exploration_rate=0.0, replay_steps=0, limits=LimitsConfig(), fixes=fixes,
    ))
    genome = np.zeros(architecture.genome_size, dtype=np.float32)
    with pytest.raises(GameFault) as raised:
        runner.score(genome, 7, 1, False)
    assert str(raised.value).startswith(f"{call} raised ValueError")
    assert "test_fixes.py, line" in str(raised.value)


def test_a_game_that_fails_the_probe_stops_the_run_before_it_has_a_directory(tmp_path):
    config = cartpole_in(tmp_path, game="test_guards:Trouble",
                         settings=TroubleSettings(trouble="raise"), probe=QUICKLY)
    with pytest.raises(GameFault, match=r"Trouble\.step raised"):
        Trainer(config, seed=1, run_id="never")
    assert not (tmp_path / "runs").exists()


def test_train_says_why_a_game_that_fails_the_probe_cannot_start(tmp_path, capsys):
    runs = tmp_path / "runs"
    code = main(["train", "--headless", "--game", "test_guards:Trouble", "--set",
                 "trouble=raise", "probe.enabled=true", "probe.episodes=2",
                 "probe.search_steps=0", f"paths.runs_dir={runs}"])
    assert code == 1
    out = capsys.readouterr().out
    assert "cannot train:" in out and "Trouble.step raised" in out
    assert not runs.exists()


def test_the_dashboard_says_why_a_game_that_fails_the_probe_cannot_start(tmp_path):
    lab = Lab(cartpole_in(tmp_path), can_start=True)
    try:
        reply = lab.start("test_guards:Trouble", {
            "trouble": "raise", "probe.enabled": "true", "probe.episodes": "2",
            "probe.search_steps": "0",
        })
        assert "Trouble.step raised" in reply["problems"][0]
        assert lab.live == []
    finally:
        lab.close()
    assert not (tmp_path / "runs").exists()


# ------------------------------------------------------------------- training


def test_a_fixed_game_scores_the_same_in_worker_processes():
    config = cartpole_config()
    serial = Population(config, seed=32, fixes=EVERYTHING)
    parallel = Population(config, seed=32, fixes=EVERYTHING)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(2):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)
            assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
            assert a.ends == b.ends


def test_a_game_with_nothing_wrong_trains_the_same_with_the_probe_on(tmp_path):
    off = train(Trainer(cartpole_in(tmp_path), run_id="off", seed=5, max_generations=TOTAL))
    on = train(Trainer(cartpole_in(tmp_path, probe=QUICKLY), run_id="on", seed=5,
                       max_generations=TOTAL))
    assert on.population.fixes is None
    assert type(on.population.game) is CartPoleGame
    assert_same_population(off.population, on.population)
    assert read_json(on.run_dir / "run.json")["fixes"] is None
    assert not (off.run_dir / "probe.json").exists()
    assert probe.Report.from_dict(read_json(on.run_dir / "probe.json")).fixes.empty


def test_a_run_trains_on_what_the_probe_fixed_and_records_it(tmp_path):
    config = cartpole_in(tmp_path, game=DEAD, probe=QUICKLY)
    trainer = train(Trainer(config, run_id="fixed", seed=5, max_generations=SPLIT))
    population = trainer.population
    assert population.architecture.input_size == len(population.fixes.keep)
    assert population.architecture.input_size < len(population.game.base.observation.names)

    report = probe.Report.from_dict(read_json(trainer.run_dir / "probe.json"))
    assert "dead" in report.kinds()
    assert report.fixes == population.fixes
    assert Fixes.from_dict(read_json(trainer.run_dir / "run.json")["fixes"]) == population.fixes
    assert Fixes.from_dict(checkpoint.read_meta(newest(trainer))["fixes"]) == population.fixes


def test_what_the_probe_found_reaches_a_view_of_the_run(tmp_path):
    events, heard = EventBus(), []
    events.subscribe(heard.append, Note)
    config = cartpole_in(tmp_path, game=DEAD, probe=QUICKLY)
    train(Trainer(config, run_id="notes", seed=3, events=events, max_generations=1))
    notes = [event.message for event in heard]
    assert any(note.startswith("probe: played ") for note in notes)
    assert any(note.startswith("probe: ") and "never change" in note for note in notes)


def test_a_report_already_made_is_used_and_not_made_again(tmp_path, monkeypatch):
    config = cartpole_in(tmp_path, game=DEAD, probe=QUICKLY)
    report = probe.run(config)
    monkeypatch.setattr(probe, "run_guarded", not_again)
    trainer = Trainer(config, run_id="ready", seed=3, max_generations=1, probed=report)
    assert trainer.population.fixes == report.fixes
    trainer.stop()


def test_the_dashboard_probes_the_runs_it_starts(tmp_path):
    lab = Lab(cartpole_in(tmp_path), can_start=True)
    try:
        reply = lab.start(DEAD, {"probe.enabled": "true", "probe.episodes": "5",
                                 "probe.search_steps": "0"}, generations=1)
        assert reply["ok"], reply
        wait_for(lambda: not lab.live)
    finally:
        lab.close()
    run_dir = tmp_path / "runs" / reply["run"]
    assert Fixes.from_dict(read_json(run_dir / "run.json")["fixes"]).keep is not None
    assert "dead" in probe.Report.from_dict(read_json(run_dir / "probe.json")).kinds()


# ----------------------------------------------------------- stopped and going on


def test_a_checkpoint_keeps_the_fixes_and_carries_on_the_same(tmp_path):
    config = cartpole_config()
    first = Population(config, seed=9, fixes=EVERYTHING)
    first.run_generation()
    path = tmp_path / "saved.npz"
    checkpoint.save(first, path)
    assert Fixes.from_dict(checkpoint.read_meta(path)["fixes"]) == EVERYTHING

    second = checkpoint.load(path, config)
    assert second.fixes == EVERYTHING
    assert second.architecture.input_size == len(EVERYTHING.keep)
    first.run_generation()
    second.run_generation()
    assert_same_population(first, second)


def test_a_run_resumed_elsewhere_keeps_its_fixes_and_report(tmp_path, monkeypatch):
    config = cartpole_in(tmp_path, game=DEAD, probe=QUICKLY)
    straight = train(Trainer(config, run_id="straight", seed=5, max_generations=TOTAL))
    first = train(Trainer(config, run_id="first", seed=5, max_generations=SPLIT))
    # It plays the game its checkpoint was trained on, so there is nothing to look at.
    monkeypatch.setattr(probe, "run_guarded", not_again)

    resumed = Trainer(config, run_id="resumed", resume_from=newest(first), max_generations=TOTAL)
    assert resumed.population.fixes == first.population.fixes
    recorded = read_json(resumed.run_dir / "run.json")["fixes"]
    assert Fixes.from_dict(recorded) == resumed.population.fixes
    assert read_json(resumed.run_dir / "probe.json") == read_json(first.run_dir / "probe.json")
    train(resumed)
    assert_same_population(straight.population, resumed.population)


@pytest.mark.parametrize("saved", [0, SPLIT])
def test_a_run_cut_off_carries_on_with_the_fixes_it_started_with(tmp_path, monkeypatch, saved):
    config = cartpole_in(tmp_path, game=DEAD, probe=QUICKLY)
    straight = train(Trainer(config, run_id="straight", seed=5, max_generations=TOTAL))
    run_dir = cut_off(config, "cut", 5, TOTAL, reached=SPLIT + 1, saved=saved)
    report = read_json(run_dir / "probe.json")
    monkeypatch.setattr(probe, "run_guarded", not_again)

    found = checkpoint.list_checkpoints(run_dir / "checkpoints")
    carried = Trainer(config, run_id="cut", seed=5, max_generations=TOTAL, carry_on=True,
                      resume_from=found[-1].path if found else None)
    assert carried.population.fixes == straight.population.fixes
    train(carried)
    assert_same_population(straight.population, carried.population)
    assert read_json(run_dir / "probe.json") == report


# --------------------------------------------------------------- saved models


def test_a_model_of_a_fixed_game_plays_as_it_trained(tmp_path):
    path = tmp_path / "fixed.json"
    population = Population(cartpole_config(), seed=33, fixes=INPUTS)
    population.run_generation()
    genome, document = exported(population, path)
    assert document["version"] == 3
    assert document["inputs"]["names"] == list(CartPoleGame.observation.names)
    assert document["inputs"]["used"] == list(INPUTS.keep)
    # Written as the singles the game's inputs were worked out in, as the weights are.
    assert document["inputs"]["offsets"] == np.array(INPUTS.offset, dtype=np.float32).tolist()
    assert document["inputs"]["scales"] == np.array(INPUTS.scale, dtype=np.float32).tolist()
    assert Fixes.from_dict(document["fixes"]) == INPUTS

    # The player is given every input the game has and does the sums itself.
    player = Player.load(path)
    assert "the network reads 3 of them" in player.describe()
    network = PolicyNetwork(population.architecture).load(genome)
    game = population.game
    for seed in (5, 6):
        game.reset(seed)
        while not game.over:
            everything = game.base.observation.encode(game.base.observe("agent"))
            kept = game.observation.encode(game.observe("agent"))
            force = game.actions.decode(network.forward(kept))
            assert player.action(everything.tolist()) == pytest.approx(force, rel=1e-5, abs=1e-5)
            game.step({"agent": force})

    champion = held_out.load_champion(path)
    assert champion.context.fixes == INPUTS
    assert champion.context.architecture["input_size"] == len(INPUTS.keep)
    held_out.score([champion], [40, 41], workers=1)
    assert champion.scores.tolist() == scores_of(population.context, genome, (40, 41))


def test_a_model_that_reads_codes_plays_as_it_trained(tmp_path):
    path = tmp_path / "coded.json"
    config = dataclasses.replace(cartpole_config(), game=games.reference(Coded), settings=None)
    population = Population(config, seed=12, fixes=CODES)
    population.run_generation()
    genome, document = exported(population, path)
    assert document["version"] == 4
    assert document["inputs"]["names"] == ["kind", "wave"]
    assert document["inputs"]["codes"] == [[0.0, 1.0, 2.0, 3.0], None]
    assert population.architecture.input_size == 5

    player = Player.load(path)
    assert "some as codes" in player.describe()
    network = PolicyNetwork(population.architecture).load(genome)
    game = population.game
    for seed in (5, 6):
        game.reset(seed)
        while not game.over:
            everything = game.base.observation.encode(game.base.observe("agent"))
            move = game.actions.decode(network.forward(game.observation.encode(
                game.observe("agent"))))
            assert player.action(everything.tolist()) == move
            game.step({"agent": move})
    # A value the game never gave is read as the nearest, as the engine reads it.
    assert player.outputs([7.0, 0.5]) == player.outputs([3.0, 0.5])


@pytest.mark.parametrize(("codes", "message"), [
    ([[0.0, 1.0, 2.0, 3.0]], "inputs.codes needs 2 entries"),
    ([[0.0], None], "a list of at least two numbers"),
    ([[0.0, "one"], None], "is not a finite number"),
    (None, "inputs.codes needs 2 entries"),
])
def test_a_model_s_codes_are_checked_when_it_is_read(tmp_path, codes, message):
    config = dataclasses.replace(cartpole_config(), game=games.reference(Coded), settings=None)
    population = Population(config, seed=12, fixes=CODES)
    _, document = exported(population, tmp_path / "coded.json")
    document["inputs"]["codes"] = codes
    with pytest.raises(model_file.ModelError, match=message):
        Player(document)


def test_limits_alone_keep_a_model_s_inputs_and_are_scored_by_it(tmp_path):
    fixes = Fixes(step_cap=3)
    path = tmp_path / "capped.json"
    population = Population(cartpole_config(), seed=33, fixes=fixes)
    population.run_generation()
    genome, document = exported(population, path)
    assert document["version"] == 2
    assert "used" not in document["inputs"]
    assert Fixes.from_dict(document["fixes"]) == fixes

    champion = held_out.load_champion(path)
    held_out.score([champion], [40, 41], workers=1)
    assert champion.scores.tolist() == scores_of(population.context, genome, (40, 41))
    uncapped = dataclasses.replace(population.context, fixes=None)
    assert champion.scores.tolist() != scores_of(uncapped, genome, (40, 41))


def test_a_game_with_nothing_fixed_saves_a_model_with_no_fixes(tmp_path):
    population = Population(cartpole_config(), seed=33)
    population.run_generation()
    _, document = exported(population, tmp_path / "clean.json")
    assert document["version"] == 2 and "fixes" not in document


def test_a_checkpoint_is_scored_under_its_fixes(tmp_path):
    trainer = trained(cartpole_in(tmp_path), EVERYTHING)
    path = newest(trainer)
    assert held_out.run_context(path).fixes == EVERYTHING
    champion = held_out.load_champion(path)
    held_out.score([champion], [40, 41], workers=1)
    assert champion.scores.tolist() == scores_of(
        trainer.population.context, champion.genome, (40, 41)
    )


def test_the_export_command_writes_a_run_s_fixes_into_its_model(tmp_path, monkeypatch):
    import evolvarium.cli as cli

    config = cartpole_in(tmp_path)
    trainer = trained(config, INPUTS)
    monkeypatch.setattr(cli, "CONFIG", config)
    assert main(["export", "--checkpoint", str(newest(trainer)), "--name", "model"]) == 0
    document = model_file.read(config.paths.exports_dir / "model.json")
    assert document["version"] == 3
    assert Fixes.from_dict(document["fixes"]) == INPUTS


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_the_dashboard_plays_a_saved_run_by_its_fixes(tmp_path, with_checkpoint):
    config = cartpole_in(tmp_path)
    trainer = trained(config, INPUTS)
    if not with_checkpoint:
        for found in checkpoint.list_checkpoints(trainer.checkpoint_dir):
            found.path.unlink()

    saved = SavedRun(trainer.run_dir, config.history)
    assert saved.problem is None
    assert saved.fixes == INPUTS
    assert saved.game.base.name == "cartpole"
    assert saved.architecture.input_size == len(INPUTS.keep)
    assert saved.context().fixes == INPUTS
    genome = trainer.population.members[0].genome
    assert make_runner(saved.context()).score(genome, 40, 1, True).score is not None


# ---------------------------------------------------------------- comparing runs


def manifest_of(**probing) -> dict:
    """What a run's manifest records of its configuration."""
    config = dataclasses.replace(games.resolve(CONFIG), probe=ProbeConfig(**probing))
    return json.loads(json.dumps(dataclasses.asdict(config), default=str))


def test_runs_with_the_probe_on_are_compared_apart_from_runs_without_it():
    off = held_out.search_settings(manifest_of())
    on = held_out.search_settings(manifest_of(enabled=True))
    assert on != off
    assert held_out.describe(off) == "defaults"
    assert held_out.describe(on) == "probe.enabled=true"
    changed = held_out.search_settings(manifest_of(enabled=True, episodes=5))
    assert held_out.describe(changed) == "probe.enabled=true, probe.episodes=5"


def test_the_probe_s_settings_mean_nothing_while_it_is_off():
    off = held_out.search_settings(manifest_of())
    assert held_out.search_settings(manifest_of(episodes=5)) == off


def test_a_run_from_before_the_probe_is_compared_with_runs_that_have_it_off():
    older = manifest_of()
    del older["probe"]
    assert held_out.search_settings(older) == held_out.search_settings(manifest_of())
