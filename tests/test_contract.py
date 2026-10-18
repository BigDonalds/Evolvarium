"""The game contract, checked with a game that is nothing like Snake.

`Herd` has two agents sharing an episode, an action made of a choice and a
number, an input that is a category, and a score where lower is better. Every
path the engine takes for Snake has a second branch for at least one of those,
and this module is where those branches run.

The registry imports it as "test_contract:Herd". Worker processes find it the
same way, since they start with the parent's import path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pytest

from evolvarium import games, held_out
from evolvarium.config import CONFIG
from evolvarium.contract import (
    Actions,
    Category,
    Choice,
    Game,
    Label,
    Layer,
    Number,
    Observation,
    Range,
    Scene,
    load_settings,
    override,
    settings_dict,
)
from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.evolution.population import Population
from evolvarium.games.snake.settings import SnakeSettings
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import checkpoint, export_model, model_file
from evolvarium.player import Player

HERD = "test_contract:Herd"
MOVES = {"left": -1.0, "stay": 0.0, "right": 1.0}
CLOSE = 0.25


@dataclass(frozen=True)
class HerdSettings:
    start: float = 4.0
    max_steps: int = 20


class Herd(Game):
    """Two agents on a line, scored on how far from the middle they finish."""

    name = "herd"
    Settings = HerdSettings
    direction = "lower"
    agents = ("a", "b")
    observation = Observation([
        Number("position"),
        Number("gap"),
        Category("side", ("left", "centre", "right")),
    ])
    actions = Actions(Choice("move", ("left", "stay", "right")), Range("stride", 0.5, 1.5))

    def reset(self, seed):
        start = self.settings.start
        rng = np.random.default_rng(seed)
        self.positions = dict(zip(self.agents, rng.uniform(-start, start, 2).tolist(),
                                  strict=True))
        self._steps = 0
        self._end = None

    def observe(self, agent):
        position = self.positions[agent]
        other = next(p for name, p in self.positions.items() if name != agent)
        if abs(position) < CLOSE:
            side = "centre"
        else:
            side = "left" if position < 0 else "right"
        return {"side": side, "gap": other - position, "position": position}

    def step(self, actions):
        assert set(actions) == set(self.agents), actions
        for agent, action in actions.items():
            self.positions[agent] += MOVES[action["move"]] * action["stride"]
        self._steps += 1
        if all(abs(p) < CLOSE for p in self.positions.values()):
            self._end = "gathered"
        elif self._steps >= self.settings.max_steps:
            self._end = "timeout"

    @property
    def over(self):
        return self._end is not None

    @property
    def end(self):
        return self._end

    @property
    def score(self):
        return sum(abs(p) for p in self.positions.values())

    @property
    def steps(self):
        return self._steps

    def scene(self):
        start = self.settings.start
        return Scene(2 * start + 2, 2, [Layer("herd", "dot", "agent", size=0.2)],
                     [Label("spread", 2)], origin=(-start - 1, -1), y_up=True, unit=0.01)

    def draw(self):
        return {"herd": [(p, 0.0) for p in self.positions.values()], "spread": self.score}


@dataclass(frozen=True)
class Lanes:
    widths: tuple[float, ...] = (1.0, 2.0)


@dataclass(frozen=True)
class Track:
    lanes: Lanes = field(default_factory=Lanes)
    laps: int = 3


class Sideways(Herd):
    direction = "sideways"


class NotAGame:
    pass


def herd_config(**evolution):
    return CONFIG.with_overrides(
        game=HERD,
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=12, evaluation_episodes=3, elite_count=2,
            immigrant_count=1, workers=1, **evolution,
        ),
    )


# ------------------------------------------------------------------ inputs


def test_an_observation_encodes_numbers_and_categories_in_order():
    observation = Herd.observation
    assert observation.names == ("position", "gap", "side=left", "side=centre", "side=right")
    assert observation.size == 5

    from_list = observation.encode([1.5, -2.0, "right"])
    from_mapping = observation.encode({"side": "right", "gap": -2.0, "position": 1.5})
    assert from_list.dtype == np.float32
    assert from_list.tolist() == [1.5, -2.0, 0.0, 0.0, 1.0]
    assert np.array_equal(from_list, from_mapping)


def test_a_plain_observation_takes_an_array_as_it_is():
    observation = Observation(["x", "y"])
    assert observation.encode(np.array([3.0, 4.0])).tolist() == [3.0, 4.0]
    assert observation.encode({"y": 4.0, "x": 3.0}).tolist() == [3.0, 4.0]


@pytest.mark.parametrize(("values", "message"), [
    ({"position": 0.0, "gap": 1.0}, "missing"),
    ([0.0, 1.0], "2 values"),
    ([0.0, 1.0, "up"], "one of"),
])
def test_an_observation_says_what_is_wrong_with_it(values, message):
    with pytest.raises(ValueError, match=message):
        Herd.observation.encode(values)


def test_a_plain_observation_rejects_the_wrong_count():
    with pytest.raises(ValueError, match="declares 2 inputs"):
        Observation(["x", "y"]).encode([1.0, 2.0, 3.0])


@pytest.mark.parametrize("inputs", [
    [],
    ["x", "x"],
    ["side=left", Category("side", ("left", "right"))],
])
def test_observation_inputs_need_distinct_names(inputs):
    with pytest.raises(ValueError):
        Observation(inputs)


def test_a_category_needs_two_distinct_values():
    with pytest.raises(ValueError, match="two values"):
        Category("side", ("left",))
    with pytest.raises(ValueError, match="twice"):
        Category("side", ("left", "left"))


# ------------------------------------------------------------------ actions


def test_one_choice_delivers_the_option_itself():
    actions = Actions(Choice("turn", ("left", "straight", "right")))
    assert actions.names == ("left", "straight", "right")
    assert actions.decode(np.array([0.1, 0.9, 0.3])) == "straight"


def test_a_choice_given_a_count_delivers_numbers():
    choice = Choice("lane", 4)
    assert choice.options == (0, 1, 2, 3)
    assert choice.labels == ("0", "1", "2", "3")
    assert choice.decode(np.array([0.0, 0.0, 0.0, 1.0])) == 3


def test_several_parts_deliver_a_dict_by_part_name():
    actions = Herd.actions
    assert actions.size == 4
    assert actions.names == ("move=left", "move=stay", "move=right", "stride")
    assert actions.decode(np.array([0.0, 0.0, 2.0, 0.0])) == {"move": "right", "stride": 1.0}


def test_a_range_stays_inside_its_bounds():
    stride = Range("stride", 0.5, 1.5)
    assert stride.decode(np.array([0.0])) == pytest.approx(1.0)
    assert stride.decode(np.array([1e6])) == pytest.approx(1.5)
    assert stride.decode(np.array([-1e6])) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="low below high"):
        Range("stride", 1.0, 1.0)


def test_random_actions_are_valid_actions():
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = Herd.actions.explore(rng)
        assert action["move"] in MOVES
        assert 0.5 <= action["stride"] <= 1.5


def test_action_parts_need_distinct_names_and_options():
    with pytest.raises(ValueError, match="distinct"):
        Actions(Choice("move", 2), Range("move"))
    with pytest.raises(ValueError, match="twice"):
        Choice("move", ("up", "up"))
    with pytest.raises(ValueError, match="two options"):
        Choice("move", 1)


# ----------------------------------------------------------------- settings


def test_settings_rebuild_from_plain_data():
    track = load_settings(Track, {"lanes": {"widths": [1.0, 1.5, 2.0]}, "laps": 5})
    assert track == Track(lanes=Lanes(widths=(1.0, 1.5, 2.0)), laps=5)
    assert load_settings(Track, {"laps": 4}) == Track(laps=4)
    assert load_settings(Track, None) == Track()

    snake = override(SnakeSettings(), "board.width", 9)
    assert load_settings(SnakeSettings, settings_dict(snake)) == snake


def test_unknown_settings_are_refused_or_dropped():
    with pytest.raises(ValueError, match="no setting 'colour'"):
        load_settings(HerdSettings, {"colour": "red"})
    assert load_settings(HerdSettings, {"colour": "red", "start": 2.0}, strict=False) == (
        HerdSettings(start=2.0)
    )


def test_override_replaces_one_nested_setting():
    settings = override(SnakeSettings(), "board.width", 9)
    assert settings.board.width == 9
    assert settings.board.height == SnakeSettings().board.height
    assert settings.episode == SnakeSettings().episode


# ----------------------------------------------------------------- registry


def test_the_registry_finds_built_in_and_outside_games():
    assert games.get("snake").name == "snake"
    assert games.get(HERD) is Herd
    assert games.reference(Herd) == HERD
    assert games.reference(games.create("snake")) == "snake"
    assert games.create(HERD, HerdSettings(start=1.0)).settings.start == 1.0
    assert games.settings_for(herd_config()) == HerdSettings()


@pytest.mark.parametrize(("name", "message"), [
    ("tetris", "built in: snake"),
    ("no_such_module:Game", "cannot import"),
    ("test_contract:NotAGame", "not a Game subclass"),
    ("test_contract:Missing", "not a Game subclass"),
    ("test_contract:Sideways", "direction 'sideways'"),
])
def test_the_registry_says_why_a_name_is_not_a_game(name, message):
    with pytest.raises(games.UnknownGame, match=message):
        games.get(name)


# ----------------------------------------------------------------- engine


def herd_runner(rate: float = 0.0):
    game = Herd()
    architecture = Architecture(game.observation.size, (6,), ("tanh",), game.actions.size)
    ctx = EvalContext(game=HERD, settings=HerdSettings(), architecture=architecture.to_dict(),
                      exploration_rate=rate, replay_steps=0)
    return make_runner(ctx), architecture


def test_a_lower_score_is_the_better_one():
    runner, architecture = herd_runner()
    rng = np.random.default_rng(3)
    genome = rng.normal(size=architecture.genome_size).astype(np.float32)

    outcomes = [runner.play(genome, 40 + trial, False) for trial in range(4)]
    result = runner.score(genome, 40, 4, False)

    scores = [o.score for o in outcomes]
    assert len(set(scores)) > 1, "the trials should not all score the same"
    assert result.best_episode_score == min(scores)
    assert result.end == max(outcomes, key=lambda o: o.score).end
    assert result.fitness == pytest.approx(-np.mean(scores))
    assert result.terms == {"score": pytest.approx(-np.mean(scores))}
    assert all(o.end in ("gathered", "timeout") for o in outcomes)


def test_every_agent_acts_every_step_with_exploration_on():
    runner, architecture = herd_runner(rate=0.5)
    genome = np.zeros(architecture.genome_size, dtype=np.float32)
    first = runner.play(genome, 7, False)
    second = runner.play(genome, 7, False)
    assert (first.score, first.steps, first.end) == (second.score, second.steps, second.end)


def test_a_population_improves_on_a_lower_is_better_game():
    population = Population(herd_config(), seed=21)
    assert not population.higher
    first = population.run_generation()
    assert first.record
    assert population.best_ever_score == first.best.best_episode_score
    assert first.max_episode_score == min(m.best_episode_score for m in first.evaluated)

    for _ in range(5):
        report = population.run_generation()
        if report.record:
            assert population.best_ever_score == report.best.best_episode_score
        assert population.best_ever_score <= first.best.best_episode_score
        assert set(report.ends) <= {"gathered", "timeout"}
        assert sum(report.ends.values()) == len(report.evaluated)
    assert population.best_ever_score < first.best.best_episode_score


def test_a_checkpoint_of_another_game_round_trips(tmp_path):
    config = herd_config()
    population = Population(config, seed=22)
    population.run_generation()
    path = tmp_path / "herd.npz"
    info = checkpoint.save(population, path)
    assert info.game == HERD
    assert info.encoding == "herd"

    restored = checkpoint.load(path, config)
    assert restored.best_ever_score == population.best_ever_score
    a, b = population.run_generation(), restored.run_generation()
    assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
    assert population.best_ever_score == restored.best_ever_score

    snake = CONFIG.with_overrides(evolution=config.evolution)
    with pytest.raises(checkpoint.CheckpointMismatch, match="trained on 'test_contract:Herd'"):
        checkpoint.load(path, snake)


def export_herd(path, seed):
    population = Population(herd_config(), seed=seed)
    population.run_generation()
    genome = population.members[0].genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="run-herd", destination=path,
    )
    assert report["ready"]
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=path)
    return population, genome


def test_a_choice_and_a_range_export_and_play_as_trained(tmp_path):
    path = tmp_path / "herd.json"
    population, genome = export_herd(path, seed=24)
    document = model_file.read(path)
    assert document["actions"]["parts"] == Herd.actions.describe()
    assert document["game"]["name"] == HERD

    player = Player.load(path)
    network = PolicyNetwork(population.architecture).load(genome)
    game = Herd()
    for seed in (5, 6):
        game.reset(seed)
        while not game.over:
            actions = {}
            for agent in game.agents:
                inputs = game.observation.encode(game.observe(agent))
                engine = game.actions.decode(network.forward(inputs))
                mine = player.action(inputs.tolist())
                assert mine["move"] == engine["move"]
                assert mine["stride"] == pytest.approx(engine["stride"], rel=1e-5)
                actions[agent] = engine
            game.step(actions)

    champion = held_out.load_champion(path)
    assert champion.context.game == HERD
    assert champion.context.settings == HerdSettings()
    assert np.array_equal(champion.genome, genome)
    held_out.score([champion], [40, 41], workers=1)
    runner = make_runner(dataclasses.replace(population.context, exploration_rate=0.0))
    assert champion.scores.tolist() == [runner.score(genome, s, 1, False).score for s in (40, 41)]


def test_the_engine_refuses_a_model_whose_actions_differ_from_its_game(tmp_path):
    path = tmp_path / "herd.json"
    export_herd(path, seed=25)
    document = model_file.read(path)
    document["actions"]["parts"][1]["high"] = 2.0
    model_file.write(document, path)
    with pytest.raises(held_out.TargetError, match="but the game takes"):
        held_out.load_champion(path)


def test_a_group_game_scores_the_same_in_worker_processes():
    from evolvarium.evolution.pool import EvaluationPool

    config = herd_config(exploration_rate=0.1)
    serial = Population(config, seed=23)
    parallel = Population(config, seed=23)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(2):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)
            assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
            assert a.ends == b.ends
    assert serial.best_ever_score == parallel.best_ever_score
