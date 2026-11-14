"""The grid encoder: inputs built from what a game says is on its grid.

`Forage` declares a grid and no inputs of its own, so every run of it is played
through the encoder. The registry imports it as "test_grid:Forage", and worker
processes find it the same way.
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass

import numpy as np
import pytest
from test_golden import GOLDEN, golden_settings
from test_jobs import train

from evolvarium import games, held_out, probe, profiling, settings
from evolvarium.config import CONFIG, GridConfig, ProbeConfig
from evolvarium.contract import STEPS, Actions, Choice, Game, Grid, View
from evolvarium.evolution.evaluation import GameFault, make_runner
from evolvarium.evolution.population import Population
from evolvarium.grid import Encoder, GridGame
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import checkpoint, export_model, model_file
from evolvarium.settings import SettingsError
from evolvarium.trainer import Trainer
from evolvarium.web.runs import SavedRun

FORAGE = "test_grid:Forage"
COMPASS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
ON = GridConfig(enabled=True)
UP, RIGHT, DOWN, LEFT = STEPS
# A 5 by 5 grid whose agents turn, with food to find and one gauge.
TURNING = Grid(5, 5, {"l": "left", "a": "ahead", "r": "right"}, things=("food",), gauges=("hp",))


@dataclass(frozen=True)
class ForageSettings:
    size: int = 7
    walls: int = 6
    max_steps: int = 60


class Forage(Game):
    """One forager on a grid with a few walls, eating food that comes back elsewhere."""

    name = "forage"
    Settings = ForageSettings
    actions = Actions(Choice("move", tuple(COMPASS)))

    def grid(self):
        size = self.settings.size
        return Grid(size, size, COMPASS, things=("food",))

    def reset(self, seed):
        size = self.settings.size
        self.rng = random.Random(seed)
        cells = [(x, y) for y in range(size) for x in range(size)]
        self.walls = set(self.rng.sample(cells, self.settings.walls))
        self.open = [cell for cell in cells if cell not in self.walls]
        self.at, self.food = self.rng.sample(self.open, 2)
        self.eaten = 0
        self._steps = 0

    def view(self, agent):
        return View(at=self.at, blocked=self.walls, things={"food": (self.food,)})

    def step(self, actions):
        dx, dy = COMPASS[actions["agent"]]
        cell = (self.at[0] + dx, self.at[1] + dy)
        size = self.settings.size
        if 0 <= cell[0] < size and 0 <= cell[1] < size and cell not in self.walls:
            self.at = cell
        if self.at == self.food:
            self.eaten += 1
            self.food = self.rng.choice([c for c in self.open if c != self.at])
        self._steps += 1

    @property
    def over(self):
        return self._steps >= self.settings.max_steps

    @property
    def end(self):
        return "timeout"

    @property
    def score(self):
        return self.eaten

    @property
    def steps(self):
        return self._steps


class LostForage(Forage):
    """Forage whose view puts the forager off the grid."""

    def view(self, agent):
        return View(at=(-1, 0))


class FailingForage(Forage):
    """Forage whose view raises."""

    def view(self, agent):
        raise KeyError("no such cell")


def forage_config(tmp_path=None, **changes):
    config = CONFIG.with_overrides(**{
        "game": FORAGE,
        "evolution": dataclasses.replace(
            CONFIG.evolution, population_size=30, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1,
        ),
        **changes,
    })
    if tmp_path is None:
        return config
    return dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, runs_dir=tmp_path / "runs",
                                  exports_dir=tmp_path / "exports"),
    )


def inputs(view, heading=UP, grid=TURNING, radius=1, horizon=32) -> dict[str, float]:
    encoder = Encoder(grid, radius, horizon)
    return dict(zip(encoder.names, encoder.encode(view, heading).tolist(), strict=True))


def blocked(found: dict[str, float]) -> set[str]:
    return {name[len("blocked_"):] for name, value in found.items()
            if name.startswith("blocked_") and value == 1.0}


# ------------------------------------------------------------------ the grid


@pytest.mark.parametrize(("moves", "message"), [
    ({"go": "ahead"}, "at least two"),
    ({"a": "ahead", "b": "sideways"}, "turns are"),
    ({"a": "ahead", "b": "ahead"}, "same way"),
    ({**COMPASS, "west": (-2, 0)}, "one cell"),
    ({"north": (0, -1), "east": (1, 0), "south": (0, 1)}, "all four"),
    ({"north": (0, -1), "on": "ahead"}, "one cell"),
])
def test_a_grid_says_what_is_wrong_with_its_moves(moves, message):
    with pytest.raises(ValueError, match=message):
        Grid(5, 5, moves)


def test_a_grid_names_its_things_and_gauges_once():
    with pytest.raises(ValueError, match="distinct"):
        Grid(5, 5, COMPASS, things=("food",), gauges=("food",))
    with pytest.raises(ValueError, match="it can name 8"):
        Grid(5, 5, COMPASS, things=tuple("abcdefghi"))
    with pytest.raises(ValueError, match="whole number"):
        Grid(0, 5, COMPASS)


def test_steps_given_as_lists_are_read_as_steps():
    grid = Grid(5, 5, {name: list(step) for name, step in COMPASS.items()})
    assert not grid.turning
    assert grid.moves == COMPASS


# ------------------------------------------------------------------ the inputs


def test_the_cells_around_an_agent_are_named_from_where_it_faces():
    names = Encoder(TURNING, 1, 32).names
    assert names == (
        "blocked_ahead_left", "blocked_ahead", "blocked_ahead_right",
        "blocked_left", "blocked_right",
        "blocked_behind_left", "blocked_behind", "blocked_behind_right",
        "space_left", "space_ahead", "space_right",
        "food_via_left", "food_via_ahead", "food_via_right",
        "food_forward", "food_lateral", "crowding", "hp",
    )
    wider = Encoder(TURNING, 2, 32).names
    assert sum(name.startswith("blocked_") for name in wider) == 24
    assert wider[:2] == ("blocked_2ahead_2left", "blocked_2ahead_left")


def test_steps_give_the_network_every_turn_and_staying_put():
    grid = Grid(5, 5, {**COMPASS, "wait": (0, 0)})
    assert Encoder(grid, 1, 32).turns == ("left", "ahead", "right", "back", "stay")
    assert Encoder(Grid(5, 5, COMPASS), 1, 32).turns == ("left", "ahead", "right", "back")


def _turned(cell, times):
    """`cell` on the 5 by 5 grid, turned clockwise about its centre `times` times."""
    x, y = cell
    for _ in range(times):
        x, y = 4 - y, x
    return (x, y)


def _turned_step(step, times):
    x, y = step
    for _ in range(times):
        x, y = -y, x
    return (x, y)


def test_one_scene_reads_the_same_whichever_way_it_is_turned():
    walls = [(1, 1), (3, 0), (2, 4), (0, 3)]
    body = {(1, 2): 3, (1, 3): 1}
    food = [(4, 4), (0, 0)]
    readings = []
    for times in range(4):
        view = View(
            at=_turned((2, 2), times),
            blocked=[_turned(cell, times) for cell in walls],
            clearing={_turned(cell, times): moves for cell, moves in body.items()},
            things={"food": [_turned(cell, times) for cell in food]},
            gauges={"hp": 0.5},
        )
        readings.append(inputs(view, _turned_step(UP, times)))
    assert all(reading == readings[0] for reading in readings[1:])
    assert blocked(readings[0]) == {"ahead_left", "left"}


def test_the_edge_of_the_grid_is_blocked():
    found = inputs(View(at=(0, 0), gauges={"hp": 1.0}))
    assert blocked(found) == {"ahead_left", "ahead", "ahead_right", "left", "behind_left"}
    assert found["space_ahead"] == found["space_left"] == 0.0


def test_a_cell_that_opens_as_the_agent_arrives_reads_open():
    def ahead(moves, cell=(2, 1)):
        return blocked(inputs(View(at=(2, 2), clearing={cell: moves}, gauges={"hp": 1.0})))

    assert ahead(1) == set()
    assert ahead(2) == {"ahead"}
    # Two moves reach a corner, so it may open as late as the second.
    assert ahead(2, (3, 1)) == set()
    assert ahead(3, (3, 1)) == {"ahead_right"}


def test_space_tells_a_pocket_from_open_floor():
    # A wall down the middle with the agent in its gap, and the left side smaller.
    walls = [(2, 0), (2, 1), (2, 3), (2, 4), (0, 0), (1, 0), (0, 4), (1, 4)]
    # The agent's own cell stays taken, as a body's would.
    found = inputs(View(at=(2, 2), blocked=walls, clearing={(2, 2): 50}, gauges={"hp": 1.0}))
    assert found["space_ahead"] == 0.0
    assert found["space_left"] == pytest.approx(6 / 16)
    assert found["space_right"] == pytest.approx(10 / 16)


@pytest.mark.parametrize(("opens", "space"), [(2, 1.0), (3, 1 / 3)])
def test_space_waits_for_a_body_to_move_out_of_the_way(opens, space):
    corridor = Grid(5, 1, {"l": "left", "a": "ahead", "r": "right"})
    # Facing along a corridor, with the agent's body behind and one cell in the way.
    view = View(at=(0, 0), clearing={(0, 0): 50, (2, 0): opens})
    found = inputs(view, RIGHT, grid=corridor)
    assert found["space_ahead"] == pytest.approx(space)
    assert found["space_left"] == found["space_right"] == 0.0


def via(found: dict[str, float], thing: str = "food") -> set[str]:
    return {name.rpartition("_")[2] for name, value in found.items()
            if name.startswith(f"{thing}_via_") and value == 1.0}


def test_the_moves_that_start_a_shortest_way_to_a_thing_are_marked():
    view = View(at=(2, 2), things={"food": [(2, 0)]}, gauges={"hp": 1.0})
    found = inputs(view)
    assert via(found) == {"ahead"}
    assert found["food_forward"] == pytest.approx(2 / 5)
    assert found["food_lateral"] == 0.0

    # With a wall in the way, going round either side is as short.
    assert via(inputs(dataclasses.replace(view, blocked=[(2, 1)]))) == {"left", "right"}
    # Facing the other way, the food is behind, and every turn takes four moves.
    behind = inputs(view, DOWN)
    assert behind["food_forward"] == pytest.approx(-2 / 5)
    assert via(behind) == {"left", "ahead", "right"}


def test_the_nearest_thing_of_a_kind_is_the_one_headed_for():
    view = View(at=(2, 2), things={"food": [(0, 2), (4, 0)]}, gauges={"hp": 1.0})
    assert via(inputs(view)) == {"left"}
    assert inputs(view)["food_lateral"] == pytest.approx(-2 / 5)


def test_a_thing_beyond_the_horizon_has_a_place_but_no_way_to_it():
    view = View(at=(2, 2), things={"food": [(2, 0)]}, gauges={"hp": 1.0})
    assert via(inputs(view, horizon=2)) == {"ahead"}
    near = inputs(view, horizon=1)
    assert via(near) == set()
    assert near["food_forward"] == pytest.approx(2 / 5)


def test_no_thing_of_a_kind_reads_as_nothing_anywhere():
    found = inputs(View(at=(2, 2), gauges={"hp": 1.0}))
    assert [found[f"food_{side}"] for side in ("via_left", "via_ahead", "via_right",
                                                "forward", "lateral")] == [0.0] * 5


def test_crowding_counts_what_is_blocked_now_and_gauges_pass_through():
    view = View(at=(2, 2), blocked=[(0, 0), (1, 0), (9, 9)], clearing={(4, 4): 2},
                gauges={"hp": -3.5})
    found = inputs(view)
    assert found["crowding"] == pytest.approx(3 / 25)
    assert found["hp"] == -3.5


@pytest.mark.parametrize(("view", "heading", "message"), [
    (View(at=(5, 0), gauges={"hp": 1.0}), UP, "off the 5 by 5 grid"),
    (View(at=(2, 2), gauges={"hp": 1.0}), (1, 1), "heading"),
    (View(at=(2, 2)), UP, "gauge 'hp'"),
])
def test_a_view_the_encoder_cannot_read_is_refused(view, heading, message):
    with pytest.raises(ValueError, match=message):
        Encoder(TURNING, 1, 32).encode(view, heading)


def test_two_inputs_with_one_name_are_refused():
    with pytest.raises(ValueError, match="'crowding'"):
        Encoder(Grid(5, 5, COMPASS, gauges=("crowding",)), 1, 32)


@pytest.mark.parametrize(("radius", "horizon"), [(0, 32), (1, 0), (1, 255)])
def test_the_encoder_sees_and_looks_ahead_within_its_range(radius, horizon):
    with pytest.raises(ValueError):
        Encoder(TURNING, radius, horizon)


# ------------------------------------------------------------------ Snake


def test_snake_through_the_grid_reads_what_its_own_inputs_read():
    """Ten of Snake's eleven inputs are ones the encoder builds for any grid game.

    The eleventh, how far the food is, becomes which turns start a shortest way
    to it. The 81.5 model plays one of its held-out games on its own inputs, its
    snake growing to 45 long, and the grid reads the same at every step.
    """
    same = {
        "danger_left": "blocked_left", "danger_straight": "blocked_ahead",
        "danger_right": "blocked_right", "space_left": "space_left",
        "space_straight": "space_ahead", "space_right": "space_right",
        "food_forward": "food_forward", "food_lateral": "food_lateral",
        "length_ratio": "crowding", "starvation_progress": "hunger",
    }
    own = games.create("snake", golden_settings())
    gridded = GridGame(games.create("snake", golden_settings()), ON)
    assert gridded.actions is gridded.base.actions
    forward = PolicyNetwork(Architecture.from_dict(GOLDEN["architecture"])).load(
        np.array(GOLDEN["genome"], dtype=np.float32)).forward
    mine, theirs = own.observation.names, gridded.observation.names
    columns = [(mine.index(name), theirs.index(other)) for name, other in same.items()]
    seed = 900_041
    own.reset(seed)
    gridded.reset(seed)
    while not own.over:
        a = own.observation.encode(own.observe("agent"))
        b = gridded.observe("agent")
        assert all(a[i] == b[j] for i, j in columns), own.steps
        turn = own.actions.decode(forward(a))
        own.step({"agent": turn})
        gridded.step({"agent": turn})
    recorded = next(game for game in GOLDEN["games"] if game["seed"] == seed)
    assert (own.score, own.steps) == (recorded["score"], recorded["steps"])
    assert (gridded.score, gridded.steps) == (own.score, own.steps)


# ------------------------------------------------------------------ the arena


def test_the_arena_is_steered_by_turns_from_the_way_each_agent_faces():
    game = GridGame(games.create("arena"), ON)
    assert game.actions.names == ("left", "ahead", "right", "back", "stay")
    game.reset(900_000)
    match = game.base.match
    first, second = game.agents[:2]
    start = match.team[0].cell
    for agent in game.agents:
        game.observe(agent)
    # Everyone starts facing up, so a right turn steps right.
    game.step({agent: "right" if agent == first else "stay" for agent in game.agents})
    assert match.team[0].cell == (start[0] + 1, start[1])
    assert game._facing[first] == RIGHT
    assert game._facing[second] == UP
    # Facing right, another right turn steps down, and staying keeps the heading.
    game.step({agent: "right" if agent == first else "stay" for agent in game.agents})
    assert match.team[0].cell == (start[0] + 1, start[1] + 1)
    game.step({agent: "stay" for agent in game.agents})
    assert game._facing[first] == DOWN


def test_an_agent_s_heading_starts_up_again_each_episode():
    game = GridGame(games.create("arena"), ON)
    game.reset(1)
    game.step({agent: "left" for agent in game.agents})
    game.reset(1)
    assert game._facing == {}


def test_the_arena_s_view_has_its_pickups_rival_and_teammates():
    game = GridGame(games.create("arena"), ON)
    game.reset(900_000)
    match = game.base.match
    values = dict(zip(game.observation.names, game.observe(game.agents[0]).tolist(),
                      strict=True))
    assert values["health"] == 1.0
    assert values["hurt"] == 0.0
    assert values["crowding"] == pytest.approx(len(match.pillars) / (15 * 15))
    for thing in ("food", "packs", "rival", "mates"):
        assert max(values[f"{thing}_via_{turn}"] for turn in game.actions.names) > 0


# ------------------------------------------------------------------ playing through it


def test_a_game_with_no_grid_cannot_be_played_through_one():
    with pytest.raises(ValueError, match="no grid"):
        GridGame(games.create("cartpole"), ON)


def test_a_grid_s_moves_must_be_its_game_s_one_action():
    class Mismatched(Forage):
        def grid(self):
            return Grid(7, 7, {**COMPASS, "wait": (0, 0)})

    class TwoParts(Forage):
        actions = Actions(Choice("move", tuple(COMPASS)), Choice("say", ("hi", "bye")))

    with pytest.raises(ValueError, match="one of"):
        GridGame(Mismatched(), ON)
    with pytest.raises(ValueError, match="one action"):
        GridGame(TwoParts(), ON)


def test_turns_need_the_view_to_say_which_way_the_agent_faces():
    class Headless(Forage):
        actions = Actions(Choice("move", ("left", "ahead", "right")))

        def grid(self):
            return Grid(7, 7, {"left": "left", "ahead": "ahead", "right": "right"})

    game = GridGame(Headless(), ON)
    game.reset(1)
    with pytest.raises(ValueError, match="no heading"):
        game.observe("agent")


def test_a_game_with_only_a_grid_is_always_played_through_it():
    population = Population(forage_config(), seed=1)
    assert population.grid == ON
    assert isinstance(population.game, GridGame)
    assert population.game.observation.names[-1] == "crowding"
    assert population.game.model_inputs()["encoding"] == "grid"
    # And a game with inputs of its own only when a run asks.
    assert Population(CONFIG, seed=1).grid is None
    assert Population(dataclasses.replace(CONFIG, grid=ON), seed=1).grid == ON


def test_a_fault_in_a_view_is_blamed_on_the_view():
    for cls, message in ((FailingForage, "FailingForage.view raised KeyError"),
                         (LostForage, "reading LostForage.view's output as inputs raised")):
        config = forage_config(game=games.reference(cls))
        population = Population(config, seed=1)
        runner = make_runner(population.context)
        with pytest.raises(GameFault, match=message):
            runner.score(population.members[0].genome, 7, 1, False)


def test_a_game_through_the_grid_learns(tmp_path):
    """Random moves eat 0.3 food a game. Following the encoder's food_via inputs
    by hand eats 11."""
    trainer = train(Trainer(forage_config(tmp_path), run_id="forage", seed=4,
                            max_generations=12))
    summaries = trainer.history.summaries()
    assert summaries[0]["mean_score"] < 0.5
    assert summaries[-1]["mean_score"] >= 1.0
    assert max(summary["best_individual"]["score"] for summary in summaries) >= 5


def test_a_run_through_the_grid_keeps_it_from_checkpoint_to_model(tmp_path):
    config = forage_config(tmp_path)
    trainer = train(Trainer(config, run_id="forage", seed=4, max_generations=2))
    path = checkpoint.list_checkpoints(trainer.checkpoint_dir)[-1].path
    meta = checkpoint.read_meta(path)
    assert meta["encoding"] == "grid"
    assert meta["grid"] == dataclasses.asdict(ON)
    assert held_out.run_context(path).grid == ON

    genome = trainer.population.best_ever_genome
    champion = held_out.load_champion(path)
    held_out.score([champion], [40, 41], workers=1)
    runner = make_runner(dataclasses.replace(trainer.population.context, exploration_rate=0.0))
    expected = [runner.score(genome, seed, 1, False).score for seed in (40, 41)]
    assert champion.scores.tolist() == expected

    destination = tmp_path / "forage.json"
    report = export_model.build_report(
        generation=2, genome=genome, summaries=trainer.history.summaries(),
        game=trainer.population.game, architecture=trainer.population.architecture,
        run_id="forage", destination=destination,
    )
    assert report["game"]["name"] == FORAGE
    assert report["inputs"]["names"] == list(trainer.population.game.observation.names)
    export_model.write_model(report=report, genome=genome,
                             architecture=trainer.population.architecture,
                             destination=destination)
    document = model_file.read(destination)
    assert document["inputs"]["encoding"] == "grid"
    assert document["inputs"]["settings"] == {
        "radius": 1, "horizon": 32, "game": {"encoding": "forage", "settings": {}},
    }
    model = held_out.load_champion(destination)
    assert model.context.grid == ON
    held_out.score([model], [40, 41], workers=1)
    assert model.scores.tolist() == expected

    saved = SavedRun(trainer.run_dir, config.history)
    assert saved.problem is None
    assert saved.context().grid == ON


def test_a_snake_model_through_the_grid_names_snake_s_own_inputs(tmp_path):
    config = dataclasses.replace(CONFIG, grid=ON)
    population = Population(config, seed=1)
    destination = tmp_path / "snake.json"
    genome = population.members[0].genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="snake", destination=destination,
    )
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=destination)
    inner = model_file.read(destination)["inputs"]["settings"]["game"]
    assert inner["encoding"] == "egocentric_v3"
    model = held_out.load_champion(destination)
    assert model.context.grid == ON
    assert model.context.settings.features.spec == "egocentric_v3"


def test_a_checkpoint_resumes_only_under_the_grid_it_trained_with(tmp_path):
    config = dataclasses.replace(CONFIG, grid=ON, paths=dataclasses.replace(
        CONFIG.paths, runs_dir=tmp_path / "runs"))
    population = Population(config, seed=1)
    path = tmp_path / "grid.npz"
    checkpoint.save(population, path)
    assert checkpoint.load(path, config).grid == ON
    with pytest.raises(checkpoint.CheckpointMismatch, match="'grid' inputs"):
        checkpoint.load(path, CONFIG)
    farther = dataclasses.replace(config, grid=dataclasses.replace(ON, horizon=40))
    with pytest.raises(checkpoint.CheckpointMismatch, match="grid settings"):
        checkpoint.load(path, farther)


def test_the_probe_measures_the_inputs_the_encoder_builds():
    config = forage_config(probe=ProbeConfig(enabled=True, episodes=5, search_steps=0,
                                             seconds=600.0))
    report = probe.run(config)
    names = [row.name for row in report.inputs]
    assert names == list(Population(config, seed=1).game.observation.names)
    # Six walls on every floor leave the share blocked the same in every episode.
    dead = [name for finding in report.findings if finding.kind == "dead"
            for name in finding.inputs]
    assert "crowding" in dead


# ------------------------------------------------------------------ settings


def check(config, named=()):
    settings.check(games.resolve(config), named=named)


class Blank(Game):
    """A game with no inputs at all."""

    actions = Actions(Choice("move", ("left", "right")))


def test_a_run_through_the_grid_needs_a_game_with_one():
    with pytest.raises(SettingsError, match="cartpole has no grid to read"):
        check(dataclasses.replace(CONFIG, game="cartpole", grid=ON))
    check(dataclasses.replace(CONFIG, grid=ON))
    with pytest.raises(SettingsError, match="neither inputs of its own nor a grid"):
        check(dataclasses.replace(CONFIG, game=games.reference(Blank)))


def test_the_grid_s_settings_have_no_effect_while_it_is_off():
    with pytest.raises(SettingsError, match="grid.radius has no effect, since grid.enabled"):
        check(dataclasses.replace(CONFIG, grid=GridConfig(radius=2)), named=["grid.radius"])
    with pytest.raises(SettingsError, match="always played through the grid"):
        check(forage_config(grid=ON), named=["grid.enabled"])


@pytest.mark.parametrize(("changes", "message"), [
    ({"radius": 0}, "grid.radius"),
    ({"horizon": 0}, "grid.horizon"),
    ({"horizon": 255}, "grid.horizon"),
])
def test_the_grid_sees_and_looks_ahead_within_its_range(changes, message):
    with pytest.raises(SettingsError, match=message):
        check(dataclasses.replace(CONFIG, grid=dataclasses.replace(ON, **changes)))


def test_runs_through_the_grid_are_compared_apart_from_runs_without_it():
    def manifest_of(grid):
        config = dataclasses.replace(games.resolve(CONFIG), grid=grid)
        return json.loads(json.dumps(dataclasses.asdict(config), default=str))

    off = held_out.search_settings(manifest_of(GridConfig()))
    older = manifest_of(GridConfig())
    del older["grid"]
    assert held_out.search_settings(older) == off
    on = held_out.search_settings(manifest_of(ON))
    assert on != off
    assert held_out.describe(off) == "defaults"
    assert held_out.describe(on) == "grid.enabled=true"


def test_a_profile_times_the_grid_encoder():
    population = Population(forage_config(), seed=1)
    genome = population.members[0].genome
    parts = {part.name for part in profiling.split_inputs(population.context, genome, [700_000])}
    # Forage's walls never open, so its searches all go through the shared one.
    assert {"encode itself", "grid.py _together"} <= parts
