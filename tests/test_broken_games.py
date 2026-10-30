"""The broken games: each plays its game's rules and has the fault it is named for.

Most tests play a broken game beside the game it comes from, on the same seed
and with the same moves, and compare what the two give the network.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from evolvarium import games, settings
from evolvarium.config import CONFIG
from evolvarium.evolution.pool import EvaluationPool
from evolvarium.evolution.population import Population
from evolvarium.games import broken
from evolvarium.games.arena import ArenaSettings
from evolvarium.games.cartpole import CartPoleGame, CartPoleSettings, scripted
from evolvarium.games.snake import SnakeSettings

EVERY = [(game, fault, cls) for game, faults in broken.VARIANTS.items()
         for fault, cls in faults.items()]


def every(*faults: str) -> list:
    return [pytest.param(game, cls, id=f"{game}-{fault}")
            for game, fault, cls in EVERY if fault in faults]


def side_by_side(game: str, cls: type, seed: int = 4, steps: int = 40):
    """Inputs from the game and from `cls` on each step they play together."""
    base, other = games.create(game), cls()
    base.reset(seed)
    other.reset(seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        assert other.over == base.over
        if base.over:
            return
        for agent in base.agents:
            yield (base.observation.encode(base.observe(agent)),
                   other.observation.encode(other.observe(agent)))
        actions = {agent: base.actions.explore(rng) for agent in base.agents}
        base.step(actions)
        other.step(actions)


def inputs_of(game, names: list[str]) -> list[int]:
    return [game.observation.names.index(name) for name in names]


# ------------------------------------------------------------------- finding them


def test_every_version_is_found_by_its_reference_and_none_is_built_in():
    assert set(broken.VARIANTS) == set(games.names())
    assert all(tuple(faults) == broken.FAULTS for faults in broken.VARIANTS.values())
    for game, _, cls in EVERY:
        reference = games.reference(cls)
        assert reference == f"evolvarium.games.broken:{cls.__name__}"
        assert games.get(reference) is cls
        assert issubclass(cls, games.get(game))
    assert len({cls.name for _, _, cls in EVERY} | set(games.names())) == len(EVERY) + 3


@pytest.mark.parametrize(("game", "cls"), every(*broken.FAULTS))
def test_every_version_passes_the_checks_a_run_makes(game, cls):
    config = games.resolve(CONFIG.with_overrides(game=games.reference(cls)))
    settings.check(config)
    assert isinstance(config.settings, cls.Settings)


# ------------------------------------------------------------------- the faults


@pytest.mark.parametrize(("game", "cls"), every("unscaled"))
def test_unscaled_inputs_are_the_game_s_own_in_other_units(game, cls):
    size = games.create(game).observation.size
    scale = np.resize(np.array(broken._SCALES, dtype=np.float32), size)
    offset = np.resize(np.array(broken._OFFSETS, dtype=np.float32), size)
    assert cls().observation.names == games.create(game).observation.names
    seen = []
    for theirs, ours in side_by_side(game, cls):
        np.testing.assert_allclose(ours, theirs * scale + offset, rtol=1e-6, atol=1e-6)
        seen.append(ours)
    seen = np.array(seen)
    assert np.abs(seen).max() >= 100
    assert np.all(np.abs(seen[:, 3] + 3.0) < 0.01)


@pytest.mark.parametrize(("game", "cls"), every("dead"))
def test_dead_inputs_never_change_and_sit_among_the_real_ones(game, cls):
    ours = cls()
    names = ours.observation.names
    dead = [name for name in names if name.startswith("constant ")]
    assert len(dead) == len(broken._CONSTANTS)
    places = inputs_of(ours, dead)
    assert places[0] > 0 and places[-1] < len(names) - 1
    real = [i for i, name in enumerate(names) if name not in dead]
    assert [names[i] for i in real] == list(games.create(game).observation.names)
    for theirs, values in side_by_side(game, cls):
        np.testing.assert_array_equal(values[places], broken._CONSTANTS)
        np.testing.assert_array_equal(values[real], theirs)


@pytest.mark.parametrize(("game", "cls"), every("duplicate"))
def test_duplicate_inputs_copy_others_and_one_is_flipped(game, cls):
    ours = cls()
    base = games.create(game)
    names = ours.observation.names
    copies = [name for name in names if name.startswith("copy of ")]
    assert len(copies) == 3
    sources = [base.observation.names.index(name.removeprefix("copy of ")) for name in copies]
    factors = [1.0, 1.0, -2.0]
    real = [i for i, name in enumerate(names) if name not in copies]
    assert [names[i] for i in real] == list(base.observation.names)
    for theirs, values in side_by_side(game, cls):
        np.testing.assert_array_equal(values[real], theirs)
        np.testing.assert_allclose(values[inputs_of(ours, copies)],
                                   theirs[sources] * factors, rtol=1e-6)


@pytest.mark.parametrize(("game", "cls"), every("unseeded"))
def test_an_unseeded_game_plays_a_different_episode_on_the_same_seed(game, cls):
    def played(game: object) -> np.ndarray:
        game.reset(9)
        rng = np.random.default_rng(9)
        seen = []
        for _ in range(10):
            if game.over:
                break
            seen.extend(game.observation.encode(game.observe(a)) for a in game.agents)
            game.step({a: game.actions.explore(rng) for a in game.agents})
        return np.concatenate(seen)

    base, ours = games.create(game), cls()
    same = [played(base) for _ in range(4)]
    assert all(np.array_equal(same[0], s) for s in same)
    differ = [played(ours) for _ in range(4)]
    assert not all(np.array_equal(differ[0], d) for d in differ)


def test_endless_rules_lift_only_the_step_caps():
    def lifted(settings, section: str, *caps: str) -> dict:
        part = getattr(settings, section)
        part = dataclasses.replace(part, **dict.fromkeys(caps, broken._NEVER))
        return dataclasses.asdict(dataclasses.replace(settings, **{section: part}))

    def endless(cls: type) -> dict:
        return dataclasses.asdict(cls.Settings())

    assert endless(broken.EndlessSnake) == lifted(
        SnakeSettings(), "episode", "starvation_limit", "max_steps")
    assert endless(broken.EndlessCartPole) == lifted(
        CartPoleSettings(), "episode", "max_steps")
    assert endless(broken.EndlessArena) == lifted(
        ArenaSettings(), "match", "max_steps", "idle_limit")


def test_an_endless_cart_pole_kept_upright_keeps_going():
    def steps(game) -> int:
        game.reset(1)
        rng = random.Random(1)
        played = 0
        while not game.over and played < 2_000:
            view = game.observe("agent")
            game.step({"agent": scripted.steady(view, rng) * game.settings.cart.force})
            played += 1
        return played

    assert steps(CartPoleGame()) == CartPoleSettings().episode.max_steps
    assert steps(broken.EndlessCartPole()) == 2_000


# ------------------------------------------------------------------- training on them


def cartpole_config(cls: type):
    short = dataclasses.replace(CartPoleSettings().episode, max_steps=60)
    return CONFIG.with_overrides(
        game=games.reference(cls),
        settings=dataclasses.replace(cls.Settings(), episode=short),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1,
        ),
    )


# The endless version is left out: a genome that balances the pole would play
# until the episode's time ran out.
@pytest.mark.parametrize("fault", ["unscaled", "dead", "duplicate", "unseeded"])
def test_a_population_trains_on_each_broken_cart_pole(fault):
    cls = broken.VARIANTS["cartpole"][fault]
    population = Population(cartpole_config(cls), seed=41)
    assert population.architecture.input_size == cls().observation.size
    for _ in range(2):
        report = population.run_generation()
        assert len(report.evaluated) == 10
        assert sum(report.ends.values()) == 10


def test_a_broken_game_scores_the_same_in_worker_processes():
    config = cartpole_config(broken.DeadInputCartPole)
    serial = Population(config, seed=42)
    parallel = Population(config, seed=42)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        a = serial.run_generation()
        b = parallel.run_generation(pool=pool)
    assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
