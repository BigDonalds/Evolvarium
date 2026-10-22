"""Cart-pole: its motion, what the network sees, and the engine running it end to end.

Motion tests start a cart from a seed and then set its state by hand, so each
one checks a single rule from a known position.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from test_settings import configure, refused

from evolvarium import games, held_out, settings
from evolvarium.config import CONFIG
from evolvarium.contract import Range, override
from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.evolution.population import Population
from evolvarium.games.cartpole import CartPoleGame, CartPoleSettings, scripted
from evolvarium.games.cartpole.engine import INPUTS, Cart, EndCause
from evolvarium.games.cartpole.game import milestones
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import export_model, model_file
from evolvarium.player import Player
from evolvarium.replays import unpack

CARTPOLE = CartPoleSettings()
ENDS = {cause.value for cause in EndCause}


def cartpole(**changes) -> CartPoleSettings:
    """The default settings with each change, named section__field."""
    result = CARTPOLE
    for name, value in changes.items():
        result = override(result, name.replace("__", "."), value)
    return result


def at(cart: Cart, x=0.0, speed=0.0, angle=0.0, spin=0.0) -> Cart:
    cart.x, cart.speed, cart.angle, cart.spin = x, speed, angle, spin
    return cart


def cartpole_config(**evolution):
    return CONFIG.with_overrides(
        game="cartpole",
        settings=cartpole(episode__max_steps=60),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1, **evolution,
        ),
    )


# ----------------------------------------------------------------- the motion


def test_an_episode_is_a_function_of_its_seed_and_forces():
    def played(seed):
        cart = Cart(CARTPOLE, seed)
        forces = np.random.default_rng(seed).uniform(-10, 10, size=200)
        states = []
        for force in forces:
            if not cart.step(force):
                break
            states.append((cart.x, cart.speed, cart.angle, cart.spin))
        return states, cart.end

    assert played(3) == played(3)
    assert played(3) != played(4)


def test_every_start_lies_within_the_spread():
    spread = CARTPOLE.episode.start_spread
    starts = [Cart(CARTPOLE, seed) for seed in range(200)]
    for cart in starts:
        assert all(abs(v) <= spread for v in (cart.x, cart.speed, cart.angle, cart.spin))
    assert len({cart.angle for cart in starts}) == len(starts)


def test_a_push_from_rest_follows_the_equations_of_motion():
    cart = at(Cart(CARTPOLE, 0))
    cart.step(10.0)
    # From upright and still, F / (M - 3m/4) for the cart and -F / (l (4M/3 - m))
    # for the pole, with M the mass of both and l half the pole.
    assert cart.speed == pytest.approx(0.02 * 10 / (1.1 - 0.075))
    assert cart.spin == pytest.approx(-0.02 * 10 / (0.5 * (4 * 1.1 / 3 - 0.1)))
    assert (cart.x, cart.angle) == (0.0, 0.0)


def test_a_leaning_pole_falls_further_and_rolls_the_cart_back():
    cart = at(Cart(CARTPOLE, 0), angle=0.05)
    cart.step(0.0)
    assert cart.spin > 0 and cart.speed < 0


def test_a_push_beyond_the_hardest_is_cut_back_to_it():
    hard, harder = at(Cart(CARTPOLE, 0)), at(Cart(CARTPOLE, 0))
    hard.step(-10.0)
    harder.step(-250.0)
    assert harder.force == -10.0
    assert (harder.speed, harder.spin) == (hard.speed, hard.spin)
    assert harder.effort == 1.0


def test_a_pole_left_alone_falls_on_the_step_it_passes_the_angle():
    cart = scripted.play(CARTPOLE, scripted.still, 11)
    assert cart.end is EndCause.FELL
    assert abs(cart.angle) > cart.fall
    assert cart.balanced == cart.steps - 1


def test_a_cart_past_the_end_of_the_track_is_off_it():
    cart = at(Cart(CARTPOLE, 0), x=2.39, speed=1.0)
    assert not cart.step(0.0)
    assert cart.end is EndCause.OFF_TRACK and cart.balanced == 0


def test_an_episode_ends_balanced_on_its_last_step():
    cart = scripted.play(cartpole(episode__max_steps=40), scripted.steady, 11)
    assert cart.end is EndCause.BALANCED
    assert cart.balanced == cart.steps == 40
    assert cart.step(0.0) is False and cart.steps == 40


def test_the_means_are_over_the_steps_played():
    cart = at(Cart(CARTPOLE, 0), x=1.2, angle=0.1)
    cart.step(5.0)
    assert cart.drift == pytest.approx(0.5)
    assert cart.lean == pytest.approx(0.1 / cart.fall)
    assert cart.effort == pytest.approx(0.5)


# ----------------------------------------------------------------- the view


def test_the_cart_sees_its_place_and_lean_as_shares_of_their_limits():
    cart = at(Cart(CARTPOLE, 0), x=-1.2, speed=0.7, angle=math.radians(6), spin=-1.5)
    assert cart.observe() == pytest.approx([-0.5, 0.7, 0.5, -1.5])
    assert len(cart.observe()) == len(INPUTS)


def test_the_game_asks_for_one_force_within_the_cart_s_hardest_push():
    game = CartPoleGame()
    assert game.agents == ("agent",)
    assert game.observation.size == len(INPUTS)
    (force,) = game.actions.parts
    assert isinstance(force, Range) and (force.low, force.high) == (-10.0, 10.0)
    gentle = CartPoleGame(cartpole(cart__force=4.0)).actions.parts[0]
    assert (gentle.low, gentle.high) == (-4.0, 4.0)


def test_the_pole_is_drawn_as_fallen_only_once_it_has():
    game = CartPoleGame()
    game.reset(11)
    scene = game.scene()
    left, bottom = scene.origin
    right, top = left + scene.width, bottom + scene.height
    while True:
        frame = game.draw()
        for name in ("track", "stops", "cart", "pole", "fallen"):
            for item in frame[name]:
                xs, ys = item[0::2], item[1::2]
                if name == "cart":
                    xs, ys = (item[0], item[0] + item[2]), (item[1], item[1] + item[3])
                assert all(left <= x <= right for x in xs)
                assert all(bottom <= y <= top for y in ys)
        assert frame["steps"] == game.score
        assert len(frame["pole"] + frame["fallen"]) == 1
        if game.over:
            break
        game.step({"agent": 0.0})
    assert game.end == "fell"
    assert frame["pole"] == [] and len(frame["fallen"]) == 1


def test_a_cartpole_replay_draws_the_episode_it_recorded():
    game = CartPoleGame()
    architecture = Architecture(game.observation.size, (8,), ("tanh",), game.actions.size)
    ctx = EvalContext(game="cartpole", settings=CARTPOLE, architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    genome = np.random.default_rng(4).normal(size=architecture.genome_size).astype(np.float32)
    runner = make_runner(ctx)
    outcome = runner.play(genome, 900_001, True)
    unwatched = runner.play(genome, 900_001, False)
    assert (outcome.score, outcome.steps, outcome.end) == (
        unwatched.score, unwatched.steps, unwatched.end)

    frames = unpack(game.scene(), outcome.replay)
    assert len(frames) == outcome.steps + 1 and outcome.replay["truncated"] is False
    assert frames[-1]["steps"] == outcome.score
    for frame in frames:
        assert frame["track"] == frames[0]["track"]
        assert len(frame["cart"]) == 1


def test_the_signals_are_the_episode_s_means():
    game = CartPoleGame()
    game.cart = scripted.play(CARTPOLE, scripted.steady, 11)
    signals = game.signals()
    assert signals["balanced"] == 1.0
    assert (signals["drift"], signals["lean"], signals["effort"]) == (
        game.cart.drift, game.cart.lean, game.cart.effort)
    game.cart = scripted.play(CARTPOLE, scripted.still, 11)
    assert game.signals()["balanced"] == 0.0


# ----------------------------------------------------------------- settings


def test_the_defaults_pass_every_check():
    settings.check(CONFIG.with_overrides(game="cartpole"))
    assert CartPoleGame.setting_problems(CARTPOLE) == []


@pytest.mark.parametrize(("assignments", "problem"), [
    ((("cart.force", "0"),), "cart.force must be above 0, not 0.0"),
    ((("pole.fall_angle", "95"),), "pole.fall_angle must be above 0 and at most 90, not 95.0"),
    ((("world.step_time", "0"),), "world.step_time must be above 0, not 0.0"),
    ((("episode.max_steps", "0"),), "episode.max_steps must be at least 1, not 0"),
    ((("episode.start_spread", "0.3"),),
     "episode.start_spread must be below 0.2094, the smaller of cart.track and "
     "pole.fall_angle in radians, not 0.3"),
])
def test_cartpole_values_that_cannot_train_are_refused(assignments, problem):
    config = CONFIG.with_overrides(game="cartpole")
    assert problem in refused(configure(*assignments, config=config))


# ----------------------------------------------------------------- milestones


def test_milestones_come_from_the_steps_and_the_champion_s_signals():
    def summary(generation, best, balanced, drift):
        return {
            "generation": generation, "max_episode_score": best,
            "best_individual": {"signals": {"balanced": balanced, "drift": drift}},
        }

    summaries = [
        {"generation": 0, "max_episode_score": 0, "best_individual": None},
        summary(1, 60, 0.0, 0.3),
        summary(2, 500, 0.5, 0.2),
        summary(3, 500, 1.0, 0.25),
        summary(4, 500, 1.0, 0.05),
    ]
    reached = {m.key: m.generation for m in milestones(summaries)}
    assert reached == {
        "steps_50": 1, "steps_100": 2, "steps_250": 2,
        "full_length": 2, "every_time": 3, "near_the_middle": 4,
    }
    assert not any(m.reached for m in milestones(summaries[:1]))


# ----------------------------------------------------------------- the engine


def test_the_registry_knows_cartpole_by_name():
    assert "cartpole" in games.names()
    assert games.get("cartpole") is CartPoleGame
    assert games.reference(CartPoleGame()) == "cartpole"
    assert games.settings_for(CONFIG.with_overrides(game="cartpole")) == CARTPOLE


def test_a_population_plays_whole_episodes():
    population = Population(cartpole_config(), seed=31)
    first = population.run_generation()
    for _ in range(2):
        report = population.run_generation()
        assert set(report.ends) <= ENDS
        assert sum(report.ends.values()) == len(report.evaluated)
        assert population.best_ever_score >= first.best.best_episode_score
    summary = report.to_dict()
    assert set(summary["best_individual"]["signals"]) == {"balanced", "drift", "lean", "effort"}
    assert len(milestones([summary])) == 6


def test_cartpole_scores_the_same_in_worker_processes():
    from evolvarium.evolution.pool import EvaluationPool

    config = cartpole_config(exploration_rate=0.1)
    serial = Population(config, seed=32)
    parallel = Population(config, seed=32)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(2):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)
            assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
            assert a.ends == b.ends


def test_a_cartpole_network_exports_and_plays_as_trained(tmp_path):
    path = tmp_path / "cartpole.json"
    population = Population(cartpole_config(), seed=33)
    population.run_generation()
    genome = population.members[0].genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="run-cartpole", destination=path,
    )
    assert report["ready"]
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=path)
    assert model_file.read(path)["game"]["name"] == "cartpole"

    player = Player.load(path)
    network = PolicyNetwork(population.architecture).load(genome)
    game = CartPoleGame(population.config.settings)
    for seed in (5, 6):
        game.reset(seed)
        while not game.over:
            inputs = game.observation.encode(game.observe("agent"))
            force = game.actions.decode(network.forward(inputs))
            # The player works in doubles and the engine in singles, so the two
            # forces agree to a few parts in a million of the cart's hardest push.
            assert player.action(inputs.tolist()) == pytest.approx(force, rel=1e-5, abs=1e-5)
            game.step({"agent": force})

    champion = held_out.load_champion(path)
    assert champion.context.game == "cartpole"
    assert champion.context.settings == population.config.settings
    held_out.score([champion], [40, 41], workers=1)
    runner = make_runner(dataclasses.replace(population.context, exploration_rate=0.0))
    assert champion.scores.tolist() == [runner.score(genome, s, 1, False).score for s in (40, 41)]


# ----------------------------------------------------------------- the bar


def test_only_the_controllers_that_read_the_pole_keep_it_up():
    seeds = range(900_000, 900_010)
    carts = {name: [scripted.play(CARTPOLE, policy, seed) for seed in seeds]
             for name, policy in scripted.POLICIES.items()}
    for name in ("still", "wander", "lean"):
        assert all(cart.end is EndCause.FELL for cart in carts[name])
    for name in ("upright", "steady"):
        assert all(cart.end is EndCause.BALANCED for cart in carts[name])

    def drift(name):
        return np.mean([cart.drift for cart in carts[name]])

    assert drift("steady") < drift("upright")
