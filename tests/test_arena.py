"""The arena: its rules, what an agent sees, and the engine running it end to end.

Rule tests build a match from a seed and then place the team, the rival and the
pickups by hand, so each one checks a single rule on a known floor.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np
import pytest
from test_settings import configure, refused

from evolvarium import games, held_out, settings
from evolvarium.config import CONFIG
from evolvarium.contract import override
from evolvarium.evolution.evaluation import EvalContext, make_runner
from evolvarium.evolution.population import Population
from evolvarium.games.arena import ArenaGame, ArenaSettings, scripted
from evolvarium.games.arena.engine import (
    FALLEN_VIEW,
    INPUTS,
    EndCause,
    Match,
    distance,
    pillar_sites,
)
from evolvarium.games.arena.game import milestones
from evolvarium.nn.network import Architecture, PolicyNetwork
from evolvarium.persistence import export_model, model_file
from evolvarium.player import Player
from evolvarium.replays import unpack

ARENA = ArenaSettings()
ENDS = {cause.value for cause in EndCause}


def arena(**changes) -> ArenaSettings:
    """The default settings with each change, named section__field."""
    result = ARENA
    for name, value in changes.items():
        result = override(result, name.replace("__", "."), value)
    return result


def open_floor(**changes) -> ArenaSettings:
    """A floor without pillars, so a hand-placed match has no surprises on it."""
    return arena(floor__pillars=0, **changes)


def place(match: Match, cells, *, rival, food=(), packs=(), health=None) -> Match:
    for index, cell in enumerate(cells):
        match.team[index].cell = cell
        if health is not None:
            match.team[index].health = health[index]
    match.rival = rival
    match.food = list(food)
    match.packs = list(packs)
    match._due = [0] * len(match.packs)
    return match


def arena_config(**evolution):
    return CONFIG.with_overrides(
        game="arena",
        settings=arena(match__max_steps=60),
        evolution=dataclasses.replace(
            CONFIG.evolution, population_size=10, evaluation_episodes=2, elite_count=2,
            immigrant_count=1, workers=1, **evolution,
        ),
    )


# ----------------------------------------------------------------- the floor


def test_a_match_is_a_function_of_its_seed_and_moves():
    def played(seed):
        match = scripted.play(ARENA, scripted.hunter, seed)
        return (match.score, match.steps, match.end, match.eaten, match.catches,
                match.healed, match.strikes, [m.cell for m in match.team], match.rival)

    assert played(5) == played(5)
    assert Match(ARENA, 5).pillars != Match(ARENA, 6).pillars


@pytest.mark.parametrize("pillars", [10, len(pillar_sites(15, 15))])
def test_pillars_never_cut_the_floor_in_two(pillars):
    sites = set(pillar_sites(15, 15))
    for seed in range(20):
        match = Match(arena(floor__pillars=pillars), seed)
        assert len(match.pillars) == pillars and match.pillars <= sites
        start = match.open[0]
        seen, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for cell in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                if match.passable(cell) and cell not in seen:
                    seen.add(cell)
                    queue.append(cell)
        assert seen == set(match.open)


def test_everything_starts_on_a_cell_of_its_own():
    for seed in range(20):
        match = Match(ARENA, seed)
        cells = [m.cell for m in match.team] + [match.rival] + match.food + match.packs
        assert len(set(cells)) == len(cells)
        assert all(match.passable(cell) for cell in cells)
        assert all(m.health == ARENA.team.health for m in match.team)
        assert min(distance(m.cell, match.rival) for m in match.team) >= (
            ARENA.rival.respawn_distance
        )


# ----------------------------------------------------------------- the rules


def test_moving_onto_the_rival_catches_it_and_sends_it_far_away():
    match = place(Match(open_floor(team__size=1, pickups__packs=0), 3),
                  [(5, 5)], rival=(6, 5), food=[(0, 0)])
    assert match.step(["right"])
    assert (match.catches, match.score, match.idle) == (1, ARENA.points.catch, 0)
    assert distance(match.team[0].cell, match.rival) >= ARENA.rival.respawn_distance


def test_eating_scores_and_puts_new_food_on_an_empty_cell():
    match = place(Match(open_floor(team__size=1, pickups__food=1, pickups__packs=0), 3),
                  [(5, 5)], rival=(14, 14), food=[(5, 4)])
    match.step(["up"])
    assert (match.eaten, match.score) == (1, ARENA.points.food)
    (food,) = match.food
    assert food not in ((5, 4), match.team[0].cell, match.rival)


def test_the_rival_runs_from_a_healthy_team():
    match = place(Match(open_floor(team__size=1), 3), [(7, 7)], rival=(7, 9))
    match.step(["stay"])
    # Down, right and left all leave it three away; right comes first.
    assert match.rival == (8, 9)


def test_the_rival_hunts_a_hurt_agent_it_can_see_until_it_falls():
    match = place(Match(open_floor(team__size=1), 3), [(7, 7)], rival=(7, 10),
                  health=[ARENA.team.hurt_at])
    match.step(["stay"])
    assert match.rival == (7, 9)
    assert match.step(["stay"]) is False
    assert match.rival == (7, 8)
    assert (match.strikes, match.team[0].health, match.end) == (1, 0, EndCause.FALLEN)


def test_a_hurt_agent_out_of_sight_is_left_alone():
    match = place(Match(open_floor(team__size=1), 3), [(0, 0)], rival=(7, 7),
                  health=[ARENA.team.hurt_at])
    match.step(["stay"])
    assert distance((0, 0), match.rival) > distance((0, 0), (7, 7))


def test_a_strike_hits_the_weakest_agent_in_reach():
    still = open_floor(team__size=3, rival__rest_every=1)
    match = place(Match(still, 3), [(5, 5), (7, 5), (6, 7)], rival=(6, 5), health=[6, 4, 1])
    match.step(["stay"] * 3)
    assert [m.health for m in match.team] == [6, 2, 1]

    match = place(Match(still, 3), [(5, 5), (7, 5), (0, 0)], rival=(6, 5), health=[4, 4, 1])
    match.step(["stay"] * 3)
    assert [m.health for m in match.team] == [2, 4, 1]


def test_the_rival_rests_the_step_after_a_strike():
    moving = open_floor(team__size=1, rival__rest_every=0, rival__damage=1)
    match = place(Match(moving, 3), [(5, 5)], rival=(5, 7), health=[2])
    match.step(["stay"])
    assert (match.rival, match.team[0].health) == ((5, 6), 1)
    match.step(["up"])
    assert (match.rival, match.team[0].health) == ((5, 6), 1)
    match.step(["stay"])
    assert (match.rival, match.team[0].health) == ((5, 5), 0)


def test_a_struck_agent_can_catch_the_rival_while_it_rests():
    moving = open_floor(team__size=1, rival__rest_every=0, rival__damage=1)
    match = place(Match(moving, 3), [(5, 5)], rival=(5, 7), health=[2])
    match.step(["stay"])
    match.step(["down"])
    assert match.catches == 1
    assert distance(match.team[0].cell, match.rival) >= ARENA.rival.respawn_distance


def test_a_rival_without_damage_never_strikes_or_rests_for_it():
    harmless = open_floor(team__size=1, rival__rest_every=0, rival__damage=0)
    match = place(Match(harmless, 3), [(5, 5)], rival=(5, 7), health=[2])
    match.step(["stay"])
    match.step(["up"])
    assert (match.rival, match.team[0].health, match.strikes) == ((5, 5), 2, 0)


def test_a_pack_heals_an_agent_short_of_health_and_comes_back_after_its_delay():
    rules = open_floor(team__size=1, pickups__food=1, pickups__packs=1, pickups__pack_delay=5)
    match = place(Match(rules, 3), [(5, 5)], rival=(14, 14), food=[(0, 0)], packs=[(6, 5)])
    match.step(["right"])
    # At full health the pack stays where it is.
    assert (match.packs, match.healed) == ([(6, 5)], 0)

    match.team[0].health = 3
    match.step(["stay"])
    assert (match.packs, match.healed, match.team[0].health) == ([None], 1, ARENA.team.health)
    for _ in range(4):
        match.step(["stay"])
        assert match.packs == [None]
    match.step(["stay"])
    (pack,) = match.packs
    assert pack is not None and pack not in ((6, 5), match.rival, *match.food)


def test_a_fallen_agent_stays_put_sees_nothing_and_eats_nothing():
    match = place(Match(open_floor(team__size=2), 3), [(5, 5), (9, 9)], rival=(14, 0),
                  food=[(5, 5), (0, 14), (14, 14)], health=[0, 6])
    match.step(["right", "stay"])
    assert match.team[0].cell == (5, 5) and match.eaten == 0
    assert match.observe(0) == list(FALLEN_VIEW)
    mate = slice(INPUTS.index("mate_x"), INPUTS.index("mate_distance") + 1)
    assert match.observe(1)[mate] == [0.0, 0.0, 1.0]


def test_a_team_that_scores_nothing_runs_out_of_idle_steps():
    match = scripted.play(arena(match__idle_limit=30), scripted.still, 3)
    assert (match.end, match.steps, match.score) == (EndCause.IDLE, 30, 0)
    assert match.step(["stay"] * 3) is False and match.steps == 30


def test_a_match_ends_on_its_last_step():
    match = scripted.play(arena(match__max_steps=50), scripted.hunter, 3)
    assert (match.end, match.steps) == (EndCause.TIMEOUT, 50)


# ----------------------------------------------------------------- seeing


def test_an_agent_sees_itself_the_pickups_the_rival_and_its_mates():
    match = place(Match(open_floor(team__size=2, pickups__packs=1), 3), [(1, 0), (4, 0)],
                  rival=(0, 0), food=[(1, 4)], packs=[(3, 2)], health=[2, 6])
    match.pillars = frozenset({(1, 1)})
    span = 15 + 15 - 2
    seen = dict(zip(INPUTS, match.observe(0), strict=True))
    assert seen == pytest.approx({
        "health": 2 / 6, "hurt": 1.0,
        "food_x": 0.0, "food_y": 1.0, "food_distance": 4 / span,
        "pack_x": 0.5, "pack_y": 0.5, "pack_distance": 4 / span,
        "rival_x": -1.0, "rival_y": 0.0, "rival_distance": 1 / span, "rival_adjacent": 1.0,
        "mate_x": 1.0, "mate_y": 0.0, "mate_distance": 3 / span,
        "blocked_up": 1.0, "blocked_down": 1.0, "blocked_left": 0.0, "blocked_right": 0.0,
    })


def test_what_is_not_on_the_floor_is_seen_as_nowhere():
    match = place(Match(open_floor(team__size=1, pickups__packs=1), 3), [(7, 7)],
                  rival=(0, 0), food=[(7, 7)], packs=[None])
    seen = dict(zip(INPUTS, match.observe(0), strict=True))
    assert [seen["pack_x"], seen["pack_y"], seen["pack_distance"]] == [0.0, 0.0, 1.0]
    assert [seen["mate_x"], seen["mate_y"], seen["mate_distance"]] == [0.0, 0.0, 1.0]
    assert [seen["food_x"], seen["food_y"], seen["food_distance"]] == [0.0, 0.0, 0.0]


def test_the_game_names_one_agent_for_each_member_of_the_team():
    game = ArenaGame(arena(team__size=4))
    assert game.agents == ("agent1", "agent2", "agent3", "agent4")
    assert game.observation.names == INPUTS and game.observation.size == len(INPUTS)
    assert game.actions.names == ("up", "down", "left", "right", "stay")
    assert len(FALLEN_VIEW) == len(INPUTS)


# ----------------------------------------------------------------- watching


def test_every_agent_is_drawn_once_as_it_stands():
    game = ArenaGame()
    game.reset(7)
    while not game.over:
        frame = game.draw()
        drawn = frame["team"] + frame["hurt"] + frame["fallen"]
        assert len(drawn) == ARENA.team.size
        assert len(frame["rival"]) == 1 and len(frame["pillars"]) == ARENA.floor.pillars
        assert len(frame["food"]) == ARENA.pickups.food
        assert len(frame["packs"]) <= ARENA.pickups.packs
        assert (frame["score"], frame["standing"]) == (game.score, game.match.standing)
        assert len(frame["fallen"]) == ARENA.team.size - frame["standing"]
        actions = {agent: scripted.reckless(game.observe(agent), None) for agent in game.agents}
        game.step(actions)
    assert game.end in ENDS


def test_an_arena_replay_draws_the_match_it_recorded():
    game = ArenaGame()
    architecture = Architecture(game.observation.size, (8,), ("tanh",), game.actions.size)
    ctx = EvalContext(game="arena", settings=ARENA, architecture=architecture.to_dict(),
                      exploration_rate=0.0, replay_steps=0)
    genome = np.random.default_rng(4).normal(size=architecture.genome_size).astype(np.float32)
    runner = make_runner(ctx)
    outcome = runner.play(genome, 900_001, True)
    unwatched = runner.play(genome, 900_001, False)
    assert (outcome.score, outcome.steps, outcome.end) == (
        unwatched.score, unwatched.steps, unwatched.end)

    frames = unpack(game.scene(), outcome.replay)
    assert len(frames) == outcome.steps + 1 and outcome.replay["truncated"] is False
    assert frames[-1]["score"] == outcome.score
    pillars = frames[0]["pillars"]
    for frame in frames:
        assert frame["pillars"] == pillars
        assert len(frame["team"] + frame["hurt"] + frame["fallen"]) == ARENA.team.size


# ----------------------------------------------------------------- settings


def test_the_defaults_pass_every_check():
    settings.check(CONFIG.with_overrides(game="arena"))
    assert ArenaGame.setting_problems(ARENA) == []
    assert ArenaGame.unused_settings(ARENA) == {}


@pytest.mark.parametrize(("assignments", "problem"), [
    ((("floor.width", "4"),), "floor.width must be at least 5, not 4"),
    ((("floor.pillars", "50"),), "floor.pillars must be between 0 and 49, not 50"),
    ((("team.hurt_at", "6"),), "team.hurt_at must be between 0 and 5, not 6"),
    ((("rival.sight", "-1"),), "rival.sight must be at least 0, not -1"),
    ((("match.idle_limit", "0"),), "match.idle_limit must be at least 1, not 0"),
    ((("floor.width", "5"), ("floor.height", "5"), ("floor.pillars", "0"), ("team.size", "20")),
     "team.size, pickups.food and pickups.packs need 26 open cells with the rival, "
     "and the floor has 25"),
])
def test_arena_values_that_cannot_train_are_refused(assignments, problem):
    config = CONFIG.with_overrides(game="arena")
    assert problem in refused(configure(*assignments, config=config))


@pytest.mark.parametrize(("assignments", "reason"), [
    ((("pickups.packs", "0"), ("pickups.pack_delay", "10")), "pickups.packs is 0"),
    ((("pickups.food", "0"), ("points.food", "2")), "pickups.food is 0"),
    ((("rival.damage", "0"), ("rival.reach", "2")), "rival.damage is 0, so no agent loses health"),
    ((("rival.damage", "0"), ("team.hurt_at", "1")),
     "rival.damage is 0, so no agent loses health"),
])
def test_an_arena_setting_with_no_effect_is_refused(assignments, reason):
    config = configure(*assignments, config=CONFIG.with_overrides(game="arena"))
    name = assignments[-1][0]
    problems = refused(config, named=[name for name, _ in assignments])
    assert problems == [f"{name} has no effect, since {reason}"]
    settings.check(config)


# ----------------------------------------------------------------- milestones


def test_milestones_come_from_the_score_and_the_champion_s_signals():
    def summary(generation, best, fraction, catches, healed, fallen):
        return {
            "generation": generation, "max_episode_score": best, "scoring_fraction": fraction,
            "best_individual": {"signals": {"catches": catches, "healed": healed,
                                            "fallen": fallen}},
        }

    summaries = [
        {"generation": 0, "max_episode_score": 0, "best_individual": None},
        summary(1, 4, 0.2, 0.0, 0.0, 1.0),
        summary(2, 60, 0.6, 0.5, 0.25, 0.5),
        summary(3, 210, 0.9, 2.0, 1.0, 0.0),
    ]
    reached = {m.key: m.generation for m in milestones(summaries)}
    assert reached == {
        "score_1": 1, "score_10": 2, "score_50": 2, "score_100": 3, "score_200": 3,
        "first_catch": 2, "patched_up": 2, "regular_catches": 3, "nobody_left": 3,
        "majority_scoring": 2,
    }
    assert not any(m.reached for m in milestones(summaries[:1]))


# ----------------------------------------------------------------- the engine


def test_the_registry_knows_the_arena_by_name():
    assert "arena" in games.names()
    assert games.get("arena") is ArenaGame
    assert games.reference(ArenaGame()) == "arena"
    assert games.settings_for(CONFIG.with_overrides(game="arena")) == ARENA


def test_a_population_plays_whole_teams():
    population = Population(arena_config(), seed=31)
    first = population.run_generation()
    for _ in range(2):
        report = population.run_generation()
        assert set(report.ends) <= ENDS
        assert sum(report.ends.values()) == len(report.evaluated)
        assert population.best_ever_score >= first.best.best_episode_score
    summary = report.to_dict()
    assert set(summary["best_individual"]["signals"]) == {
        "food", "catches", "healed", "struck", "fallen"}
    assert len(milestones([summary])) == 10


def test_the_arena_scores_the_same_in_worker_processes():
    from evolvarium.evolution.pool import EvaluationPool

    config = arena_config(exploration_rate=0.1)
    serial = Population(config, seed=32)
    parallel = Population(config, seed=32)
    with EvaluationPool(parallel.context, workers=2) as pool:
        assert pool.active
        for _ in range(2):
            a = serial.run_generation()
            b = parallel.run_generation(pool=pool)
            assert [m.fitness for m in a.evaluated] == [m.fitness for m in b.evaluated]
            assert a.ends == b.ends


def test_an_arena_team_exports_and_plays_as_trained(tmp_path):
    path = tmp_path / "arena.json"
    population = Population(arena_config(), seed=33)
    population.run_generation()
    genome = population.members[0].genome
    report = export_model.build_report(
        generation=0, genome=genome, summaries=[], game=population.game,
        architecture=population.architecture, run_id="run-arena", destination=path,
    )
    assert report["ready"]
    export_model.write_model(report=report, genome=genome,
                             architecture=population.architecture, destination=path)
    assert model_file.read(path)["game"]["name"] == "arena"

    player = Player.load(path)
    network = PolicyNetwork(population.architecture).load(genome)
    game = ArenaGame(population.config.settings)
    for seed in (5, 6):
        game.reset(seed)
        while not game.over:
            actions = {}
            for agent in game.agents:
                inputs = game.observation.encode(game.observe(agent))
                actions[agent] = game.actions.decode(network.forward(inputs))
                assert player.action(inputs.tolist()) == actions[agent]
            game.step(actions)

    champion = held_out.load_champion(path)
    assert champion.context.game == "arena"
    assert champion.context.settings == population.config.settings
    held_out.score([champion], [40, 41], workers=1)
    runner = make_runner(dataclasses.replace(population.context, exploration_rate=0.0))
    assert champion.scores.tolist() == [runner.score(genome, s, 1, False).score for s in (40, 41)]


# ----------------------------------------------------------------- the bar


def test_the_scripted_hunter_beats_every_team_with_less_skill():
    seeds = range(900_000, 900_010)

    def mean(policy, read):
        return np.mean([read(scripted.play(ARENA, policy, seed)) for seed in seeds])

    scores = {name: mean(policy, lambda m: m.score) for name, policy in scripted.POLICIES.items()}
    assert scores["still"] == 0
    assert scores["wander"] < scores["forager"] < scores["hunter"]
    assert scores["reckless"] < scores["hunter"]

    def lost(m):
        return len(m.team) - m.standing

    assert mean(scripted.hunter, lost) < mean(scripted.reckless, lost)
