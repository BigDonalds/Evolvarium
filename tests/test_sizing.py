"""Fitting a run to the time someone has for it.

The size is worked out from a measurement, so most of these give it the
measurement and check the arithmetic and the run. A sized run has to be the same
run as one given that size by hand, and the time has to hold however it was spent.
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest
from test_guards import Trouble  # noqa: F401
from test_jobs import assert_same_population, make_config, played, train, wait_for

from evolvarium import probe, settings, sizing
from evolvarium.cli import main
from evolvarium.config import CONFIG
from evolvarium.events import EventBus, Note, RunFinished
from evolvarium.evolution.evaluation import GameFault
from evolvarium.evolution.pool import WorkerBudget
from evolvarium.games import broken
from evolvarium.games import reference as game_reference
from evolvarium.persistence import checkpoint, jobs
from evolvarium.settings import SettingsError
from evolvarium.telemetry.history import read_summaries
from evolvarium.trainer import Trainer
from evolvarium.web.runs import Lab

EVOLUTION = CONFIG.evolution
POPULATION = "evolution.population_size"
EPISODES = "evolution.evaluation_episodes"
ELITES = "evolution.elite_count"
IMMIGRANTS = "evolution.immigrant_count"


def timed(tmp_path, minutes: float = 10.0):
    config = make_config(tmp_path)
    return dataclasses.replace(
        config, sizing=dataclasses.replace(config.sizing, minutes=minutes)
    )


def a_plan(population=10, episodes=2, *, minutes=10.0, setup=1.0, elites=2, immigrants=1,
           generations=150, wanted=150, kept=()) -> sizing.Plan:
    return sizing.Plan(
        minutes=minutes, workers=1, seconds_per_game=0.01, games_sampled=100, mean_steps=50.0,
        population=population, episodes=episodes, elites=elites, immigrants=immigrants,
        generations=generations, wanted=wanted, setup_seconds=setup, kept=kept,
    )


def measured_at(monkeypatch, seconds_per_game: float, setup: float = 0.0) -> list:
    """Have the sample report `seconds_per_game`, recording what each one was given."""
    calls = []

    def sample(config, fixes, workers):
        calls.append({"config": config, "fixes": fixes, "workers": workers})
        return seconds_per_game, 100, 50.0

    monkeypatch.setattr(sizing, "_sample", sample)
    return calls


# ---------------------------------------------------------------------- the size


def seconds_for(generations: int, population: int, episodes: int, seconds_per_game: float) -> float:
    return generations * population * episodes * seconds_per_game * sizing._GROWTH


def test_the_largest_size_that_leaves_time_for_the_generations_is_chosen():
    sizes = sizing._sizes(EVOLUTION, set())
    for size in sizes:
        left = seconds_for(150, *size, 0.01) * 1.001
        assert sizing._choose(sizes, left, 0.01, 150) == size


def test_time_for_more_than_the_largest_size_gets_the_largest():
    sizes = sizing._sizes(EVOLUTION, set())
    assert sizing._choose(sizes, 1e9, 0.01, 150) == sizes[-1]


def test_too_little_time_for_any_size_gets_the_smallest():
    sizes = sizing._sizes(EVOLUTION, set())
    assert sizing._choose(sizes, 0.5, 0.01, 150) == sizes[0]


def test_the_sizes_run_from_cheapest_to_dearest_and_end_at_the_defaults():
    costs = [population * episodes for population, episodes in sizing.SIZES]
    assert costs == sorted(costs)
    assert sizing.SIZES[-1] == (EVOLUTION.population_size, EVOLUTION.evaluation_episodes)


def test_a_population_asked_for_by_name_stays_and_the_games_fit_around_it():
    sizes = sizing._sizes(dataclasses.replace(EVOLUTION, population_size=40), {POPULATION})
    assert {population for population, _ in sizes} == {40}
    assert {episodes for _, episodes in sizes} == {2, 3, 4}


def test_games_asked_for_by_name_stay_and_the_population_fits_around_them():
    sizes = sizing._sizes(dataclasses.replace(EVOLUTION, evaluation_episodes=7), {EPISODES})
    assert {episodes for _, episodes in sizes} == {7}
    assert sizes[0][0] == sizing.SIZES[0][0]


def test_both_asked_for_by_name_leave_nothing_to_choose():
    evolution = dataclasses.replace(EVOLUTION, population_size=40, evaluation_episodes=7)
    assert sizing._sizes(evolution, {POPULATION, EPISODES}) == [(40, 7)]


def test_elites_and_immigrants_keep_their_share_of_the_population():
    assert sizing._carried(EVOLUTION, EVOLUTION.population_size, set()) == (
        EVOLUTION.elite_count, EVOLUTION.immigrant_count)
    assert sizing._carried(EVOLUTION, 75, set()) == (3, 2)
    assert sizing._carried(EVOLUTION, 30, set()) == (1, 1)


def test_a_run_with_no_immigrants_is_not_given_any():
    evolution = dataclasses.replace(EVOLUTION, immigrant_count=0)
    assert sizing._carried(evolution, 30, set()) == (1, 0)


def test_elites_and_immigrants_asked_for_by_name_stay():
    evolution = dataclasses.replace(EVOLUTION, elite_count=40, immigrant_count=9)
    assert sizing._carried(evolution, 75, {ELITES, IMMIGRANTS}) == (40, 9)


def test_a_size_that_leaves_no_room_for_offspring_is_not_open():
    evolution = dataclasses.replace(EVOLUTION, elite_count=60, immigrant_count=4)
    sizes = sizing._sizes(evolution, {ELITES, IMMIGRANTS})
    assert sizes and all(population > 64 for population, _ in sizes)


def test_a_size_too_small_for_the_tournament_is_not_open():
    evolution = dataclasses.replace(EVOLUTION, tournament_size=60)
    assert all(population >= 60 for population, _ in sizing._sizes(evolution, set()))


def test_settings_that_leave_every_size_closed_fall_back_on_the_ones_given():
    evolution = dataclasses.replace(EVOLUTION, tournament_size=200)
    assert sizing._sizes(evolution, set()) == [(150, 4)]


# ---------------------------------------------------------------------- the plan


def test_a_plan_comes_from_what_the_sample_measured(tmp_path, monkeypatch):
    calls = measured_at(monkeypatch, 0.001)
    config = timed(tmp_path, minutes=1.0)
    plan = sizing.plan(config, workers=3)
    assert calls[0]["workers"] == 3
    assert plan.seconds_per_game == 0.001
    assert plan.games_sampled == 100
    assert plan.workers == 3
    left = 60.0 - plan.setup_seconds
    assert plan.generations == sizing._generations(left, 0.001, plan.population, plan.episodes)
    assert plan.generations >= plan.wanted
    assert plan.elites >= 1


def test_the_time_a_probe_took_comes_out_of_what_the_run_has(tmp_path, monkeypatch):
    measured_at(monkeypatch, 0.001)
    config = timed(tmp_path, minutes=1.0)
    assert sizing.plan(config, probe_seconds=30.0).setup_seconds >= 30.0
    roomy = sizing.plan(config)
    tight = sizing.plan(config, probe_seconds=59.9)
    assert (tight.population * tight.episodes) < (roomy.population * roomy.episodes)


def test_a_plan_names_what_it_left_alone(tmp_path, monkeypatch):
    measured_at(monkeypatch, 0.001)
    plan = sizing.plan(timed(tmp_path), named=[POPULATION, "board.width"])
    assert plan.kept == (POPULATION,)
    assert any(POPULATION in note for note in plan.notes())


def test_a_plan_says_so_when_the_time_will_not_stretch(tmp_path, monkeypatch):
    measured_at(monkeypatch, 1.0)
    plan = sizing.plan(timed(tmp_path, minutes=1.0))
    assert plan.generations < plan.wanted
    assert (plan.population, plan.episodes) == sizing.SIZES[0]
    assert any("more time would allow" in note for note in plan.notes())


def test_a_plan_applied_changes_only_the_size(tmp_path):
    config = timed(tmp_path)
    sized = a_plan(30, 2, elites=1, immigrants=1).apply(config)
    assert sized.evolution.population_size == 30
    assert sized.evolution.evaluation_episodes == 2
    assert sized.evolution.elite_count == 1
    assert sized.evolution.immigrant_count == 1
    assert dataclasses.replace(
        sized.evolution,
        population_size=config.evolution.population_size,
        evaluation_episodes=config.evolution.evaluation_episodes,
        elite_count=config.evolution.elite_count,
        immigrant_count=config.evolution.immigrant_count,
    ) == config.evolution
    assert sized.network == config.network


def test_the_sample_times_real_games_on_real_workers(tmp_path):
    plan = sizing.plan(timed(tmp_path, minutes=0.5), workers=2)
    assert plan.seconds_per_game > 0
    assert plan.games_sampled > 0
    assert plan.mean_steps > 0


def test_a_slow_game_ends_the_sample_on_time(tmp_path):
    # Half a second a game, against the sample's floor of a second.
    config = dataclasses.replace(
        timed(tmp_path, minutes=0.1),
        game=game_reference(Trouble),
        settings=Trouble.Settings(trouble="slow", at=1),
    )
    seconds_per_game, games_sampled, mean_steps = sizing._sample(config, None, 1)
    assert seconds_per_game >= 0.45
    assert games_sampled <= 5
    assert mean_steps == 6


def test_a_game_that_fails_in_the_sample_stops_the_run_with_its_fault(tmp_path):
    config = dataclasses.replace(
        timed(tmp_path, minutes=0.1),
        game=game_reference(Trouble),
        settings=Trouble.Settings(trouble="raise"),
    )
    with pytest.raises(GameFault):
        Trainer(config, run_id="never", seed=1)
    assert not (tmp_path / "runs" / "never").exists()


# ---------------------------------------------------------------------- the run


def test_a_sized_run_is_the_same_run_as_one_given_that_size_by_hand(tmp_path):
    config = make_config(tmp_path)
    plan = a_plan(14, 3, minutes=60.0, elites=2, immigrants=1)
    sized = train(Trainer(
        dataclasses.replace(config, sizing=dataclasses.replace(config.sizing, minutes=60.0)),
        run_id="sized", seed=5, max_generations=3, sized=plan,
    ))
    by_hand = train(Trainer(
        dataclasses.replace(config, evolution=dataclasses.replace(
            config.evolution, population_size=14, evaluation_episodes=3,
            elite_count=2, immigrant_count=1,
        )),
        run_id="by-hand", seed=5, max_generations=3,
    ))
    assert_same_population(sized.population, by_hand.population)
    assert played(sized.run_dir) == played(by_hand.run_dir)


def test_a_run_with_no_time_named_is_not_sized(tmp_path, monkeypatch):
    def not_measured(*args, **kwargs):
        raise AssertionError("sized a run that named no time")

    monkeypatch.setattr(sizing, "plan", not_measured)
    trainer = train(Trainer(make_config(tmp_path), run_id="plain", seed=2, max_generations=1))
    manifest = json.loads((trainer.run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["sizing"] is None
    assert not trainer._time_is_up()


def test_the_manifest_records_the_size_and_where_it_came_from(tmp_path):
    plan = a_plan(14, 3)
    trainer = train(Trainer(timed(tmp_path), run_id="sized", seed=2, max_generations=1, sized=plan))
    manifest = json.loads((trainer.run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["config"]["evolution"]["population_size"] == 14
    assert manifest["config"]["evolution"]["evaluation_episodes"] == 3
    assert manifest["config"]["sizing"]["minutes"] == 10.0
    assert manifest["sizing"]["seconds_per_game"] == plan.seconds_per_game
    assert manifest["sizing"]["kept"] == []


def test_the_plan_is_in_the_runs_log(tmp_path):
    events = EventBus()
    heard = []
    events.subscribe(heard.append, Note)
    train(Trainer(timed(tmp_path), run_id="sized", seed=2, events=events, max_generations=1,
                  sized=a_plan(14, 3)))
    assert any(event.message.startswith("sizing: population 14") for event in heard)


def test_named_settings_reach_the_plan_a_run_makes_for_itself(tmp_path, monkeypatch):
    measured_at(monkeypatch, 1e-6)
    config = timed(tmp_path)
    trainer = Trainer(config, run_id="named", seed=2, max_generations=1,
                      named=[POPULATION, EPISODES])
    trainer.stop()
    assert trainer.config.evolution.population_size == config.evolution.population_size
    assert trainer.config.evolution.evaluation_episodes == config.evolution.evaluation_episodes


# ---------------------------------------------------------------------- the time


def test_every_run_plays_at_least_one_generation_however_long_it_took_to_get_ready(tmp_path):
    finished = []
    events = EventBus()
    events.subscribe(finished.append, RunFinished)
    plan = a_plan(minutes=0.001, setup=5.0)
    trainer = train(Trainer(timed(tmp_path, minutes=0.001), run_id="late", seed=2, events=events,
                            sized=plan))
    assert trainer.population.generation == 1
    assert finished[0].reason == "budget"


def test_a_run_ends_before_a_generation_that_would_not_fit(tmp_path):
    trainer = Trainer(timed(tmp_path, minutes=1.0), run_id="close", seed=2,
                      sized=a_plan(minutes=1.0))
    trainer.population.generation = 3
    trainer._spent = 50.0
    trainer._last_generation_seconds = 5.0
    assert not trainer._time_is_up()
    trainer._last_generation_seconds = 11.0
    assert trainer._time_is_up()
    trainer.stop()


def test_a_run_says_when_its_time_is_up_and_keeps_what_it_reached(tmp_path):
    events = EventBus()
    heard = []
    events.subscribe(heard.append, Note)
    trainer = train(Trainer(timed(tmp_path, minutes=0.001), run_id="up", seed=2, events=events,
                            sized=a_plan(minutes=0.001, setup=0.0)))
    assert any("minutes are up" in event.message for event in heard)
    found = checkpoint.list_checkpoints(trainer.checkpoint_dir)
    assert found[-1].generation == trainer.population.generation


def test_a_limit_on_generations_still_ends_a_run_with_time_left(tmp_path):
    finished = []
    events = EventBus()
    events.subscribe(finished.append, RunFinished)
    trainer = train(Trainer(timed(tmp_path), run_id="short", seed=2, events=events,
                            max_generations=2, sized=a_plan()))
    assert trainer.population.generation == 2
    assert finished[0].reason == "limit"


def test_the_time_spent_waiting_for_a_viewer_does_not_count(tmp_path, monkeypatch):
    trainer = Trainer(timed(tmp_path), run_id="held", seed=2, max_generations=2,
                      sized=a_plan(setup=0.0))
    monkeypatch.setattr(trainer.gate, "wait_at_boundary",
                        lambda *args, **kwargs: time.sleep(1.0))
    train(trainer)
    assert trainer._spent < 1.0


def test_a_run_carried_on_counts_the_time_it_had_already_spent(tmp_path):
    config = timed(tmp_path)
    plan = a_plan(setup=7.0)
    first = train(Trainer(config, run_id="carried", seed=3, max_generations=4, sized=plan))
    newest = checkpoint.list_checkpoints(first.checkpoint_dir)[-1]
    kept = read_summaries(first.run_dir / "history.jsonl")[:newest.generation]

    again = Trainer(first.config, run_id="carried", resume_from=newest.path, carry_on=True,
                    max_generations=4)
    expected = 7.0 + sum(summary["elapsed"] for summary in kept)
    assert again._spent == pytest.approx(expected)
    again.stop()


def test_a_run_carried_on_with_its_time_gone_ends_at_once(tmp_path):
    finished = []
    events = EventBus()
    events.subscribe(finished.append, RunFinished)
    config = timed(tmp_path, minutes=0.001)
    first = train(Trainer(config, run_id="gone", seed=3, sized=a_plan(minutes=0.001, setup=60.0)))
    newest = checkpoint.list_checkpoints(first.checkpoint_dir)[-1]
    again = train(Trainer(first.config, run_id="gone", resume_from=newest.path, carry_on=True,
                          events=events))
    assert again.population.generation == newest.generation
    assert finished[0].reason == "budget"


def test_a_resumed_run_is_a_new_stretch_with_its_own_time(tmp_path):
    first = train(Trainer(timed(tmp_path), run_id="first", seed=3, max_generations=2,
                          sized=a_plan(setup=100.0)))
    newest = checkpoint.list_checkpoints(first.checkpoint_dir)[-1]
    again = Trainer(first.config, run_id="second", resume_from=newest.path, max_generations=3)
    assert again._spent == 0.0
    # Sized once, when it was new.
    assert again.config.evolution == first.config.evolution
    again.stop()


# ---------------------------------------------------------------------- the probe


def test_a_probe_gets_a_share_of_the_time_and_no_more(tmp_path):
    config = timed(tmp_path, minutes=1.0)
    assert sizing.with_probe_limit(config).probe.seconds == pytest.approx(12.0)
    roomy = timed(tmp_path, minutes=100.0)
    assert sizing.with_probe_limit(roomy).probe.seconds == CONFIG.probe.seconds
    assert sizing.with_probe_limit(make_config(tmp_path)).probe == CONFIG.probe


def test_the_run_probes_within_its_time_and_sizes_the_game_as_fixed(tmp_path, monkeypatch):
    calls = measured_at(monkeypatch, 1e-4)
    real = probe.run_guarded
    limits = []

    def spying(config):
        limits.append(config.probe.seconds)
        return real(config)

    monkeypatch.setattr(probe, "run_guarded", spying)
    config = dataclasses.replace(
        timed(tmp_path, minutes=1.0),
        game=game_reference(broken.VARIANTS["cartpole"]["dead"]),
        settings=None,
        probe=dataclasses.replace(CONFIG.probe, enabled=True, episodes=5, search_steps=0),
    )
    trainer = Trainer(config, run_id="probed", seed=2, max_generations=1)
    trainer.stop()
    assert limits == [pytest.approx(12.0)]
    assert calls[0]["fixes"] is not None
    assert calls[0]["fixes"] == trainer.population.fixes
    # The recorded setting stays what was asked for.
    assert trainer.config.probe.seconds == CONFIG.probe.seconds


# ---------------------------------------------------------------------- settings


@pytest.mark.parametrize(("assignments", "words"), [
    ((("sizing.minutes", "-1"),), "sizing.minutes"),
    ((("sizing.minutes", "5"), ("sizing.generations", "0")), "sizing.generations"),
])
def test_sizing_settings_that_cannot_work_are_refused(tmp_path, assignments, words):
    config = settings.apply(make_config(tmp_path), assignments)
    with pytest.raises(SettingsError) as refused:
        settings.check(config)
    assert any(words in problem for problem in refused.value.problems)


def test_generations_with_no_time_named_have_no_effect(tmp_path):
    config = settings.apply(make_config(tmp_path), [("sizing.generations", "50")])
    with pytest.raises(SettingsError) as refused:
        settings.check(config, named=["sizing.generations"])
    assert any("sizing.minutes is 0" in problem for problem in refused.value.problems)


def test_the_form_and_the_settings_command_list_the_time(tmp_path, capsys):
    assert "sizing.minutes" in {row["name"] for row in settings.table(make_config(tmp_path))}
    assert main(["settings"]) == 0
    assert "sizing.minutes" in capsys.readouterr().out


# ---------------------------------------------------------------------- command line


def test_minutes_on_the_command_line_fit_the_run_around_what_was_named(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main([
        "train", "--game", "cartpole", "--headless", "--minutes", "0.02", "--workers", "1",
        "--population", "20", "--seed", "4",
        "--set", f"paths.runs_dir={runs}", "--report-every", "1000",
    ]) == 0
    out = capsys.readouterr().out
    assert "minutes are up" in out
    run_dir = next(runs.iterdir())
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["config"]["evolution"]["population_size"] == 20
    assert POPULATION in manifest["sizing"]["kept"]
    assert EPISODES not in manifest["sizing"]["kept"]


def test_minutes_that_cannot_work_are_refused_on_the_command_line(tmp_path, capsys):
    assert main(["train", "--headless", "--minutes", "-2",
                 "--set", f"paths.runs_dir={tmp_path}"]) == 1
    assert "sizing.minutes" in capsys.readouterr().out


# ---------------------------------------------------------------------- dashboard


def test_a_preview_of_a_share_changes_nobodys_share():
    budget = WorkerBudget(6)
    budget.join("a")
    budget.join("b")
    assert budget.preview() == 2
    assert budget.preview(1) == 1
    assert budget.share("a") == 3
    assert budget.jobs == 2


def test_the_dashboard_sizes_a_run_and_marks_it_finished_when_time_is_up(tmp_path):
    lab = Lab(make_config(tmp_path), can_start=True)
    try:
        reply = lab.start("cartpole", {"sizing.minutes": "0.01", POPULATION: "12",
                                       EPISODES: "1", ELITES: "2", IMMIGRANTS: "1"})
        assert reply["ok"], reply
        wait_for(lambda: not lab.live)
    finally:
        lab.close()
    run_dir = tmp_path / "runs" / reply["run"]
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["sizing"]["minutes"] == 0.01
    assert manifest["config"]["evolution"]["population_size"] == 12
    assert jobs.read(run_dir)["state"] == "finished"


def test_the_dashboard_reports_a_game_that_fails_in_the_sample(tmp_path):
    lab = Lab(make_config(tmp_path), can_start=True)
    try:
        reply = lab.start(game_reference(Trouble), {
            "trouble": "raise", "sizing.minutes": "0.1",
        })
        assert reply.get("problems"), reply
        assert lab.live == []
    finally:
        lab.close()
