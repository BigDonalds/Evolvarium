"""Settings by name: set from text, and refused before a run when they cannot train."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_contract import HERD, Herd, herd_config

from evolvarium import games, held_out, settings
from evolvarium.cli import main
from evolvarium.config import CONFIG
from evolvarium.games.snake import SnakeSettings
from evolvarium.settings import SettingsError
from evolvarium.trainer import Trainer


def configure(*assignments, config=CONFIG):
    return settings.apply(config, assignments)


def refused(config, **kwargs) -> list[str]:
    with pytest.raises(SettingsError) as raised:
        settings.check(config, **kwargs)
    return raised.value.problems


# ------------------------------------------------------------------ names


def test_the_defaults_train_with_and_without_a_dashboard():
    for config in (games.resolve(CONFIG), herd_config()):
        settings.check(config)
        settings.check(config, dashboard=False)


def test_every_setting_has_one_name_the_game_s_first():
    names = list(settings.values(games.resolve(CONFIG)))
    assert names[0].partition(".")[0] not in ("network", "evolution")
    assert {"board.width", "features.spec", "evolution.population_size",
            "network.hidden_layers", "server.port"} <= set(names)
    assert len(names) == len(set(names))

    herd = settings.values(herd_config())
    assert herd["start"] == 4.0 and herd["max_steps"] == 20


def test_a_game_without_choices_has_only_the_engine_s():
    assert settings.choices(HERD) == settings.ENGINE_CHOICES
    assert "features.spec" in settings.choices("snake")


# ------------------------------------------------------------------ text


@pytest.mark.parametrize(("text", "like", "value"), [
    ("12", 9, 12),
    ("0.5", 1.0, 0.5),
    ("3", 1.0, 3.0),
    ("yes", False, True),
    ("Off", True, False),
    ("32, 16", (64,), (32, 16)),
    ("", (64,), ()),
    ("relu,tanh", ("tanh",), ("relu", "tanh")),
    ("runs/a", Path("runs"), Path("runs/a")),
])
def test_text_is_read_as_the_setting_s_type(text, like, value):
    parsed = settings.parse(text, like)
    assert parsed == value and type(parsed) is type(value)


@pytest.mark.parametrize(("text", "like"), [("twelve", 9), ("1.5", 9), ("maybe", True),
                                            ("1,x", (1,))])
def test_text_that_is_not_the_setting_s_type_is_refused(text, like):
    with pytest.raises(ValueError):
        settings.parse(text, like)


def test_a_value_is_shown_the_way_it_is_read():
    for value in (True, False, (32, 16), ("relu", "tanh"), 0.25):
        assert settings.parse(settings.show(value), value) == value


# ------------------------------------------------------------------ apply


def test_game_and_engine_settings_are_set_by_name():
    config = configure(
        ("board.width", "12"), ("evolution.population_size", 40),
        ("network.hidden_layers", "32,16,8"), ("network.hidden_activations", "relu,tanh,tanh"),
    )
    assert config.settings.board.width == 12
    assert config.settings.board.height == SnakeSettings().board.height
    assert config.evolution.population_size == 40
    assert config.network.hidden_layers == (32, 16, 8)
    assert CONFIG.evolution.population_size == 150

    herd = configure(("start", "2.5"), config=herd_config())
    assert herd.settings.start == 2.5


@pytest.mark.parametrize(("name", "hint"), [
    ("evolution.tournment_size", "did you mean evolution.tournament_size?"),
    ("width", "did you mean board.width?"),
    ("zzzz", "the settings command lists them"),
])
def test_an_unknown_name_comes_back_with_the_closest_one(name, hint):
    with pytest.raises(SettingsError) as raised:
        configure((name, "1"))
    assert raised.value.problems == [f"no setting named {name}; {hint}"]


def test_every_bad_assignment_is_reported_at_once():
    with pytest.raises(SettingsError) as raised:
        configure(("bad", "1"), ("board.width", "wide"), ("evolution.workers", "2"),
                  ("evolution.workers", "3"))
    problems = raised.value.problems
    assert len(problems) == 3
    assert problems[0].startswith("no setting named bad")
    assert problems[1] == "board.width takes a whole number, not 'wide'"
    assert problems[2] == "evolution.workers is set more than once"


# ------------------------------------------------------------------ check


def test_a_value_of_the_wrong_type_is_refused_before_anything_else():
    evolution = dataclasses.replace(CONFIG.evolution, population_size=1.5, workers=True,
                                    mutation_rate=float("nan"), elite_count=-1)
    problems = refused(games.resolve(CONFIG.with_overrides(evolution=evolution)))
    assert problems == [
        "evolution.population_size takes a whole number, not 1.5",
        "evolution.mutation_rate takes a number, not nan",
        "evolution.workers takes a whole number, not True",
    ]


def test_settings_a_game_does_not_play_under_are_refused():
    config = dataclasses.replace(herd_config(), settings=SnakeSettings())
    assert refused(config) == [f"{HERD} plays under HerdSettings, not SnakeSettings"]


def test_a_value_outside_its_choices_comes_back_with_the_closest_one():
    config = configure(("evolution.selection", "rnk"),
                       ("network.hidden_activations", "tanh,rleu"))
    problems = refused(config)
    assert problems[0].endswith("not 'rleu'; did you mean relu?")
    assert problems[1] == (
        "evolution.selection must be one of tournament, rank, roulette, not 'rnk'; "
        "did you mean rank?"
    )


def test_choices_are_reported_with_every_other_problem():
    config = configure(("features.spec", "egocentric_v9"), ("board.width", "1"),
                       ("evolution.elite_count", "200"))
    problems = refused(config)
    assert len(problems) == 3
    assert problems[0].startswith("features.spec must be one of")


@pytest.mark.parametrize(("assignments", "problem"), [
    ((("evolution.population_size", "10"), ("evolution.elite_count", "6"),
      ("evolution.immigrant_count", "4")),
     "evolution.elite_count and evolution.immigrant_count take 10 of a population of 10, "
     "leaving no room for offspring"),
    ((("evolution.mutation_rate", "0.5"),),
     "evolution.mutation_rate of 0.5 is above evolution.max_mutation_rate, 0.35, which caps it"),
    ((("network.hidden_layers", "32,16"), ("network.hidden_activations", "tanh")),
     "network.hidden_activations names 1 activation for 2 hidden layers; it needs one for each"),
    ((("network.hidden_layers", "32,0"),),
     "network.hidden_layers needs at least 1 unit in every layer, not 32,0"),
    ((("evolution.tournament_size", "1"),),
     "evolution.tournament_size must be between 2 and 150, not 1"),
    ((("evolution.crossover_rate", "1.5"),),
     "evolution.crossover_rate must be between 0 and 1, not 1.5"),
    ((("evolution.population_size", "1"), ("evolution.elite_count", "0"),
      ("evolution.immigrant_count", "0")),
     "evolution.population_size must be at least 2, not 1"),
    ((("network.weight_init_scale", "0"),), "network.weight_init_scale must be above 0, not 0.0"),
    ((("history.memory_window", "0"),), "history.memory_window must be at least 1, not 0"),
    ((("server.port", "70000"),), "server.port must be between 1 and 65535, not 70000"),
    ((("server.host", ""),), "server.host is empty"),
])
def test_engine_values_that_cannot_train_are_refused(assignments, problem):
    assert problem in refused(configure(*assignments))


@pytest.mark.parametrize(("assignments", "problem"), [
    ((("board.height", "14"), ("board.initial_length", "40")),
     "board.initial_length of 40 does not fit below the centre of a board 14 high, "
     "which has room for 7"),
    ((("board.width", "1"),), "board.width must be at least 2, not 1"),
    ((("fitness.food_reward", "-1"),), "fitness.food_reward must be at least 0, not -1.0"),
    ((("fitness.food_exponent", "0"),), "fitness.food_exponent must be above 0, not 0.0"),
    ((("features.spec", "egocentric_v1"), ("features.flood_fill_limit", "0")),
     "features.flood_fill_limit must be at least 1, not 0"),
])
def test_snake_values_that_cannot_train_are_refused(assignments, problem):
    assert problem in refused(configure(*assignments))


@pytest.mark.parametrize(("assignments", "reason"), [
    ((("evolution.selection", "rank"), ("evolution.tournament_size", "5")),
     "evolution.selection is rank"),
    ((("evolution.crossover_rate", "0"), ("evolution.crossover_method", "uniform")),
     "evolution.crossover_rate is 0"),
    ((("evolution.polynomial_eta", "10"),), "evolution.mutation_method is gaussian"),
    ((("evolution.stagnation_step", "0"), ("evolution.max_mutation_rate", "0.5")),
     "evolution.stagnation_step is 0"),
    ((("evolution.mutation_strength", "0"), ("evolution.mutation_rate", "0.2")),
     "evolution.mutation_strength is 0"),
    ((("evolution.target_species", "4"),), "evolution.use_speciation is false"),
    ((("checkpoint.save_every", "0"), ("checkpoint.keep_last", "3")),
     "checkpoint.save_every is 0"),
    ((("held_out.episodes", "10"),), "training never plays the held-out games"),
    ((("features.spec", "egocentric_v2"), ("features.flood_fill_limit", "50")),
     "features.spec is egocentric_v2 and only egocentric_v1 reads it"),
    ((("fitness.food_reward", "0"), ("fitness.food_exponent", "2")),
     "fitness.food_reward is 0"),
])
def test_a_named_setting_with_no_effect_is_refused(assignments, reason):
    config = configure(*assignments)
    name = assignments[-1][0]
    problems = refused(config, named=[name for name, _ in assignments])
    assert problems == [f"{name} has no effect, since {reason}"]
    settings.check(config)


def test_dashboard_settings_have_no_effect_without_one():
    config = configure(("server.port", "9000"), ("paths.exports_dir", "out"))
    named = ["server.port", "paths.exports_dir"]
    settings.check(config, named=named)
    assert refused(config, named=named, dashboard=False) == [
        "server.port has no effect, since the run has no dashboard",
        "paths.exports_dir has no effect, since only the dashboard exports during a run",
    ]


def test_a_setting_without_effect_is_not_checked_for_range():
    config = configure(("evolution.selection", "rank"), ("evolution.tournament_size", "0"),
                       ("evolution.use_speciation", "false"),
                       ("evolution.compatibility_threshold", "-1"))
    settings.check(config)


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class ClashSettings:
    evolution: int = 1


class Clash(Herd):
    Settings = ClashSettings


class Loose(Herd):
    Settings = dict


@pytest.mark.parametrize(("name", "message"), [
    ("test_settings:Clash", "a setting named 'evolution', which the engine's settings use"),
    ("test_settings:Loose", "settings that are not a dataclass"),
    ("snak", "did you mean 'snake'?"),
])
def test_the_registry_refuses_games_whose_settings_cannot_be_named(name, message):
    with pytest.raises(games.UnknownGame, match=message):
        games.get(name)


# ------------------------------------------------------------------ runs


def test_a_run_with_bad_settings_stops_before_it_has_a_directory(tmp_path):
    config = configure(("evolution.elite_count", "200"),
                       ("paths.runs_dir", str(tmp_path / "runs")))
    with pytest.raises(SettingsError, match="leaving no room for offspring"):
        Trainer(config, seed=1, run_id="run-bad")
    assert not (tmp_path / "runs").exists()


def test_search_settings_leave_out_settings_the_engine_no_longer_has():
    recorded = {"evolution": {"population_size": 40, "min_mutation_rate": 0.01, "workers": 4}}
    assert held_out.search_settings(recorded)["evolution"] == {"population_size": 40}


# ------------------------------------------------------------------ command line


@pytest.mark.parametrize(("argv", "problems"), [
    (["--set", "bad=1", "board.width=wide"],
     ["no setting named bad", "board.width takes a whole number, not 'wide'"]),
    (["--port", "9000"], ["server.port has no effect, since the run has no dashboard"]),
    (["--open"], ["--open has no effect with --headless, which starts no dashboard"]),
    (["--population", "40", "--set", "evolution.population_size=50"],
     ["evolution.population_size is set more than once"]),
    (["--game", "snak"], ["did you mean 'snake'?"]),
    (["--game", HERD, "--set", "start=-"], ["start takes a number, not '-'"]),
])
def test_train_refuses_settings_before_it_starts(argv, problems, capsys):
    assert main(["train", "--headless", *argv]) == 1
    printed = capsys.readouterr().out
    assert "cannot train:" in printed
    for problem in problems:
        assert problem in printed


def test_set_needs_a_name_and_a_value(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["train", "--headless", "--set", "board.width"])
    assert raised.value.code == 2
    assert "'board.width' is not NAME=VALUE" in capsys.readouterr().err


def test_the_settings_command_lists_values_and_choices(capsys):
    assert main(["settings"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.split()[:2] == ["board.width", str(SnakeSettings().board.width)]
               for line in lines if line.strip())
    assert any(line.split()[0] == "evolution.selection"
               and line.endswith("(tournament, rank, roulette)")
               for line in lines if line.strip())

    assert main(["settings", "--game", HERD]) == 0
    assert "max_steps" in capsys.readouterr().out
    assert main(["settings", "--game", "snak"]) == 1
