"""Every setting by name: listing them, setting them from text and checking them.

A game's settings are named by their path inside its settings, such as
"board.width" for Snake, and the engine's by section, such as
"evolution.population_size". `apply` sets them from text the way the command
line gives them. `check` finds everything that would stop a configuration from
training, or train it on something other than what was asked for, before the
run starts rather than partway through it.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from pathlib import Path

from . import games
from .config import SECTIONS, Config
from .contract import Problems, closest, override
from .evolution import operators, selection
from .nn.activations import ACTIVATIONS

ENGINE_CHOICES = {
    "network.hidden_activations": tuple(ACTIVATIONS),
    "evolution.selection": selection.METHODS,
    "evolution.crossover_method": operators.BLENDS,
    "evolution.mutation_method": operators.MUTATIONS,
}

_TRUE = ("true", "yes", "on", "1")
_FALSE = ("false", "no", "off", "0")


class SettingsError(ValueError):
    """Settings a run cannot start with, holding one sentence for each problem."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# ------------------------------------------------------------------ names


def values(config: Config) -> dict[str, object]:
    """Every setting `config` trains under, by name, the game's before the engine's."""
    found = _leaves(games.settings_for(config))
    for section in SECTIONS:
        found.update(_leaves(getattr(config, section), f"{section}."))
    return found


def choices(game: str) -> dict[str, tuple]:
    """The allowed values of every setting that has a fixed set, by name.

    For a setting that holds several values, each of them must be one of these.
    """
    return {**games.get(game).setting_choices(), **ENGINE_CHOICES}


def table(config: Config, leave_out: Iterable[str] = ()) -> list[dict]:
    """Every setting with what a form needs to change it, in the order of `values`.

    Each row holds the name, the value written as `parse` reads it, the kind of
    value in words, the input it takes ("flag", "whole", "number" or "text"),
    whether it holds several values, and its choices if it has a fixed set.
    Settings in the sections named in `leave_out` are not listed.
    """
    allowed = choices(config.game)
    rows = []
    for name, value in values(config).items():
        section = name.partition(".")[0]
        if section in leave_out:
            continue
        rows.append({
            "name": name,
            "group": section if section in SECTIONS else "game",
            "value": show(value),
            "kind": _kind(value),
            "input": _input(value),
            "several": isinstance(value, tuple),
            "choices": [show(option) for option in allowed[name]] if name in allowed else None,
        })
    return rows


def _input(like) -> str:
    if isinstance(like, bool):
        return "flag"
    if isinstance(like, Integral):
        return "whole"
    if isinstance(like, Real):
        return "number"
    return "text"


def _leaves(settings, prefix: str = "") -> dict[str, object]:
    found = {}
    for field in dataclasses.fields(settings):
        value = getattr(settings, field.name)
        if dataclasses.is_dataclass(value):
            found.update(_leaves(value, f"{prefix}{field.name}."))
        else:
            found[f"{prefix}{field.name}"] = value
    return found


# ------------------------------------------------------------------ text


def parse(text: str, like):
    """`text` read as a value of the same type as `like`.

    Several values are separated by commas, and a blank gives none.
    """
    if isinstance(like, tuple):
        text = text.strip()
        item = like[0] if like else ""
        return tuple(parse(part, item) for part in text.split(",")) if text else ()
    if isinstance(like, bool):
        word = text.strip().lower()
        if word not in _TRUE + _FALSE:
            raise ValueError(text)
        return word in _TRUE
    if isinstance(like, Integral):
        return int(text)
    if isinstance(like, Real):
        return float(text)
    if isinstance(like, Path):
        return Path(text)
    if like is None:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def show(value) -> str:
    """`value` written the way `parse` reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ",".join(show(item) for item in value)
    return str(value)


def _kind(like) -> str:
    if isinstance(like, tuple):
        return f"a comma-separated list of {_kind(like[0]) if like else 'text'}"
    if isinstance(like, bool):
        return "true or false"
    if isinstance(like, Integral):
        return "a whole number"
    if isinstance(like, Real):
        return "a number"
    if isinstance(like, Path):
        return "a path"
    return "text" if isinstance(like, str) else type(like).__name__


def _fits(value, like) -> bool:
    if isinstance(like, tuple):
        if not isinstance(value, tuple):
            return False
        return not like or all(_fits(item, like[0]) for item in value)
    if isinstance(like, bool):
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if isinstance(like, Integral):
        return isinstance(value, Integral)
    if isinstance(like, Real):
        return isinstance(value, Real) and math.isfinite(value)
    if isinstance(like, Path):
        return isinstance(value, Path)
    return like is None or isinstance(value, type(like))


# ------------------------------------------------------------------ apply


def apply(config: Config, assignments: Iterable[tuple[str, object]]) -> Config:
    """`config` with each named setting replaced, its game's settings filled in.

    A value given as text is read as the setting's type. Every name that is not
    a setting and every value that cannot be read is reported at once, each
    unknown name with the closest one that exists.
    """
    config = games.resolve(config)
    known = values(config)
    problems = []
    seen = set()
    for name, value in assignments:
        if name not in known:
            match = closest(name, known)
            hint = f"did you mean {match}?" if match else "the settings command lists them"
            problems.append(f"no setting named {name}; {hint}")
            continue
        if name in seen:
            problems.append(f"{name} is set more than once")
            continue
        seen.add(name)
        like = known[name]
        if isinstance(value, str) and not isinstance(like, str):
            try:
                value = parse(value, like)
            except ValueError:
                problems.append(f"{name} takes {_kind(like)}, not {value!r}")
                continue
        config = _replace(config, name, value)
    if problems:
        raise SettingsError(problems)
    return config


def _replace(config: Config, name: str, value) -> Config:
    if name.partition(".")[0] in SECTIONS:
        return override(config, name, value)
    return dataclasses.replace(config, settings=override(config.settings, name, value))


# ------------------------------------------------------------------ check


def check(config: Config, *, named: Iterable[str] = (), dashboard: bool = True) -> None:
    """Raise SettingsError with everything that keeps `config` from training as given.

    Every value is checked for its type, its range and its choices, and against
    the others it depends on. Of the settings in `named`, the ones someone asked
    for by name, any that the rest leave without effect is refused as well.
    `dashboard` is whether the run serves one.
    """
    game = games.get(config.game)
    settings = games.settings_for(config)
    if not isinstance(settings, game.Settings):
        raise SettingsError([
            f"{config.game} plays under {game.Settings.__name__}, "
            f"not {type(settings).__name__}"
        ])
    current = values(config)
    defaults = values(Config(game=config.game))

    problems = Problems()
    for name, value in current.items():
        if not _fits(value, defaults[name]):
            problems.append(f"{name} takes {_kind(defaults[name])}, not {value!r}")
    if problems:
        raise SettingsError(problems)

    for name, allowed in choices(config.game).items():
        value = current[name]
        for item in value if isinstance(value, tuple) else (value,):
            if item not in allowed:
                problems.append(_not_a_choice(name, item, allowed))
    unused = {**game.unused_settings(settings), **_unused(config, dashboard)}
    problems.extend(game.setting_problems(settings))
    problems.extend(_problems(config, unused))
    for name in dict.fromkeys(named):
        if name in unused:
            problems.append(f"{name} has no effect, since {unused[name]}")
    if problems:
        raise SettingsError(problems)


def _not_a_choice(name: str, value, allowed: Sequence) -> str:
    message = f"{name} must be one of {', '.join(map(str, allowed))}, not {value!r}"
    match = closest(str(value), [str(option) for option in allowed])
    return f"{message}; did you mean {match}?" if match else message


def _unused(config: Config, dashboard: bool) -> dict[str, str]:
    """The engine's settings that the others leave without effect, with the reason."""
    evolution = config.evolution
    unused = {}
    if evolution.selection != "tournament":
        unused["evolution.tournament_size"] = f"evolution.selection is {evolution.selection}"
    if evolution.crossover_rate == 0:
        unused["evolution.crossover_method"] = "evolution.crossover_rate is 0"
    if evolution.mutation_method != "polynomial":
        unused["evolution.polynomial_eta"] = (
            f"evolution.mutation_method is {evolution.mutation_method}"
        )
    if evolution.stagnation_step == 0:
        unused["evolution.max_mutation_rate"] = "evolution.stagnation_step is 0"
    if evolution.mutation_strength == 0:
        for name in ("mutation_rate", "max_mutation_rate", "stagnation_step",
                     "mutation_method", "polynomial_eta"):
            unused[f"evolution.{name}"] = "evolution.mutation_strength is 0"
    if not evolution.use_speciation:
        for name in ("compatibility_threshold", "target_species"):
            unused[f"evolution.{name}"] = "evolution.use_speciation is false"
    if config.checkpoint.save_every == 0:
        unused["checkpoint.keep_last"] = "checkpoint.save_every is 0"
    for name in _names(config.held_out):
        unused[f"held_out.{name}"] = "training never plays the held-out games"
    if not dashboard:
        for name in _names(config.server):
            unused[f"server.{name}"] = "the run has no dashboard"
        unused["paths.exports_dir"] = "only the dashboard exports during a run"
    return unused


def _problems(config: Config, unused: dict[str, str]) -> Problems:
    """Values of the engine's settings that cannot train, skipping unused ones."""
    problems = Problems()
    network, evolution = config.network, config.evolution

    layers, activations = len(network.hidden_layers), len(network.hidden_activations)
    if activations != layers:
        problems.append(
            f"network.hidden_activations names {activations} activation"
            f"{'s' if activations != 1 else ''} for {layers} hidden layer"
            f"{'s' if layers != 1 else ''}; it needs one for each"
        )
    if any(size < 1 for size in network.hidden_layers):
        problems.append(
            f"network.hidden_layers needs at least 1 unit in every layer, "
            f"not {show(network.hidden_layers)}"
        )
    problems.above("network.weight_init_scale", network.weight_init_scale, 0)
    problems.above("network.output_init_scale", network.output_init_scale, 0)

    problems.at_least("evolution.population_size", evolution.population_size, 2)
    problems.at_least("evolution.elite_count", evolution.elite_count, 0)
    problems.at_least("evolution.immigrant_count", evolution.immigrant_count, 0)
    carried = evolution.elite_count + evolution.immigrant_count
    if carried >= evolution.population_size:
        problems.append(
            f"evolution.elite_count and evolution.immigrant_count take {carried} of a "
            f"population of {evolution.population_size}, leaving no room for offspring"
        )
    problems.at_least("evolution.evaluation_episodes", evolution.evaluation_episodes, 1)
    if "evolution.tournament_size" not in unused and evolution.population_size >= 2:
        problems.between("evolution.tournament_size", evolution.tournament_size,
                         2, evolution.population_size)
    for name in ("crossover_rate", "mutation_rate", "max_mutation_rate", "exploration_rate"):
        if f"evolution.{name}" not in unused:
            problems.between(f"evolution.{name}", getattr(evolution, name), 0, 1)
    # The cap applies from the first generation, so a rate above it never runs.
    if ("evolution.mutation_rate" not in unused
            and evolution.mutation_rate > evolution.max_mutation_rate):
        problems.append(
            f"evolution.mutation_rate of {evolution.mutation_rate} is above "
            f"evolution.max_mutation_rate, {evolution.max_mutation_rate}, which caps it"
        )
    problems.at_least("evolution.mutation_strength", evolution.mutation_strength, 0)
    for name, low in (("stagnation_step", 0), ("polynomial_eta", 0), ("target_species", 1)):
        if f"evolution.{name}" not in unused:
            problems.at_least(f"evolution.{name}", getattr(evolution, name), low)
    if "evolution.compatibility_threshold" not in unused:
        problems.above("evolution.compatibility_threshold", evolution.compatibility_threshold, 0)
    problems.at_least("evolution.workers", evolution.workers, 0)

    history = config.history
    problems.at_least("history.memory_window", history.memory_window, 1)
    for name in ("replay_retention", "replay_steps", "agents_per_generation"):
        problems.at_least(f"history.{name}", getattr(history, name), 0)
    problems.at_least("checkpoint.save_every", config.checkpoint.save_every, 0)
    if "checkpoint.keep_last" not in unused:
        problems.at_least("checkpoint.keep_last", config.checkpoint.keep_last, 0)

    server = config.server
    if "server.host" not in unused and not server.host:
        problems.append("server.host is empty")
    if "server.port" not in unused:
        problems.between("server.port", server.port, 1, 65535)
    if "server.viewer_timeout" not in unused:
        problems.above("server.viewer_timeout", server.viewer_timeout, 0)
    return problems


def _names(section) -> list[str]:
    return [field.name for field in dataclasses.fields(section)]
