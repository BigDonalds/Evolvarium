"""The games the engine can train, found by name.

A built-in game is named by its short name. Any other game is named
"package.module:Class", and is imported from wherever Python finds that module,
so a game can live outside this package. Worker processes import it the same
way, which needs the module importable from a fresh interpreter too.
"""

from __future__ import annotations

import importlib

from ..config import Config
from ..contract import DIRECTIONS, Game

_BUILT_IN = {
    "snake": "evolvarium.games.snake:SnakeGame",
}


class UnknownGame(ValueError):
    """A game name that names no game."""


def names() -> list[str]:
    return sorted(_BUILT_IN)


def get(name: str) -> type[Game]:
    """The game class for a built-in name or a "module:Class" reference."""
    reference = _BUILT_IN.get(name, name)
    module_name, _, class_name = reference.partition(":")
    if not class_name:
        raise UnknownGame(f"unknown game {name!r}; built in: {', '.join(names())}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise UnknownGame(f"cannot import game {name!r}: {error}") from error
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Game)):
        raise UnknownGame(f"{reference!r} is not a Game subclass")
    if cls.direction not in DIRECTIONS:
        raise UnknownGame(
            f"{reference} has direction {cls.direction!r}; it must be one of {DIRECTIONS}"
        )
    return cls


def create(name: str, settings=None) -> Game:
    return get(name)(settings)


def settings_for(config: Config):
    """The settings a configuration plays its game under."""
    if config.settings is not None:
        return config.settings
    return get(config.game).Settings()


def resolve(config: Config) -> Config:
    """`config` with its game's settings filled in, so a record of it is complete."""
    if config.settings is not None:
        return config
    return config.with_overrides(settings=settings_for(config))
