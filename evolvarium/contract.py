"""What a game provides so the engine can train it.

A game describes itself in plain terms: the inputs each agent sees, the actions
it can take, how an episode ends and whether a higher or a lower score is
better. The engine sizes the network from that description and turns the
network's outputs into actions with it, so nothing outside a game's own package
needs to know which game is being trained.

    class Coin(Game):
        name = "coin"
        observation = Observation([Number("distance"), Category("side", ("left", "right"))])
        actions = Actions(Choice("move", ("left", "stay", "right")))

        def reset(self, seed, *, record=False): ...
        def observe(self, agent): return {"distance": 3.0, "side": "left"}
        def step(self, actions): ...   # actions == {"agent": "stay"}

An episode is reset with a seed, and the same seed must give the same episode.
Each step the engine asks every agent named in `agents` for its observation,
picks an action for each, then hands all of them to `step` together, so a group
of agents shares one episode. `over`, `end`, `score` and `steps` are read once
the episode is over.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

DIRECTIONS = ("higher", "lower")


# ------------------------------------------------------------------- inputs


class Number:
    """An input the network reads as it is."""

    __slots__ = ("name",)
    size = 1

    def __init__(self, name: str):
        self.name = name

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name,)

    def describe(self) -> dict:
        return {"name": self.name, "kind": "number"}


class Category:
    """An input that takes one of a fixed set of values.

    The network sees one column per value, set to 1 for the current value and 0
    for the rest, so no value reads as larger than another.
    """

    __slots__ = ("name", "values", "size", "_index")

    def __init__(self, name: str, values: Sequence):
        values = tuple(values)
        if len(values) < 2:
            raise ValueError(f"category {name!r} needs at least two values")
        if len(set(values)) != len(values):
            raise ValueError(f"category {name!r} lists a value twice")
        self.name = name
        self.values = values
        self.size = len(values)
        self._index = {value: i for i, value in enumerate(values)}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}={value}" for value in self.values)

    def index(self, value) -> int:
        try:
            return self._index[value]
        except (KeyError, TypeError):
            raise ValueError(
                f"input {self.name!r} is {value!r}; it takes one of {list(self.values)}"
            ) from None

    def describe(self) -> dict:
        return {"name": self.name, "kind": "category", "values": list(self.values)}


class Observation:
    """The inputs one agent sees each step, in a fixed order.

    `observe` may return the values in that order or as a mapping from input
    name to value. A category is given as the value itself and encoded here.
    """

    def __init__(self, inputs: Sequence[Number | Category | str]):
        self.inputs = tuple(Number(i) if isinstance(i, str) else i for i in inputs)
        if not self.inputs:
            raise ValueError("an observation needs at least one input")
        names = [name for i in self.inputs for name in i.names]
        if len({i.name for i in self.inputs}) != len(self.inputs) or len(set(names)) != len(names):
            raise ValueError("observation inputs need distinct names")
        self.names = tuple(names)
        self.size = len(names)
        self._shape = (self.size,)
        self._plain = all(isinstance(i, Number) for i in self.inputs)

    def encode(self, values) -> np.ndarray:
        """The network's input vector for one observation."""
        if isinstance(values, Mapping):
            values = self._ordered(values)
        if self._plain:
            array = np.asarray(values, dtype=np.float32)
            if array.shape != self._shape:
                raise ValueError(
                    f"observation has shape {array.shape}; the game declares {self.size} inputs"
                )
            return array
        if len(values) != len(self.inputs):
            raise ValueError(
                f"observation has {len(values)} values; the game declares {len(self.inputs)} inputs"
            )
        array = np.zeros(self.size, dtype=np.float32)
        column = 0
        for spec, value in zip(self.inputs, values, strict=True):
            if isinstance(spec, Category):
                array[column + spec.index(value)] = 1.0
            else:
                array[column] = value
            column += spec.size
        return array

    def _ordered(self, values: Mapping) -> list:
        missing = [i.name for i in self.inputs if i.name not in values]
        if missing:
            raise ValueError(f"observation is missing {missing}")
        return [values[i.name] for i in self.inputs]

    def describe(self) -> list[dict]:
        return [i.describe() for i in self.inputs]


# ------------------------------------------------------------------ actions


class Choice:
    """Pick one of a fixed set of options.

    `options` is a list of the values to deliver, or a count, which delivers
    0 to count - 1. The network has one output per option and the highest wins.
    """

    __slots__ = ("name", "options", "labels", "size")

    def __init__(self, name: str, options: Sequence | int):
        if isinstance(options, int):
            options = tuple(range(options))
        options = tuple(options)
        if len(options) < 2:
            raise ValueError(f"choice {name!r} needs at least two options")
        self.name = name
        self.options = options
        self.labels = tuple(str(option) for option in options)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"choice {name!r} lists an option twice")
        self.size = len(options)

    def decode(self, outputs: np.ndarray):
        return self.options[int(np.argmax(outputs))]

    def explore(self, rng: np.random.Generator):
        return self.options[int(rng.integers(self.size))]

    def describe(self) -> dict:
        return {"name": self.name, "kind": "choice", "options": list(self.labels)}


class Range:
    """A value between `low` and `high`, from one network output.

    The output is squashed with tanh and stretched onto the range, so every
    output lands inside it and the middle of the range is where an untrained
    network starts.
    """

    __slots__ = ("name", "low", "high")
    size = 1

    def __init__(self, name: str, low: float = -1.0, high: float = 1.0):
        if not low < high:
            raise ValueError(f"range {name!r} needs low below high, got {low} and {high}")
        self.name = name
        self.low = float(low)
        self.high = float(high)

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.name,)

    def decode(self, outputs: np.ndarray) -> float:
        squashed = (float(np.tanh(outputs[0])) + 1.0) / 2.0
        return self.low + squashed * (self.high - self.low)

    def explore(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def describe(self) -> dict:
        return {"name": self.name, "kind": "range", "low": self.low, "high": self.high}


class Actions:
    """Everything one agent does in a step.

    With one part the agent's action is that part's value. With several it is a
    dict from part name to value, so a game can ask for a direction and a speed
    in the same step.
    """

    def __init__(self, *parts: Choice | Range):
        if not parts:
            raise ValueError("actions need at least one part")
        if len({p.name for p in parts}) != len(parts):
            raise ValueError("action parts need distinct names")
        self.parts = parts
        self.size = sum(p.size for p in parts)
        self._slices = []
        start = 0
        for part in parts:
            self._slices.append(slice(start, start + part.size))
            start += part.size
        self._single = parts[0] if len(parts) == 1 else None

    @property
    def names(self) -> tuple[str, ...]:
        """One name per network output."""
        if self._single is not None:
            return self._single.labels
        names: list[str] = []
        for part in self.parts:
            if isinstance(part, Choice):
                names.extend(f"{part.name}={label}" for label in part.labels)
            else:
                names.append(part.name)
        return tuple(names)

    def decode(self, outputs: np.ndarray):
        if self._single is not None:
            return self._single.decode(outputs)
        return {
            part.name: part.decode(outputs[where])
            for part, where in zip(self.parts, self._slices, strict=True)
        }

    def explore(self, rng: np.random.Generator):
        """A random action, drawn part by part in order."""
        if self._single is not None:
            return self._single.explore(rng)
        return {part.name: part.explore(rng) for part in self.parts}

    def describe(self) -> list[dict]:
        return [p.describe() for p in self.parts]


# ------------------------------------------------------------------ results


@dataclass
class Outcome:
    """One finished episode, as the engine keeps it. Plain data."""

    score: float
    steps: int
    end: str
    fitness: float
    terms: dict[str, float]
    signals: dict[str, float]
    replay: dict | None = None


@dataclass
class Milestone:
    """Something a run achieved, and the generation it first happened in."""

    key: str
    label: str
    detail: str
    generation: int | None

    @property
    def reached(self) -> bool:
        return self.generation is not None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "detail": self.detail,
            "generation": self.generation,
            "unlocked": self.reached,
        }


# --------------------------------------------------------------------- game


@dataclass(frozen=True)
class NoSettings:
    """Settings of a game that has none."""


class Game:
    """Base class for a game the engine can train.

    A subclass sets `name`, `observation` and `actions`, and implements `reset`,
    `observe`, `step` and the four episode properties. Everything else has a
    default that suits a single-agent game scored on `score`.

    `Settings` is a frozen dataclass of everything that changes the game's
    rules. The engine records it with each run and passes it back when a run is
    resumed or its champion replayed, so two games built from equal settings
    must play identically.
    """

    name: ClassVar[str] = ""
    Settings: ClassVar[type] = NoSettings
    direction: ClassVar[str] = "higher"
    # What `workload` counts, as a profile labels it.
    workload_label: ClassVar[str] = "workload"

    observation: Observation
    actions: Actions
    agents: tuple[str, ...] = ("agent",)

    def __init__(self, settings=None):
        self.settings = self.Settings() if settings is None else settings

    # -------------------------------------------------------------- episode

    def reset(self, seed: int, *, record: bool = False) -> None:
        """Start a new episode. `record` asks the game to keep what `replay` needs."""
        raise NotImplementedError

    def observe(self, agent: str):
        """What `agent` sees now, as `observation` declares it."""
        raise NotImplementedError

    def step(self, actions: dict[str, Any]) -> None:
        """Advance one step with an action for every agent in `agents`."""
        raise NotImplementedError

    @property
    def over(self) -> bool:
        raise NotImplementedError

    @property
    def end(self) -> str:
        """Why the episode ended, as a short name such as "timeout"."""
        raise NotImplementedError

    @property
    def score(self) -> float:
        raise NotImplementedError

    @property
    def steps(self) -> int:
        raise NotImplementedError

    # -------------------------------------------------------------- results

    def goal(self) -> tuple[float, dict[str, float]]:
        """What evolution maximises for the finished episode, and its parts.

        The default is the score itself, negated when lower is better.
        """
        value = float(self.score) if self.direction == "higher" else -float(self.score)
        return value, {"score": value}

    def signals(self) -> dict[str, float]:
        """Named measurements of the finished episode, beyond its score."""
        return {}

    def replay(self, max_steps: int) -> dict | None:
        """A compact record of the finished episode for the dashboard.

        Called only after an episode reset with `record=True`. `max_steps` of 0
        means the whole episode.
        """
        return None

    def scene(self) -> dict | None:
        """What the dashboard draws for this game: its kind and its size."""
        return None

    def milestones(self, summaries: list[dict]) -> list[Milestone]:
        """Achievements read from a run's generation summaries."""
        return []

    # -------------------------------------------------------------- profiling

    def workload(self) -> int:
        """A number that grows with the cost of a step, used to band timings."""
        return 0

    def encoder(self):
        """The function that builds observations, so a profile can time it alone."""
        return type(self).observe

    # -------------------------------------------------------------- model file

    def model_inputs(self) -> dict:
        """Fields the model file records beside the input names."""
        return {"encoding": self.name, "settings": {}}

    def model_rules(self) -> dict:
        """The rules the model file records under `game`, beside its name."""
        return settings_dict(self.settings)

    @classmethod
    def settings_from_model(cls, document: dict):
        """The settings a model file was trained under."""
        rules = dict(document.get("game") or {})
        rules.pop("name", None)
        return load_settings(cls.Settings, rules, strict=False)


# ----------------------------------------------------------------- settings


def settings_dict(settings) -> dict:
    return dataclasses.asdict(settings) if dataclasses.is_dataclass(settings) else {}


def load_settings(cls: type, data: Mapping | None, *, strict: bool = True):
    """Rebuild a settings dataclass from plain data, as a manifest stores it.

    Nested dataclasses are rebuilt from nested mappings and lists become tuples
    where the default is one. With `strict` off, names the class no longer has
    are dropped, so settings recorded by an older version still load.
    """
    if not data:
        return cls()
    defaults = cls()
    values = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        current = getattr(defaults, field.name)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            value = load_settings(type(current), value, strict=strict)
        elif isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        values[field.name] = value
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if strict and unknown:
        raise ValueError(f"{cls.__name__} has no setting {sorted(unknown)[0]!r}")
    return cls(**values)


def override(settings, path: str, value):
    """A copy of `settings` with the setting at dotted `path` replaced."""
    head, _, rest = path.partition(".")
    current = getattr(settings, head)
    if rest:
        value = override(current, rest, value)
    return dataclasses.replace(settings, **{head: value})
