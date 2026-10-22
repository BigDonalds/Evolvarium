"""Typed configuration for the engine.

Every engine tunable lives here. A game's own rules live in its settings class,
held in `Config.settings`; None there means the game's defaults. This module
imports nothing from the package so it can be read from any other module
without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

# Repository root, derived from this file's location (evolvarium/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: tuple[int, ...] = (16, 12)
    hidden_activations: tuple[str, ...] = ("tanh", "tanh")
    weight_init_scale: float = 1.0
    output_init_scale: float = 0.5


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 150
    elite_count: int = 6
    immigrant_count: int = 4

    # Each genome is evaluated over this many episodes and scored on the MEAN.
    # Taking the best of several trials selects for lucky food placement.
    evaluation_episodes: int = 4

    tournament_size: int = 4
    selection: str = "tournament"  # tournament | rank | roulette

    # Crossover blends two unaligned networks and preserves markedly less
    # parent behaviour than mutation does, so it stays a minority operator.
    crossover_rate: float = 0.25
    crossover_method: str = "weighted_average"

    mutation_rate: float = 0.12
    mutation_strength: float = 0.08
    mutation_method: str = "gaussian"  # gaussian | polynomial
    polynomial_eta: float = 20.0

    # Mutation widens by `stagnation_step` for each generation the best score
    # refuses to move, up to `max_mutation_rate`, and falls back on a record.
    max_mutation_rate: float = 0.35
    stagnation_step: float = 0.02

    # An experiment, off by default. Equally fit genomes keep their order in the
    # population, where the elites come first, and only a best game that beats
    # the record outright sets a new one. At a score ceiling that keeps the
    # first genome to reach it as champion for the whole run, with stagnation
    # climbing every generation. On, the younger of two equally fit genomes
    # ranks first and a best game equal to the record counts as a new one, so
    # the champion follows the newest genome at the top.
    younger_wins_ties: bool = False

    # Fraction of moves replaced by a random action during evaluation. Kept small:
    # a genome's score should reflect its policy, not the noise floor.
    exploration_rate: float = 0.0

    # Worker processes used to evaluate a generation. 0 picks every core but
    # one, leaving room for the dashboard and the trainer thread; 1 runs
    # everything in this process. Results are identical either way.
    workers: int = 0

    use_speciation: bool = False
    compatibility_threshold: float = 3.0
    target_species: int = 8


@dataclass(frozen=True)
class HeldOutConfig:
    """The fixed episodes finished champions are compared on.

    One episode per seed, `episodes` consecutive seeds from `first_seed`, with
    exploration off. Training draws its base seeds uniformly below 2**30, so the
    block is not excluded from training by construction, but a 60-generation run
    lands on it with odds of about one in three hundred thousand.

    Every held-out figure in HISTORY.md uses these values. Changing either one
    makes new results incomparable with all of them.
    """

    first_seed: int = 900_000
    episodes: int = 60

    @property
    def seeds(self) -> list[int]:
        return list(range(self.first_seed, self.first_seed + self.episodes))


@dataclass(frozen=True)
class HistoryConfig:
    """Retention for per-generation records."""

    # Generations held in memory for instant scrubbing.
    memory_window: int = 24
    # Compressed replay files kept on disk. Record-setting generations are
    # exempt from eviction so milestones stay reviewable.
    replay_retention: int = 400
    # Steps recorded per agent replay after its first frame; 0 records the whole
    # game. A limit here cuts the end game out of the replay: at 1,200, a strong
    # snake's replay stopped about 40% of the way before its game did.
    replay_steps: int = 0
    # Agents whose replay is archived, ranked by fitness. 0 archives all.
    agents_per_generation: int = 0


@dataclass(frozen=True)
class CheckpointConfig:
    save_every: int = 25
    keep_last: int = 5


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    # A viewer is considered present this many seconds after its last heartbeat.
    # Once it lapses, a held trainer resumes on its own.
    viewer_timeout: float = 12.0
    # Hold training at generation boundaries while a viewer is present.
    hold_when_viewing: bool = True


@dataclass(frozen=True)
class PathConfig:
    runs_dir: Path = field(default=PROJECT_ROOT / "runs")
    exports_dir: Path = field(default=PROJECT_ROOT / "exports")


@dataclass(frozen=True)
class Config:
    # A built-in game's name, or "package.module:Class" for any other.
    game: str = "snake"
    settings: object | None = None
    network: NetworkConfig = field(default_factory=NetworkConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    held_out: HeldOutConfig = field(default_factory=HeldOutConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def with_overrides(self, **sections) -> Config:
        """Return a copy with whole sections replaced."""
        return replace(self, **sections)


# The engine's sections. A game's own settings are named beside them, such as
# "board.width" beside "evolution.population_size", so no game may use these.
SECTIONS = tuple(f.name for f in fields(Config) if f.name not in ("game", "settings"))

CONFIG = Config()
