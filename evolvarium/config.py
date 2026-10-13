"""Typed configuration for the whole project.

Every tunable lives here. This module imports nothing from the package so it
can be read from any other module without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# Repository root, derived from this file's location (evolvarium/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BoardConfig:
    """Playfield geometry, in cells. Pixel sizes belong to the renderer."""

    width: int = 14
    height: int = 14
    initial_length: int = 3

    @property
    def cell_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class EpisodeConfig:
    """Rules that end an episode."""

    # An episode ends if this many moves pass without food. Scaled by snake
    # length so a long snake gets proportionally more room to manoeuvre.
    starvation_limit: int = 200
    starvation_per_segment: int = 12
    # Hard ceiling so a perfectly cycling agent cannot run forever. Starvation
    # already ends a snake circling without food, so this only ever stops snakes
    # that are still eating. At 3,000 it cut short the three best of 60 held-out
    # games of the 73.1 champion, which went on to 101, 89 and 92 without it.
    max_steps: int = 10_000


@dataclass(frozen=True)
class FeatureConfig:
    """Which state encoding the network sees.

    `spec` selects a named feature set. Checkpoints record the name so a model
    is never loaded into a network built for a different encoding.

    egocentric_v3 -- 11 inputs, all relative to the snake's heading, with the
                     space inputs measured over the next `space_horizon` moves
                     rather than against a frozen body. Default.
    egocentric_v5 -- v3 plus seven rays, each giving the wall and the first body
                     cell still in place when the head could arrive. 25 inputs.
    egocentric_v4 -- v3 plus space as a multiple of body length. Finished level
                     with v3 at generation 120; its extra inputs added no
                     information.
    egocentric_v2 -- the same inputs with space measured against a snapshot.
    egocentric_v1 -- v2 with a capped space scan, which fed scan-order noise into
                     three inputs. Both kept for comparison; see TODO.md for the
                     measured differences.
    """

    spec: str = "egocentric_v3"
    # Upper bound on cells visited by the reachable-space flood fill. Keeps the
    # per-step cost predictable on larger boards. Used by egocentric_v1 only.
    flood_fill_limit: int = 96
    # How many moves ahead egocentric_v3 searches when measuring space. Large
    # enough to cross the board; beyond that it only costs time.
    space_horizon: int = 32


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: tuple[int, ...] = (16, 12)
    hidden_activations: tuple[str, ...] = ("tanh", "tanh")
    output_size: int = 3  # turn left, go straight, turn right
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

    # Mutation widens while the best score refuses to move, within these bounds.
    min_mutation_rate: float = 0.06
    max_mutation_rate: float = 0.35
    stagnation_step: float = 0.02

    elite_mutation_rate: float = 0.04
    elite_mutation_strength: float = 0.02

    # Fraction of moves replaced by a random turn during evaluation. Kept small:
    # a genome's score should reflect its policy, not the noise floor.
    exploration_rate: float = 0.0

    # Worker processes used to evaluate a generation. 0 picks every core but
    # one, leaving room for the dashboard and the trainer thread; 1 runs
    # everything in this process. Results are identical either way.
    workers: int = 0

    use_speciation: bool = False
    compatibility_threshold: float = 3.0
    target_species: int = 8

    hall_of_fame_size: int = 40


@dataclass(frozen=True)
class FitnessConfig:
    """Weights for the fitness terms.

    Food dominates by design. Survival and space terms exist to give a gradient
    for agents that have not yet eaten anything, not to compete with eating.
    """

    food_reward: float = 250.0
    food_exponent: float = 1.35
    survival_per_step: float = 1.0
    survival_cap: float = 120.0
    approach_reward: float = 1.5
    # Bounded for the same reason as survival: an unbounded shaping term can be
    # farmed indefinitely and would out-earn actually eating.
    approach_cap: float = 90.0
    starvation_penalty: float = 40.0
    early_death_penalty: float = 60.0
    early_death_steps: int = 12
    loop_penalty: float = 30.0
    efficiency_reward: float = 60.0

    # Both of the following are fractions of the food already earned, so they
    # scale with what the episode had at stake. Dying at 46 should not cost the
    # same as dying at 5. Both default to 0, which is the behaviour every run so
    # far was measured under.
    #
    # Charged when the fatal move had a legal alternative. Targets the specific
    # failure measured on the v3 champion: 65% of its self-collisions had a safe
    # move available.
    avoidable_death_fraction: float = 0.0
    # Charged on any death by wall, self or starvation. Running out the clock or
    # filling the board is exempt, so this rewards staying alive rather than
    # merely scoring, and a genome that never dies pays nothing.
    death_fraction: float = 0.0


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
    # Steps recorded per agent replay; 0 records the whole game. Every step is
    # stored so the body can be reconstructed exactly. A limit here cuts the end
    # game out of the replay: at 1,200, a strong snake's replay stopped about 40%
    # of the way before its game did and read as a death that never happened.
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
    board: BoardConfig = field(default_factory=BoardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    held_out: HeldOutConfig = field(default_factory=HeldOutConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def with_overrides(self, **sections) -> Config:
        """Return a copy with whole sections replaced. Used by the CLI."""
        return replace(self, **sections)


CONFIG = Config()
