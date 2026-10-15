"""Snake's settings: the board, the rules that end a game, the inputs and the fitness weights."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class SnakeSettings:
    board: BoardConfig = field(default_factory=BoardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
