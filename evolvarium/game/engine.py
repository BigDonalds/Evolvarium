"""Headless episode runner.

An episode is a pure function of (policy, seed): same genome and same seed give
the same game. That determinism is what makes evaluation comparable between
individuals and makes a recorded replay reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from ..config import BoardConfig, EpisodeConfig
from .board import Board, Cell, Direction, Turn
from .food import place_food
from .snake import Snake


class DeathCause(str, Enum):
    WALL = "wall"
    SELF = "self"
    STARVATION = "starvation"
    STEP_LIMIT = "step_limit"
    BOARD_FILLED = "board_filled"


@dataclass
class Frame:
    """One recorded tick, kept deliberately small: replays store many of these."""

    head: Cell
    food: Cell
    length: int
    score: int


@dataclass
class EpisodeResult:
    score: int = 0
    steps: int = 0
    death: DeathCause = DeathCause.WALL
    death_cell: Cell = (0, 0)
    final_length: int = 0
    # Distinct cells entered, used by the fitness function's coverage term.
    cells_visited: int = 0
    # Summed reduction in Manhattan distance to food across the episode. A
    # positive value means the agent moved toward food more than away from it.
    approach: float = 0.0
    steps_since_food: int = 0
    # Set when the closing stretch of the path revisits a very small cell set.
    looped: bool = False
    # Set when the fatal move had a legal alternative. Distinguishes a mistake
    # from being boxed in, which is the consequence of much earlier choices.
    avoidable_death: bool = False
    frames: list[Frame] = field(default_factory=list)


class Episode:
    """A single game. `record` controls whether frames are retained."""

    def __init__(
        self,
        board_config: BoardConfig,
        episode_config: EpisodeConfig,
        seed: int,
        *,
        record: bool = False,
    ):
        self.board = Board(board_config.width, board_config.height)
        self.rules = episode_config
        self.rng = random.Random(seed)
        self.record = record

        head = (self.board.width // 2, self.board.height // 2)
        self.snake = Snake(head, Direction.UP, board_config.initial_length)
        self.food: Cell | None = place_food(self.board, self.snake, self.rng)

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.finished = False
        self.result = EpisodeResult()

        self._visited: set[Cell] = {head}
        self._approach = 0.0
        self._recent: list[Cell] = []

    @property
    def starvation_limit(self) -> int:
        return self.rules.starvation_limit + self.rules.starvation_per_segment * len(self.snake)

    def _food_distance(self, cell: Cell) -> int:
        if self.food is None:
            return 0
        return abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1])

    def step(self, turn: Turn) -> bool:
        """Apply one turn. Returns True while the episode is still running."""
        if self.finished:
            return False

        heading = turn.apply(self.snake.heading)
        target = self.board.step(self.snake.head, heading)
        before = self._food_distance(self.snake.head)

        if self.snake.would_collide(target, self.board):
            cause = DeathCause.WALL if not self.board.contains(target) else DeathCause.SELF
            # Whether either other turn would have survived this tick.
            escapes = any(
                not self.snake.would_collide(
                    self.board.step(self.snake.head, other.apply(self.snake.heading)), self.board
                )
                for other in Turn
                if other is not turn
            )
            return self._finish(cause, target, avoidable=escapes)

        eating = target == self.food
        self.snake.advance(heading, self.board, grow=eating)
        self.steps += 1
        self._visited.add(target)
        self._approach += before - self._food_distance(target)

        if eating:
            self.score += 1
            self.steps_since_food = 0
            self.food = place_food(self.board, self.snake, self.rng)
            if self.food is None:
                return self._finish(DeathCause.BOARD_FILLED, target)
        else:
            self.steps_since_food += 1

        if self.record:
            self.result.frames.append(
                Frame(
                    head=target, food=self.food or target,
                    length=len(self.snake), score=self.score,
                )
            )

        self._recent.append(target)
        if len(self._recent) > 60:
            self._recent.pop(0)

        if self.steps_since_food >= self.starvation_limit:
            return self._finish(DeathCause.STARVATION, target)
        if self.steps >= self.rules.max_steps:
            return self._finish(DeathCause.STEP_LIMIT, target)
        return True

    def looping(self) -> bool:
        """True when the recent path revisits a very small set of cells."""
        if len(self._recent) < 40:
            return False
        return len(set(self._recent)) < len(self._recent) * 0.35

    def _finish(self, cause: DeathCause, cell: Cell, *, avoidable: bool = False) -> bool:
        self.finished = True
        self.result.avoidable_death = avoidable
        self.result.score = self.score
        self.result.steps = self.steps
        self.result.death = cause
        self.result.death_cell = cell
        self.result.final_length = len(self.snake)
        self.result.cells_visited = len(self._visited)
        self.result.approach = self._approach
        self.result.steps_since_food = self.steps_since_food
        self.result.looped = self.looping()
        return False
