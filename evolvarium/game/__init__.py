"""Game rules: board geometry, snake body, food placement, episode loop."""

from .board import Board, Cell, Direction, Turn
from .engine import DeathCause, Episode, EpisodeResult, Frame
from .snake import Snake

__all__ = [
    "Board", "Cell", "Direction", "Turn",
    "DeathCause", "Episode", "EpisodeResult", "Frame",
    "Snake",
]
