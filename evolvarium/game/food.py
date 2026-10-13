"""Food placement."""

from __future__ import annotations

import random

from .board import Board, Cell
from .snake import Snake

# Below this occupancy, rejection sampling finds a free cell in a couple of
# tries. Above it, enumerating the free cells is cheaper than retrying.
_REJECTION_SAMPLING_LIMIT = 0.6


def place_food(board: Board, snake: Snake, rng: random.Random) -> Cell | None:
    """Pick a free cell, or None when the snake fills the board."""
    occupancy = len(snake) / board.cell_count
    if occupancy < _REJECTION_SAMPLING_LIMIT:
        for _ in range(32):
            cell = (rng.randrange(board.width), rng.randrange(board.height))
            if not snake.occupies(cell):
                return cell

    free = [
        (x, y)
        for x in range(board.width)
        for y in range(board.height)
        if not snake.occupies((x, y))
    ]
    return rng.choice(free) if free else None
