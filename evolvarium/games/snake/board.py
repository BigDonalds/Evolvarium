"""Grid geometry and headings.

The whole simulation works in cell coordinates. Pixel sizes are a rendering
concern and appear only in the dashboard.
"""

from __future__ import annotations

from enum import IntEnum

Cell = tuple[int, int]


class Direction(IntEnum):
    """Headings in clockwise order, which is what makes turning arithmetic."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Cell:
        return _DELTAS[self]

    def turn_left(self) -> Direction:
        return Direction((self - 1) % 4)

    def turn_right(self) -> Direction:
        return Direction((self + 1) % 4)

    def opposite(self) -> Direction:
        return Direction((self + 2) % 4)


_DELTAS: dict[Direction, Cell] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class Turn(IntEnum):
    """Network output space. A reversal is unreachable by construction, so no
    output head is wasted on a move the rules would reject."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

    def apply(self, heading: Direction) -> Direction:
        if self is Turn.LEFT:
            return heading.turn_left()
        if self is Turn.RIGHT:
            return heading.turn_right()
        return heading


class Board:
    """Immutable playfield bounds with cell arithmetic helpers."""

    __slots__ = ("width", "height")

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    def contains(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, cell: Cell, heading: Direction) -> Cell:
        dx, dy = heading.delta
        return (cell[0] + dx, cell[1] + dy)

    def wall_distance(self, cell: Cell, heading: Direction) -> int:
        """Cells between `cell` and the wall it faces, exclusive."""
        x, y = cell
        if heading is Direction.UP:
            return y
        if heading is Direction.DOWN:
            return self.height - 1 - y
        if heading is Direction.LEFT:
            return x
        return self.width - 1 - x
