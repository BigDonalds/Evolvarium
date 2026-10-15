"""Snake body state.

The body is a deque for O(1) growth at the head and removal at the tail, and is
shadowed by a set so collision tests are O(1) instead of a list scan. Both are
kept in step by every mutating method.
"""

from __future__ import annotations

from collections import deque

from .board import Board, Cell, Direction


class Snake:
    __slots__ = ("_body", "_occupied", "heading")

    def __init__(self, head: Cell, heading: Direction, length: int):
        # Segments trail directly behind the head, so the first move is legal
        # regardless of which way the snake is facing.
        back = heading.opposite().delta
        self._body: deque[Cell] = deque(
            (head[0] + back[0] * i, head[1] + back[1] * i) for i in range(length)
        )
        self._occupied: set[Cell] = set(self._body)
        self.heading = heading

    @property
    def head(self) -> Cell:
        return self._body[0]

    @property
    def tail(self) -> Cell:
        return self._body[-1]

    def __len__(self) -> int:
        return len(self._body)

    def body(self) -> list[Cell]:
        return list(self._body)

    def cells(self):
        """Iterate the body without copying it. Read-only by convention."""
        return self._body

    def occupies(self, cell: Cell) -> bool:
        return cell in self._occupied

    def would_collide(self, cell: Cell, board: Board) -> bool:
        """Whether moving into `cell` ends the episode.

        The tail is excluded because it vacates on the same tick, unless the
        snake is about to grow into it.
        """
        if not board.contains(cell):
            return True
        if cell not in self._occupied:
            return False
        return cell != self.tail

    def advance(self, heading: Direction, board: Board, *, grow: bool) -> Cell:
        """Move one cell and return the new head position."""
        self.heading = heading
        head = board.step(self.head, heading)
        if not grow:
            # Free the tail before claiming the head so a snake may follow its
            # own tail into the cell it is leaving.
            old = self._body.pop()
            self._occupied.discard(old)
        self._body.appendleft(head)
        self._occupied.add(head)
        return head

    def self_collided(self) -> bool:
        """True when the head shares a cell with any other segment."""
        return len(self._occupied) != len(self._body)
