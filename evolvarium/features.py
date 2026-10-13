"""State encoding.

A feature spec turns a live episode into the vector the network reads. Specs are
named and versioned because the input width is baked into every genome: loading
a checkpoint under a different spec would reshape the first layer and silently
destroy the model, so `Checkpoint` records the spec name and refuses a mismatch.

Every spec expresses every value relative to the snake's heading. One learned rule
("turn away from a blocked side") then applies in all four world orientations,
instead of the network needing four rotated copies of the same policy keyed off
a heading flag.

v1 and v2 differ only in how the three reachable-space inputs are measured.

egocentric_v1 caps the flood fill at `flood_fill_limit` cells and divides by that
cap. The cap is the problem: when two candidate moves open onto the same free
region, whichever is scanned first consumes the budget, so the second starts a
fresh truncated scan and lands on a different number. The two inputs then differ
even though the space available is identical. Measured against a strong policy,
the board is genuinely divided into different-sized regions on 3.1% of steps
while v1 reports a difference on 53.7% of them, so most of what it reports is an
artifact of scan order rather than anything about the game.

egocentric_v2 runs the fill to the full free area and divides by the free cells
actually available. 1.0 means "this direction reaches the whole free region" and
0.25 means "this direction leads to a pocket holding a quarter of it". It reports
a difference less often than v1, which is the point: the differences it does
report correspond to real ones.
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig
from .game.engine import Episode
from .space import scanner_for


def egocentric_v1(episode: Episode, config: FeatureConfig) -> np.ndarray:
    """11 heading-relative inputs, each already in a bounded range."""
    board = episode.board
    snake = episode.snake
    head = snake.head
    heading = snake.heading

    headings = (heading.turn_left(), heading, heading.turn_right())
    cells = tuple(board.step(head, h) for h in headings)

    # 0-2: immediate lethality of the three legal moves.
    danger = [1.0 if snake.would_collide(c, board) else 0.0 for c in cells]

    # 3-5: reachable space behind each move, normalised by the cap. Separates a
    # move into open board from a move into a pocket that closes later.
    limit = config.flood_fill_limit
    scanner = scanner_for(board.width, board.height)
    space = [count / limit for count in scanner.measure(snake.cells(), cells, limit)]

    food = episode.food or head
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    # Rotate the food offset into the snake's frame: +forward is ahead of the
    # head, +lateral is to its right, whichever way it happens to be facing.
    fx, fy = heading.delta
    forward = dx * fx + dy * fy
    lateral = dx * (-fy) + dy * fx

    span = max(board.width, board.height)
    # 6-7: where the food is, in body coordinates.
    # 8: how far away it is.
    # 9: how much of the board the snake occupies.
    # 10: progress toward starving, which is the cue to stop stalling.
    return np.array(
        [
            danger[0],
            danger[1],
            danger[2],
            space[0],
            space[1],
            space[2],
            forward / span,
            lateral / span,
            (abs(dx) + abs(dy)) / (board.width + board.height),
            len(snake) / board.cell_count,
            min(episode.steps_since_food / max(episode.starvation_limit, 1), 1.0),
        ],
        dtype=np.float32,
    )


def egocentric_v2(episode: Episode, config: FeatureConfig) -> np.ndarray:
    """11 heading-relative inputs, with space scaled by available free cells."""
    board = episode.board
    snake = episode.snake
    head = snake.head
    heading = snake.heading

    headings = (heading.turn_left(), heading, heading.turn_right())
    cells = tuple(board.step(head, h) for h in headings)

    # 0-2: immediate lethality of the three legal moves.
    danger = [1.0 if snake.would_collide(c, board) else 0.0 for c in cells]

    # 3-5: share of the remaining free board reachable through each move. The
    # scan runs to the whole free area rather than a fixed budget, so two moves
    # opening onto the same region always report the same number.
    free = max(board.cell_count - len(snake), 1)
    scanner = scanner_for(board.width, board.height)
    space = [min(count / free, 1.0) for count in scanner.measure(snake.cells(), cells, free)]

    food = episode.food or head
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    fx, fy = heading.delta
    forward = dx * fx + dy * fy
    lateral = dx * (-fy) + dy * fx

    span = max(board.width, board.height)
    return np.array(
        [
            danger[0],
            danger[1],
            danger[2],
            space[0],
            space[1],
            space[2],
            forward / span,
            lateral / span,
            (abs(dx) + abs(dy)) / (board.width + board.height),
            len(snake) / board.cell_count,
            min(episode.steps_since_food / max(episode.starvation_limit, 1), 1.0),
        ],
        dtype=np.float32,
    )


def egocentric_v3(episode: Episode, config: FeatureConfig) -> np.ndarray:
    """v2 with the space inputs measured over time rather than against a snapshot.

    v2 asks "how much board can I reach if my body never moves". That is the
    wrong question for a long snake, because the body does move: the cells
    directly ahead of a snake following its own tail are vacated exactly as it
    arrives. Judged against a frozen body a safe corridor reads as a dead end,
    and a pocket that will genuinely close reads the same as one that will open.

    v3 asks instead "how much board can I still be standing in after
    `space_horizon` moves", searching outward in move order and letting each body
    cell clear on the move its segment leaves. That distinguishes a trap from a
    tail chase, which the one-step view cannot.
    """
    board = episode.board
    snake = episode.snake
    head = snake.head
    heading = snake.heading

    headings = (heading.turn_left(), heading, heading.turn_right())
    cells = tuple(board.step(head, h) for h in headings)

    danger = [1.0 if snake.would_collide(c, board) else 0.0 for c in cells]

    free = max(board.cell_count - len(snake), 1)
    counts = scanner_for(board.width, board.height).measure_over_time(
        snake.body(), cells, free, config.space_horizon
    )
    space = [min(c / free, 1.0) for c in counts]

    food = episode.food or head
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    fx, fy = heading.delta
    forward = dx * fx + dy * fy
    lateral = dx * (-fy) + dy * fx

    span = max(board.width, board.height)
    return np.array(
        [
            danger[0],
            danger[1],
            danger[2],
            space[0],
            space[1],
            space[2],
            forward / span,
            lateral / span,
            (abs(dx) + abs(dy)) / (board.width + board.height),
            len(snake) / board.cell_count,
            min(episode.steps_since_food / max(episode.starvation_limit, 1), 1.0),
        ],
        dtype=np.float32,
    )


def egocentric_v4(episode: Episode, config: FeatureConfig) -> np.ndarray:
    """v3 plus three inputs for space measured against the snake's own length.

    v3's space inputs answer "what share of the free board lies this way", which
    is the wrong denominator for staying alive. A snake dies when the region it
    is in can no longer hold it, and that threshold moves as it grows: 61
    reachable cells is roomy at length 20 and fatal at length 55. Divided by the
    free board those two situations read almost the same.

    Inputs 11-13 divide by body length instead, so the number carries the
    survival question directly. Scaling is chosen so that 0.5 is the break-even
    point where reachable space exactly equals the body: putting the boundary
    mid-range makes it a threshold the network can sit on, rather than one
    pressed against the end of the input's range.

    Measured on a v3 champion, reachable space falls from 3.25 body lengths forty
    moves before a self-collision to 1.25 one move before. That squeeze is what
    these inputs expose.
    """
    board = episode.board
    snake = episode.snake
    head = snake.head
    heading = snake.heading

    headings = (heading.turn_left(), heading, heading.turn_right())
    cells = tuple(board.step(head, h) for h in headings)

    danger = [1.0 if snake.would_collide(c, board) else 0.0 for c in cells]

    length = max(len(snake), 1)
    free = max(board.cell_count - length, 1)
    counts = scanner_for(board.width, board.height).measure_over_time(
        snake.body(), cells, free, config.space_horizon
    )
    space = [min(c / free, 1.0) for c in counts]
    # 1.0 would mean "twice my length of room"; 0.5 is exactly break-even.
    fit = [min(c / length, 2.0) / 2.0 for c in counts]

    food = episode.food or head
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    fx, fy = heading.delta
    forward = dx * fx + dy * fy
    lateral = dx * (-fy) + dy * fx

    span = max(board.width, board.height)
    return np.array(
        [
            danger[0],
            danger[1],
            danger[2],
            space[0],
            space[1],
            space[2],
            forward / span,
            lateral / span,
            (abs(dx) + abs(dy)) / (board.width + board.height),
            length / board.cell_count,
            min(episode.steps_since_food / max(episode.starvation_limit, 1), 1.0),
            fit[0],
            fit[1],
            fit[2],
        ],
        dtype=np.float32,
    )


# Rays for egocentric_v5, as (forward, rightward) multiples of the heading.
# Straight back is left out: that cell is the neck, which cannot clear before the
# head could turn round to reach it, so the ray would read the same in practically
# every state.
V5_RAYS = (
    ("forward", 1, 0),
    ("forward_left", 1, -1),
    ("left", 0, -1),
    ("back_left", -1, -1),
    ("back_right", -1, 1),
    ("right", 0, 1),
    ("forward_right", 1, 1),
)


def egocentric_v5(episode: Episode, config: FeatureConfig) -> np.ndarray:
    """v3 plus seven rays that know when the body will move out of the way.

    Every earlier spec summarises the body rather than locating it: danger one move
    out, and how much room lies behind each move. Measured on three champions, the
    eleven v3 inputs span under seven independent dimensions and v4's extra three
    added none, which is why v4 finished level with v3. A snake that dies by sealing
    itself in needs to know where the walls of the trap are, and their volume does
    not say.

    Each ray reports two closenesses, 1 / distance in cells:

    * wall    -- the edge of the board that way, of which v3 carries no trace
    * blocked -- the first cell that will still be occupied when the head could
                 first get there. A segment three cells ahead that leaves in two
                 moves is no obstacle, and the ray looks past it.

    1 / distance keeps apart the cells that decide survival: one, two and three
    cells away read 1.0, 0.5 and 0.33, where dividing by the board size would crowd
    them between 0.07 and 0.21.

    The first eleven columns are v3's, computed by v3 itself, so they mean exactly
    the same thing in both specs.
    """
    base = egocentric_v3(episode, config)
    board = episode.board
    snake = episode.snake
    fx, fy = snake.heading.delta
    rx, ry = snake.heading.turn_right().delta
    directions = [
        (forward * fx + right * rx, forward * fy + right * ry) for _, forward, right in V5_RAYS
    ]
    rays = scanner_for(board.width, board.height).rays_over_time(snake.body(), directions)

    extra = np.empty(2 * len(rays), dtype=np.float32)
    for i, (wall, blocked) in enumerate(rays):
        extra[2 * i] = 1.0 / wall
        extra[2 * i + 1] = 1.0 / blocked
    return np.concatenate([base, extra])


EGOCENTRIC_V1_NAMES = (
    "danger_left",
    "danger_straight",
    "danger_right",
    "space_left",
    "space_straight",
    "space_right",
    "food_forward",
    "food_lateral",
    "food_distance",
    "length_ratio",
    "starvation_progress",
)

# Both specs share the same 11 columns; only the scaling of space_* differs.
EGOCENTRIC_V2_NAMES = EGOCENTRIC_V1_NAMES

# v4 appends to v1's columns rather than altering them, so the first eleven
# inputs mean the same thing in every spec.
EGOCENTRIC_V4_NAMES = EGOCENTRIC_V1_NAMES + ("fit_left", "fit_straight", "fit_right")

# v5 appends two columns per ray to the same eleven: the wall's closeness, then
# the closeness of the first cell still occupied when the head could arrive.
EGOCENTRIC_V5_NAMES = EGOCENTRIC_V1_NAMES + tuple(
    f"{kind}_{ray}" for ray, _, _ in V5_RAYS for kind in ("wall", "blocked")
)

_SPECS = {
    "egocentric_v1": (egocentric_v1, len(EGOCENTRIC_V1_NAMES), EGOCENTRIC_V1_NAMES),
    "egocentric_v2": (egocentric_v2, len(EGOCENTRIC_V2_NAMES), EGOCENTRIC_V2_NAMES),
    "egocentric_v3": (egocentric_v3, len(EGOCENTRIC_V2_NAMES), EGOCENTRIC_V2_NAMES),
    "egocentric_v4": (egocentric_v4, len(EGOCENTRIC_V4_NAMES), EGOCENTRIC_V4_NAMES),
    "egocentric_v5": (egocentric_v5, len(EGOCENTRIC_V5_NAMES), EGOCENTRIC_V5_NAMES),
}


class FeatureSpec:
    """Resolved encoder: the callable, its width and its column names."""

    __slots__ = ("name", "_encode", "size", "names", "_config")

    def __init__(self, name: str, config: FeatureConfig):
        if name not in _SPECS:
            raise ValueError(f"unknown feature spec {name!r}; available: {sorted(_SPECS)}")
        encode, size, names = _SPECS[name]
        self.name = name
        self._encode = encode
        self.size = size
        self.names = names
        self._config = config

    def __call__(self, episode: Episode) -> np.ndarray:
        return self._encode(episode, self._config)


def available_specs() -> dict[str, int]:
    return {name: size for name, (_, size, _) in _SPECS.items()}
