"""Snake replays, packed small enough to send between processes and keep on disk."""

from __future__ import annotations

import base64

from .engine import EpisodeResult

# Heading index per step, matching Direction: up, right, down, left.
_DELTA_INDEX = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}


def pack_replay(result: EpisodeResult, max_steps: int) -> dict:
    """Compact a recorded episode into a reconstructable path.

    Every step is kept rather than sampled. Sampling head positions makes the
    body impossible to rebuild: consecutive kept frames are no longer adjacent
    cells, so the snake renders as scattered squares instead of a connected body.

    The path is stored as one byte per step holding the direction moved, which
    is both smaller than coordinate pairs and highly compressible, since a snake
    repeats headings. Food positions and growth points are stored as sparse
    change lists because both change only when the snake eats.
    """
    frames = result.frames[:max_steps] if max_steps > 0 else result.frames
    if not frames:
        return {"start": [0, 0], "dirs": "", "food": [], "grow": [],
                "start_length": 0, "steps": 0, "truncated": False}

    start = frames[0].head
    directions = bytearray()
    previous = start
    for frame in frames[1:]:
        delta = (frame.head[0] - previous[0], frame.head[1] - previous[1])
        directions.append(_DELTA_INDEX.get(delta, 1))
        previous = frame.head

    food: list[list[int]] = []
    grow: list[int] = []
    last_food = None
    last_score = frames[0].score
    for index, frame in enumerate(frames):
        if frame.food != last_food:
            food.append([index, frame.food[0], frame.food[1]])
            last_food = frame.food
        if frame.score != last_score:
            grow.append(index)
            last_score = frame.score

    return {
        "start": [start[0], start[1]],
        "dirs": base64.b64encode(bytes(directions)).decode("ascii"),
        "food": food,
        "grow": grow,
        "start_length": frames[0].length,
        "steps": len(frames),
        "truncated": max_steps > 0 and len(result.frames) > max_steps,
    }
