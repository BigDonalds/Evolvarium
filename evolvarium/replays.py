"""Recorded episodes, as the frames a game's `draw` returned, packed small.

A replay is kept for every member of every generation, so its size decides how
much history a run can keep. Frames are therefore stored as changes. Each frame
gives every layer and label of the scene, in the order the scene declares them,
one operation:

    same    nothing changed
    shift   drop d items from the front of the list and append k new ones
    delta   the same number of items, each stored as its change in every field
    full    the whole list again

A body that moves one cell is a shift that drops its tail and appends a head,
whatever its length, and something that moves a little each step is a delta of
small numbers. Positions are whole multiples of the scene's unit and labels are
kept to their digits, so every stored number is an integer.

Operations go in one byte stream and numbers in another, as variable-length
integers. Most frames repeat the operations of the frame before them, so the
operation stream holds a run length and one set of operations for each run.
Both streams are base64 text, which keeps a replay plain JSON data.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping

from .contract import Scene

SAME, SHIFT, DELTA, FULL = range(4)
# An operation byte holds its kind in the low two bits. A shift keeps d in the
# next three and k in the top three, a full keeps its count in the top six. A k
# or a count too large for its bits is stored among the numbers instead.
_SHIFT_ESCAPE = 7
_COUNT_ESCAPE = 63
# How many items a shift may drop from the front of a list: enough for a body
# or a trail moving a cell or two each step, without trying every split. It
# has to fit in d's three bits.
_REACH = 2


def _unsigned(out: bytearray, value: int) -> None:
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _signed(out: bytearray, value: int) -> None:
    # Zigzag, so small changes either way take one byte.
    _unsigned(out, value * 2 if value >= 0 else -value * 2 - 1)


class Recorder:
    """Collects the frames of one episode and packs them.

    `limit` is how many steps are kept after the first frame; 0 keeps them all.
    Once the recorder is `full` the engine stops drawing, and marks the replay
    `truncated` if the episode carries on.
    """

    def __init__(self, scene: Scene, limit: int = 0):
        self.limit = limit
        self.frames = 0
        self.truncated = False
        # (name, fields, scale, is_label) for each channel, in scene order.
        self._channels = [
            (layer.name, layer.fields, 1.0 / scene.unit, False) for layer in scene.layers
        ] + [(label.name, 1, 10.0 ** label.digits, True) for label in scene.labels]
        self._names = {channel[0] for channel in self._channels}
        self._raw: list[list] = [[] for _ in self._channels]
        self._kept: list[list[tuple[int, ...]]] = [[] for _ in self._channels]
        self._ops = bytearray()
        self._data = bytearray()
        self._run_ops = b""
        self._run = 0

    @property
    def full(self) -> bool:
        return self.limit > 0 and self.frames > self.limit

    def add(self, frame: Mapping) -> None:
        if not self.frames:
            extra = sorted(set(frame) - self._names)
            if extra:
                raise ValueError(f"draw() gave {extra[0]!r}, which the scene does not declare")
        ops = bytearray()
        data = self._data
        raw, kept = self._raw, self._kept
        for index, (name, fields, scale, is_label) in enumerate(self._channels):
            try:
                value = frame[name]
            except KeyError:
                raise ValueError(f"draw() gave no {name!r}, which the scene declares") from None
            items = [(value,)] if is_label else list(value)
            previous = raw[index]
            if items == previous:
                ops.append(SAME)
                continue
            raw[index] = items
            n, m = len(previous), len(items)
            stored = kept[index]

            for drop in range(min(n - 1, _REACH) + 1):
                keep = n - drop
                # The first items are compared alone before the whole lists.
                if (keep <= m and previous[drop] == items[0]
                        and previous[drop:] == items[:keep]):
                    new = _quantize(name, items[keep:], fields, scale)
                    count = m - keep
                    ops.append(SHIFT | drop << 2 | min(count, _SHIFT_ESCAPE) << 5)
                    if count >= _SHIFT_ESCAPE:
                        _unsigned(data, count)
                    _chain(data, stored[-1], new)
                    del stored[:drop]
                    stored += new
                    break
            else:
                new = _quantize(name, items, fields, scale)
                if m == n:
                    ops.append(DELTA)
                    for now, before in zip(new, stored, strict=True):
                        _chain(data, before, (now,))
                else:
                    ops.append(FULL | min(m, _COUNT_ESCAPE) << 2)
                    if m >= _COUNT_ESCAPE:
                        _unsigned(data, m)
                    _chain(data, (0,) * fields, new)
                kept[index] = new

        ops = bytes(ops)
        if ops == self._run_ops:
            self._run += 1
        else:
            self._end_run()
            self._run_ops, self._run = ops, 1
        self.frames += 1

    def _end_run(self) -> None:
        if self._run:
            _unsigned(self._ops, self._run)
            self._ops += self._run_ops

    def pack(self) -> dict:
        self._end_run()
        self._run = 0
        return {
            "frames": self.frames,
            "truncated": self.truncated,
            "ops": base64.b64encode(bytes(self._ops)).decode("ascii"),
            "data": base64.b64encode(bytes(self._data)).decode("ascii"),
        }


def _quantize(name: str, items: list, fields: int, scale: float) -> list[tuple[int, ...]]:
    try:
        if scale == 1.0:
            out = [tuple(map(round, item)) for item in items]
        else:
            out = [tuple(round(v * scale) for v in item) for item in items]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"draw() gave {name!r} a value that cannot be recorded: {error}") from None
    for item in out:
        if len(item) != fields:
            raise ValueError(
                f"draw() gave {name!r} an item of {len(item)} numbers; its shape takes {fields}"
            )
    return out


def _chain(data: bytearray, before: tuple[int, ...], items) -> None:
    """Write each item as its change from the item before it."""
    # `_signed` written out, since this runs for every number of every frame and
    # nearly all of them fit in one byte.
    append = data.append
    for item in items:
        for a, b in zip(item, before, strict=True):
            value = a - b
            value = value * 2 if value >= 0 else -value * 2 - 1
            if value < 0x80:
                append(value)
            else:
                _unsigned(data, value)
        before = item


# ------------------------------------------------------------------ reading


class _Reader:
    __slots__ = ("raw", "at")

    def __init__(self, text: str):
        self.raw = base64.b64decode(text)
        self.at = 0

    def unsigned(self) -> int:
        value = shift = 0
        while True:
            byte = self.raw[self.at]
            self.at += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7

    def signed(self) -> int:
        value = self.unsigned()
        return value >> 1 if not value & 1 else -(value >> 1) - 1

    def item(self, before: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(b + self.signed() for b in before)


def unpack(scene: Scene, replay: Mapping) -> list[dict]:
    """Every frame of a packed replay, in the game's units.

    The dashboard reads replays itself. This is the same reading in Python, for
    tests and for tools that look at replays on disk.
    """
    layers = [(layer.name, layer.fields, scene.unit) for layer in scene.layers]
    labels = [(label.name, 1, 10.0 ** -label.digits) for label in scene.labels]
    channels = layers + labels
    ops, data = _Reader(replay["ops"]), _Reader(replay["data"])
    state: list[list[tuple[int, ...]]] = [[] for _ in channels]

    frames = []
    while len(frames) < replay["frames"]:
        run = ops.unsigned()
        codes = ops.raw[ops.at:ops.at + len(channels)]
        ops.at += len(channels)
        for _ in range(run):
            for index, code in enumerate(codes):
                fields = channels[index][1]
                kind = code & 3
                if kind == SHIFT:
                    drop, count = code >> 2 & 7, code >> 5
                    count = data.unsigned() if count == _SHIFT_ESCAPE else count
                    items = state[index][drop:]
                    for _ in range(count):
                        items.append(data.item(items[-1]))
                    state[index] = items
                elif kind == DELTA:
                    state[index] = [data.item(item) for item in state[index]]
                elif kind == FULL:
                    count = code >> 2
                    count = data.unsigned() if count == _COUNT_ESCAPE else count
                    items, before = [], (0,) * fields
                    for _ in range(count):
                        before = data.item(before)
                        items.append(before)
                    state[index] = items
            frame = {}
            for (name, _, scale), items in zip(layers, state, strict=False):
                frame[name] = [tuple(v * scale for v in item) for item in items]
            for (name, _, scale), items in zip(labels, state[len(layers):], strict=True):
                frame[name] = items[0][0] * scale if items else 0.0
            frames.append(frame)
    return frames
