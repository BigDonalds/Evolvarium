"""Reachable-space measurement.

Knowing how much room lies behind a move is what separates an agent that avoids
the next wall from one that avoids sealing itself into a pocket. It is also the
single most expensive thing computed per step, so the implementation matters.

Three details carry almost all of the speed:

* The grid is stored as a flat bytearray padded with a blocked border, so a
  neighbour is `index +/- 1` or `index +/- stride` and no bounds check is needed
  in the inner loop.
* Cells are integer indices rather than coordinate tuples, which removes tuple
  allocation and hashing from the hot path entirely.
* The three candidate moves usually open onto the same free region. Labelling
  connected components means that region is traversed once and then looked up,
  instead of being walked three times.

Buffers are allocated once per board size and reused, so a step does no
allocation beyond the small work stack.
"""

from __future__ import annotations

_FREE = 0
_BLOCKED = 1
# Sentinel for a cell that never clears within any horizon we search.
_NEVER = 255


class SpaceScanner:
    """Reusable connected-component scanner for one board size."""

    __slots__ = ("width", "height", "stride", "size", "_template", "_blocked", "_label",
                 "_stack", "_clear_at", "_time_template", "_seen", "_zero")

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.stride = width + 2
        self.size = (width + 2) * (height + 2)

        # Template with the border pre-blocked. Copying it resets the grid in a
        # single C-level slice assignment rather than a Python loop.
        template = bytearray(self.size)
        for x in range(self.stride):
            template[x] = _BLOCKED
            template[self.size - self.stride + x] = _BLOCKED
        for y in range(height + 2):
            template[y * self.stride] = _BLOCKED
            template[y * self.stride + width + 1] = _BLOCKED
        self._template = bytes(template)

        self._blocked = bytearray(self.size)
        self._label = bytearray(self.size)
        self._stack: list[int] = []

        # Timed variant: interior starts at 0 (already clear), border never clears.
        timed = bytearray(self.size)
        for i, blocked in enumerate(self._template):
            if blocked:
                timed[i] = _NEVER
        self._time_template = bytes(timed)
        self._clear_at = bytearray(self.size)
        self._seen = bytearray(self.size)
        self._zero = bytes(self.size)

    def index(self, x: int, y: int) -> int:
        return (y + 1) * self.stride + (x + 1)

    def measure(self, occupied, candidates, limit: int) -> list[int]:
        """Free cells reachable through each candidate, each capped at `limit`.

        `occupied` is an iterable of (x, y) body cells; `candidates` an iterable
        of (x, y) target cells. Candidates outside the board or on the body
        measure zero.
        """
        blocked = self._blocked
        label = self._label
        stride = self.stride

        blocked[:] = self._template
        for x, y in occupied:
            blocked[(y + 1) * stride + (x + 1)] = _BLOCKED

        # Label 0 means unvisited; components are numbered from 1.
        label[:] = bytes(self.size)
        sizes = [0]
        results: list[int] = []

        for x, y in candidates:
            start = (y + 1) * stride + (x + 1)
            if blocked[start]:
                results.append(0)
                continue

            existing = label[start]
            if existing:
                results.append(sizes[existing])
                continue

            component = len(sizes)
            if component > 255:
                # Cannot label further components in a bytearray; the remaining
                # candidates fall back to an uncapped-but-unlabelled count.
                results.append(self._count_only(blocked, start, limit))
                continue

            count = 0
            stack = self._stack
            stack.clear()
            stack.append(start)
            label[start] = component

            while stack and count < limit:
                node = stack.pop()
                count += 1
                for nxt in (node - stride, node + stride, node - 1, node + 1):
                    if blocked[nxt] or label[nxt]:
                        continue
                    label[nxt] = component
                    stack.append(nxt)

            sizes.append(count)
            results.append(count)

        return results

    def measure_over_time(self, body, candidates, limit: int, horizon: int) -> list[int]:
        """Cells reachable within `horizon` moves, accounting for the tail moving.

        The static `measure` treats the whole body as permanent wall, which is
        wrong in the way that matters most at length: a snake following its own
        tail is not trapped, because the cells ahead of it are being vacated as it
        arrives. Judging a corridor against the body's *current* position calls
        that a dead end.

        Here each body cell carries the move on which it clears -- the tail on the
        next move, the segment ahead of it the move after -- and the search may
        enter a cell once it would arrive no earlier than that. Searching outward
        in move order is what makes this a lookahead rather than a snapshot: the
        count answers "how much board can I still be in after `horizon` moves",
        which is the question a trap actually poses.

        The search advances only while it can move outward, which matches the
        rules: a snake cannot stand still and wait for a wall to clear. A pocket
        whose exit opens later than the snake can survive inside it therefore
        counts as the pocket, not as the board beyond it.

        `body` is ordered head first. Returns one count per candidate.
        """
        stride = self.stride
        clear_at = self._clear_at
        seen = self._seen
        zero = self._zero

        clear_at[:] = self._time_template
        length = len(body)
        for offset, (x, y) in enumerate(body):
            # Deque index `offset` vacates after (length - offset) moves.
            clear_at[(y + 1) * stride + (x + 1)] = min(length - offset, _NEVER - 1)

        results: list[int] = []
        for x, y in candidates:
            start = (y + 1) * stride + (x + 1)
            # Arriving on move 1: the cell must be clear by then.
            if clear_at[start] > 1:
                results.append(0)
                continue

            seen[:] = zero
            seen[start] = 1
            frontier = [start]
            count = 1
            depth = 1

            while frontier and depth < horizon and count < limit:
                depth += 1
                nxt: list[int] = []
                for node in frontier:
                    for step in (node - stride, node + stride, node - 1, node + 1):
                        if seen[step] or clear_at[step] > depth:
                            continue
                        seen[step] = 1
                        count += 1
                        nxt.append(step)
                    if count >= limit:
                        break
                frontier = nxt
            results.append(count)

        return results

    def rays_over_time(self, body, directions) -> list[tuple[int, int]]:
        """Distance to the wall, and to the first cell still occupied on arrival, per ray.

        Reachable space says how much room lies each way; it does not say where the
        body is. A ray does, and a ray that knows when each segment leaves can tell
        a coil that will unwind from one that will not.

        Walking a ray out from the head, the cell `k` steps away cannot be reached
        in fewer moves than its Manhattan distance: `k` along a straight ray, `2k`
        along a diagonal. A body cell blocks the ray only if it clears later than
        that. This is the rule `measure_over_time` uses and the one the game applies
        to its own tail, so a segment leaving on exactly the move the head could
        arrive is passable. Like `measure_over_time`, it assumes nothing is eaten on
        the way, which would hold the tail in place for a move.

        `body` is ordered head first. `directions` are `(dx, dy)` steps, straight or
        diagonal, and should not point straight back into the neck. Returns
        `(wall, blocked)` per ray, in cells. Both are at least 1, and `blocked` is
        never beyond `wall` because the border never clears.
        """
        stride = self.stride
        clear_at = self._clear_at
        border = self._template

        clear_at[:] = self._time_template
        length = len(body)
        for offset, (x, y) in enumerate(body):
            clear_at[(y + 1) * stride + (x + 1)] = min(length - offset, _NEVER - 1)

        hx, hy = body[0]
        head = (hy + 1) * stride + (hx + 1)
        results: list[tuple[int, int]] = []
        for dx, dy in directions:
            step = dx + dy * stride
            cost = abs(dx) + abs(dy)
            node = head
            distance = 0
            blocked = 0
            while True:
                node += step
                distance += 1
                if not blocked and clear_at[node] > distance * cost:
                    blocked = distance
                if border[node]:
                    break
            results.append((distance, blocked))
        return results

    def _count_only(self, blocked: bytearray, start: int, limit: int) -> int:
        """Flood without labelling, for the rare case of many components."""
        stride = self.stride
        seen = {start}
        stack = [start]
        count = 0
        while stack and count < limit:
            node = stack.pop()
            count += 1
            for nxt in (node - stride, node + stride, node - 1, node + 1):
                if blocked[nxt] or nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return count


_scanners: dict[tuple[int, int], SpaceScanner] = {}


def scanner_for(width: int, height: int) -> SpaceScanner:
    """Shared scanner per board size."""
    key = (width, height)
    scanner = _scanners.get(key)
    if scanner is None:
        scanner = SpaceScanner(width, height)
        _scanners[key] = scanner
    return scanner
