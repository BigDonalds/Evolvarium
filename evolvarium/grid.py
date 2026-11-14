"""Inputs built from what is on a game's grid, so nobody has to design them.

A game on a grid declares a `Grid` and describes each agent's surroundings with a
`View` (see `contract.py`): the cells it cannot enter, the ones that open after a
few moves, where the things it cares about are and which way it faces. The
encoder turns that into inputs the same way for every game, each relative to the
way the agent faces, so a rule learned facing up holds facing any way:

* `blocked_<place>`: each cell within `radius` of the agent, 1 if it will still
  be blocked when the agent could first get there, so a cell that opens on
  exactly that move, as a snake's tail does, reads as open.
* `space_<move>`: for each move, the share of the open grid the agent can still
  reach within `horizon` moves of making it, each clearing cell opening on the
  move it does. It tells a move into open floor from one into a pocket.
* `<thing>_via_<move>`: for each kind of thing and each move, 1 if the move
  starts a shortest way to the nearest one within `horizon` moves, else 0.
  Distances in moves differ by 2 at most from one move to the next, so a scale
  such as 1 / moves left them too close to tell apart once the thing was a few
  cells off. On a small test game, networks given that scale scored about a
  third as much after 25 generations.
* `<thing>_forward` and `<thing>_lateral`: how far the nearest one is ahead and
  to the right, as a share of the grid's longer side, however far away it is.
* `crowding`: the share of the grid blocked now.
* each gauge, as the game gives it.

When a game's moves are turns, the network picks them as they are. When they are
steps in fixed directions, the network picks turns instead (left, ahead, right,
back, and stay if the game has it), and an agent whose view gives no heading
faces the way it last stepped, up to begin with.

`GridGame` plays a game through the encoder under the same contract, as
`FixedGame` plays one with the probe's fixes, and the two stack: the probe looks
at the inputs the encoder builds.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from .config import GridConfig
from .contract import STAY, STEPS, TURNS, Actions, Cell, Choice, Game, Grid, Observation, View

# The most moves a cell can take to open, and the farthest the search looks.
_LONGEST = 254
_HEADINGS = frozenset(STEPS)
_UP = STEPS[0]


def _turned(heading: Cell, turn: str) -> Cell:
    """The step `turn` makes for an agent facing `heading`, with y growing downward."""
    hx, hy = heading
    if turn == "ahead":
        return heading
    if turn == "right":
        return (-hy, hx)
    if turn == "left":
        return (hy, -hx)
    if turn == "back":
        return (-hx, -hy)
    return STAY


def _place(ahead: int, right: int) -> str:
    """Where a cell lies from the agent, as its input is named, such as "ahead_left"."""
    parts = []
    if ahead:
        parts.append(_times(abs(ahead), "ahead" if ahead > 0 else "behind"))
    if right:
        parts.append(_times(abs(right), "right" if right > 0 else "left"))
    return "_".join(parts)


def _times(count: int, word: str) -> str:
    return word if count == 1 else f"{count}{word}"


class Encoder:
    """One grid's inputs, built from an agent's view.

    Sets of cells are held as the bits of a Python integer, a row of `width + 2`
    bits for each row of the grid and a row more above and below, so the grid has
    a border of cells that never open. A search then grows by a whole layer of
    cells in a few operations on integers, rather than a loop over the cells.
    """

    def __init__(self, grid: Grid, radius: int, horizon: int):
        if radius < 1:
            raise ValueError(f"the grid encoder sees at least 1 cell each way, not {radius}")
        if not 1 <= horizon <= _LONGEST:
            raise ValueError(f"the grid encoder looks 1 to {_LONGEST} moves ahead, not {horizon}")
        self.grid = grid
        self.radius = radius
        self.horizon = horizon
        width, height = grid.width, grid.height
        self.stride = stride = width + 2
        row = (1 << width) - 1
        self._inside = sum(row << ((y + 1) * stride + 1) for y in range(height))
        self._cells = width * height
        self._span = max(width, height)
        # The cells last given as blocked for good, kept while the view gives the
        # same collection again, as a game's fixed walls usually are. Only one
        # that cannot change is kept.
        self._walls: object = None
        self._walls_bits = 0

        targets = set(grid.moves.values())
        if grid.turning:
            self.turns = tuple(turn for turn in TURNS if turn in targets)
        else:
            self.turns = tuple(turn for turn in TURNS if turn != "stay" or STAY in targets)
        # Each cell around the agent with the fewest moves that reach it, rows from
        # the farthest ahead and each row from the left.
        self.window = tuple(
            (ahead, right, abs(ahead) + abs(right))
            for ahead in range(radius, -radius - 1, -1)
            for right in range(-radius, radius + 1)
            if ahead or right
        )
        names = [f"blocked_{_place(ahead, right)}" for ahead, right, _ in self.window]
        names += [f"space_{turn}" for turn in self.turns]
        names += [f"{thing}_via_{turn}" for thing in grid.things for turn in self.turns]
        names += [f"{thing}_{side}" for thing in grid.things for side in ("forward", "lateral")]
        names += ["crowding", *grid.gauges]
        if len(set(names)) != len(names):
            seen: set[str] = set()
            twice = next(name for name in names if name in seen or seen.add(name))
            raise ValueError(f"the grid encoder would build two inputs named {twice!r}; "
                             "rename the grid's thing or gauge")
        self.names = tuple(names)

    def _bits(self, cells) -> int:
        """The cells of `cells` that are on the grid, as bits."""
        width, height, stride = self.grid.width, self.grid.height, self.stride
        bits = 0
        for x, y in cells:
            if 0 <= x < width and 0 <= y < height:
                bits |= 1 << ((y + 1) * stride + x + 1)
        return bits

    def _blocked(self, cells) -> int:
        if cells is self._walls:
            return self._walls_bits
        bits = self._bits(cells)
        if isinstance(cells, frozenset | tuple):
            self._walls, self._walls_bits = cells, bits
        return bits

    def encode(self, view: View, heading: Cell) -> np.ndarray:
        """The inputs of an agent that sees `view` and faces `heading`."""
        grid = self.grid
        width, height, stride = grid.width, grid.height, self.stride
        ax, ay = view.at
        if not (0 <= ax < width and 0 <= ay < height):
            raise ValueError(f"the view puts the agent at {view.at}, off the "
                             f"{width} by {height} grid")
        if heading.__class__ is not tuple:
            heading = tuple(heading)
        if heading not in _HEADINGS:
            raise ValueError(f"the view's heading is {heading}; it must be one of {list(STEPS)}")

        blocked = self._blocked(view.blocked)
        # The cells that open on each move, by the move. One blocked for good stays so.
        opens: dict[int, int] = {}
        clearing = 0
        for (x, y), moves in view.clearing.items():
            if 0 <= x < width and 0 <= y < height:
                bit = 1 << ((y + 1) * stride + x + 1)
                moves = int(moves)
                if moves > 0 and not blocked & bit:
                    moves = min(moves, _LONGEST)
                    opens[moves] = opens.get(moves, 0) | bit
                    clearing |= bit
        open_now = self._inside & ~blocked & ~clearing
        open_count = open_now.bit_count()

        hx, hy = heading
        rx, ry = -hy, hx
        things = grid.things
        marks = []
        nearest = []
        for thing in things:
            cells = [(x, y) for x, y in view.things.get(thing, ())
                     if 0 <= x < width and 0 <= y < height]
            marks.append(self._bits(cells))
            best = None
            for x, y in cells:
                dx, dy = x - ax, y - ay
                far = abs(dx) + abs(dy)
                if best is None or far < best[0]:
                    best = (far, dx, dy)
            nearest.append(best)

        values = []
        # What is open by the move on which each cell in the window can first be reached.
        by_move = [open_now]
        for moves in range(1, 2 * self.radius + 1):
            by_move.append(by_move[-1] | opens.get(moves, 0))
        for ahead, right, moves in self.window:
            x = ax + ahead * hx + right * rx
            y = ay + ahead * hy + right * ry
            if 0 <= x < width and 0 <= y < height:
                values.append(0.0 if by_move[moves] >> ((y + 1) * stride + x + 1) & 1 else 1.0)
            else:
                values.append(1.0)

        free = max(open_count, 1)
        starts = []
        for turn in self.turns:
            dx, dy = _turned(heading, turn)
            starts.append(1 << ((ay + dy + 1) * stride + ax + dx + 1))
        counts = reached = None
        if not opens and open_now >> ((ay + 1) * stride + ax + 1) & 1:
            counts, reached = self._together(starts, free, open_now, marks)
        if counts is None:
            searched = [
                self._search(start, turn == "stay", free, open_now, opens,
                             marks if reached is None else [])
                for turn, start in zip(self.turns, starts, strict=True)
            ]
            counts = [count for count, _ in searched]
            if reached is None:
                reached = [depths for _, depths in searched]
        values.extend(min(count / free, 1.0) for count in counts)
        for kind in range(len(things)):
            found = [depths[kind] for depths in reached]
            nearest_way = min((depth for depth in found if depth), default=0)
            values.extend(1.0 if depth and depth == nearest_way else 0.0 for depth in found)

        span = self._span
        for best in nearest:
            if best is None:
                values.extend((0.0, 0.0))
            else:
                _, dx, dy = best
                values.append((dx * hx + dy * hy) / span)
                values.append((dx * rx + dy * ry) / span)
        values.append((self._cells - open_count) / self._cells)
        gauges = view.gauges
        for gauge in grid.gauges:
            if gauge not in gauges:
                raise ValueError(f"the view gives no value for the gauge {gauge!r}")
            values.append(float(gauges[gauge]))
        return np.array(values, dtype=np.float32)

    def _together(self, starts: list[int], free: int, open_now: int,
                  marks: list[int]) -> tuple[list[int] | None, list[list[int]]]:
        """What `_search` finds for every move, in one search, on a grid where
        nothing opens later and the agent's own cell is open.

        Each move keeps a frontier of the cells it is first to reach, sharing a
        cell with any move that reaches it on the same move. On such a grid
        arriving sooner never closes a way, so a move that starts a shortest way
        to a thing keeps it to the end, and the moves that find the nearest thing
        of a kind first are the ones `_search` would mark. Every move but a blocked
        one reaches the whole of the agent's part of the grid, and does so within
        two moves of the last cell reached here, so when that is within the horizon
        the counts come from here too. Otherwise they are None and `_search` has
        to count each move.
        """
        moves = len(starts)
        fronts = [start if open_now & start else 0 for start in starts]
        depths = [[0] * len(marks) for _ in range(moves)]
        frontier = 0
        for front in fronts:
            frontier |= front
        seen = frontier
        count = frontier.bit_count()
        missing = 0
        for kind, bits in enumerate(marks):
            if bits:
                missing |= 1 << kind
        depth = 1
        stride = self.stride
        horizon = self.horizon
        while True:
            if missing:
                for kind, bits in enumerate(marks):
                    if missing >> kind & 1 and bits & frontier:
                        for move, front in enumerate(fronts):
                            if front & bits:
                                depths[move][kind] = depth
                        missing &= ~(1 << kind)
            if depth >= horizon or not (count < free or missing):
                break
            depth += 1
            if missing:
                grown = 0
                for move, front in enumerate(fronts):
                    if front:
                        front = (front << 1 | front >> 1 | front << stride | front >> stride
                                 ) & open_now & ~seen
                        fronts[move] = front
                        grown |= front
            else:
                grown = (frontier << 1 | frontier >> 1 | frontier << stride | frontier >> stride
                         ) & open_now & ~seen
            if not grown:
                # The last layer reached was the one before.
                depth -= 1
                break
            frontier = grown
            seen |= grown
            count += grown.bit_count()
        if depth + 2 > horizon:
            return None, depths
        return [count if open_now & start else 0 for start in starts], depths

    def _search(self, start: int, staying: bool, free: int, open_now: int,
                opens: dict[int, int], marks: list[int]) -> tuple[int, list[int]]:
        """Cells reachable within the horizon after the move into the cell `start`
        holds, and the move on which each kind of thing is first reached, 0 for none.

        The search goes outward one move at a time and enters a cell once it has
        opened, as the agent could, since it cannot wait for one to open. It stops
        early once it has counted every open cell and found every kind of thing.
        """
        depths = [0] * len(marks)
        allowed = open_now | opens.get(1, 0)
        # The agent stands in its own cell, whatever the view says of it.
        if not staying and not allowed & start:
            return 0, depths
        missing = 0
        for kind, bits in enumerate(marks):
            if bits & start:
                depths[kind] = 1
            elif bits:
                missing |= 1 << kind
        stride = self.stride
        horizon = self.horizon
        seen = frontier = start
        count = 1
        depth = 1
        while depth < horizon and (count < free or missing):
            depth += 1
            allowed |= opens.get(depth, 0)
            frontier = ((frontier << 1 | frontier >> 1 | frontier << stride | frontier >> stride)
                        & allowed & ~seen)
            if not frontier:
                break
            seen |= frontier
            count += frontier.bit_count()
            if missing:
                for kind, bits in enumerate(marks):
                    if missing >> kind & 1 and bits & frontier:
                        depths[kind] = depth
                        missing &= ~(1 << kind)
        return count, depths


class GridGame:
    """A game played through the grid encoder. Anything not changed goes to the game."""

    def __init__(self, game: Game, config: GridConfig):
        name = type(game).__name__
        grid = game.grid()
        if grid is None:
            raise ValueError(f"{name} has no grid for the grid encoder to read")
        parts = game.actions.parts
        if len(parts) != 1 or not isinstance(parts[0], Choice):
            raise ValueError(
                f"the grid encoder plays a game whose one action is a choice of moves; "
                f"{name} acts by {[part.name for part in parts]}"
            )
        part = parts[0]
        if set(part.options) != set(grid.moves):
            raise ValueError(
                f"{name}'s grid gives moves for {[str(m) for m in grid.moves]}, but its "
                f"action is one of {list(part.labels)}"
            )
        self.base = game
        self.config = config
        # Not `encoder`, which is the contract's method a profile times.
        self.grid_encoder = Encoder(grid, config.radius, config.horizon)
        self.observation = Observation(self.grid_encoder.names)
        self._encode = self.grid_encoder.encode
        self._turning = grid.turning
        if grid.turning:
            self.actions = game.actions
            self._options = None
        else:
            self.actions = Actions(Choice(part.name, self.grid_encoder.turns))
            self._options = {step: option for option, step in grid.moves.items()}
        # The way each agent faces: the view's, or the last step it took.
        self._facing: dict[str, Cell] = {}

    def __getattr__(self, name: str):
        # Only reached for what this class does not define itself. `base` is
        # looked up here while unpickling, before it has been set.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def reset(self, seed: int) -> None:
        self.base.reset(seed)
        self._facing = {}

    def observe(self, agent: str):
        view = self.base.view(agent)
        heading = view.heading
        if heading is None:
            if self._turning:
                raise ValueError(f"the view of {agent} gives no heading, and the grid's "
                                 "moves turn from it")
            heading = self._facing.get(agent, _UP)
        self._facing[agent] = heading
        return self._encode(view, heading)

    def step(self, actions: dict) -> None:
        options = self._options
        if options is None:
            self.base.step(actions)
            return
        facing = self._facing
        moves = {}
        for agent, turn in actions.items():
            step = _turned(tuple(facing.get(agent, _UP)), turn)
            moves[agent] = options[step]
            if step != STAY:
                facing[agent] = step
        self.base.step(moves)

    def encoder(self):
        return Encoder.encode

    def model_inputs(self) -> dict:
        return {
            "encoding": "grid",
            "settings": {
                "radius": self.config.radius,
                "horizon": self.config.horizon,
                # What the game's own inputs were, so a model file can rebuild it.
                "game": self.base.model_inputs(),
            },
        }


def own_inputs(cls: type[Game]) -> bool:
    """Whether a game declares inputs of its own, rather than only a grid."""
    return cls.observe is not Game.observe


def has_grid(cls: type[Game]) -> bool:
    return cls.grid is not Game.grid


def choose(game: Game | type[Game], config: GridConfig) -> GridConfig | None:
    """The grid settings `game` is played with, or None to play it with its own inputs."""
    cls = game if isinstance(game, type) else type(game)
    if config.enabled or not own_inputs(cls):
        return replace(config, enabled=True)
    return None


def with_grid(game: Game, config: GridConfig | None) -> Game | GridGame:
    """`game` played through the grid encoder, or `game` itself for None."""
    return game if config is None else GridGame(game, config)


def to_dict(config: GridConfig | None) -> dict | None:
    return None if config is None else asdict(config)


def from_dict(data: dict | None) -> GridConfig | None:
    """The grid settings a record holds; None for one played with its own inputs."""
    if not data:
        return None
    known = {name: data[name] for name in ("radius", "horizon") if name in data}
    return GridConfig(enabled=True, **known)


def from_inputs(inputs: dict) -> tuple[GridConfig | None, dict]:
    """The grid settings a model file's inputs record, and the game's own inputs.

    A model trained through the encoder records its settings with the game's own
    inputs inside them. Any other model has no grid settings and its inputs are
    the game's.
    """
    if inputs.get("encoding") != "grid":
        return None, inputs
    settings = dict(inputs.get("settings") or {})
    own = settings.pop("game", None) or {}
    return from_dict(settings) or GridConfig(enabled=True), {**inputs, **own}
