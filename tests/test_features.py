"""Feature encoding.

The invariance test is the important one. An earlier encoding rescaled each
input vector against its own mean and standard deviation, which made an
unchanged danger flag read differently purely because the food had moved. These
tests pin the property that was lost: physically identical danger must produce
identical numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from evolvarium.config import CONFIG
from evolvarium.features import EGOCENTRIC_V1_NAMES, FeatureSpec, available_specs
from evolvarium.game.board import Direction, Turn
from evolvarium.game.engine import Episode
from evolvarium.game.snake import Snake
from evolvarium.space import SpaceScanner


@pytest.fixture()
def spec() -> FeatureSpec:
    return FeatureSpec(CONFIG.features.spec, CONFIG.features)


def make_episode(seed: int = 3) -> Episode:
    return Episode(CONFIG.board, CONFIG.episode, seed)


def test_spec_width_matches_declared_size(spec):
    episode = make_episode()
    assert len(spec(episode)) == spec.size
    assert spec.size == len(EGOCENTRIC_V1_NAMES)
    assert available_specs()[spec.name] == spec.size


def test_danger_is_invariant_to_food_position(spec):
    """Moving the food must not change any danger or space reading."""
    episode = make_episode()
    danger_space = slice(0, 6)

    episode.food = (0, 0)
    baseline = spec(episode)[danger_space].copy()

    for food in [(13, 13), (7, 2), (2, 11), (6, 6)]:
        episode.food = food
        assert np.array_equal(spec(episode)[danger_space], baseline)


def test_danger_flags_identify_blocked_moves(spec):
    episode = make_episode()
    # Put the head against the top wall facing up: straight ahead is fatal.
    episode.snake = Snake((5, 0), Direction.UP, 3)
    values = spec(episode)
    assert values[1] == 1.0  # danger_straight
    assert values[0] == 0.0  # danger_left is open
    assert values[2] == 0.0  # danger_right is open


def test_features_are_bounded(spec):
    episode = make_episode()
    for _ in range(300):
        if episode.finished:
            break
        values = spec(episode)
        assert np.all(np.isfinite(values))
        assert np.all(np.abs(values) <= 1.5), values
        episode.step(Turn.STRAIGHT if episode.steps % 3 else Turn.RIGHT)


def test_encoding_is_rotation_equivariant(spec):
    """The same relative situation reads the same in every world orientation."""
    readings = []
    for heading in Direction:
        episode = make_episode()
        episode.snake = Snake((7, 7), heading, 3)
        # Place the food two cells directly ahead of the head in every case.
        dx, dy = heading.delta
        episode.food = (7 + dx * 2, 7 + dy * 2)
        readings.append(spec(episode))

    for other in readings[1:]:
        assert np.allclose(readings[0], other, atol=1e-6)


def test_reachable_space_detects_a_pocket():
    """A move into an enclosed pocket must measure less space than open board."""
    scanner = SpaceScanner(8, 8)
    # Wall off a 1x2 pocket at the top-left with an L of body cells.
    body = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    pocket, open_board = scanner.measure(body, [(1, 1), (5, 5)], limit=64)
    assert pocket == 4          # (0,0) (1,0) (0,1) (1,1)
    assert open_board > pocket


def test_reachable_space_is_zero_for_blocked_cells():
    scanner = SpaceScanner(6, 6)
    body = [(3, 3)]
    outside, occupied = scanner.measure(body, [(-1, 0), (3, 3)], limit=32)
    assert outside == 0
    assert occupied == 0


def test_reachable_space_shares_one_component():
    """Candidates opening onto the same region report the same size."""
    scanner = SpaceScanner(8, 8)
    left, right = scanner.measure([(0, 0)], [(4, 4), (5, 4)], limit=256)
    assert left == right


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError, match="unknown feature spec"):
        FeatureSpec("does_not_exist", CONFIG.features)


def test_v2_space_is_free_running_not_truncated():
    """Two moves onto one region must report identical space.

    Capping the fill makes this fail: whichever candidate is scanned first
    consumes the budget, and the second gets a fresh truncated count. The
    resulting difference says nothing about the game, and with the default cap
    it shows up on roughly half of all steps.
    """
    from evolvarium.game.board import Direction
    from evolvarium.game.engine import Episode

    spec = FeatureSpec("egocentric_v2", CONFIG.features)
    episode = Episode(CONFIG.board, CONFIG.episode, seed=5)
    episode.snake = Snake((7, 7), Direction.UP, 3)

    space = spec(episode)[3:6]
    # An empty board is one region, so every open move has the same space.
    assert np.allclose(space, space[0]), space
    assert space[0] == pytest.approx(1.0, abs=0.02)


def test_v2_reports_a_smaller_region_as_smaller():
    scanner = SpaceScanner(9, 9)
    # A body wall down column 3 splits the board into 3-wide and 5-wide halves.
    body = [(3, y) for y in range(9)]
    small, large = scanner.measure(body, [(1, 4), (6, 4)], limit=81)
    assert small < large
    assert small == 27 and large == 45


def test_v1_and_v2_agree_on_danger_columns():
    """Only the space columns differ between the two encodings."""
    from evolvarium.game.board import Turn

    v1 = FeatureSpec("egocentric_v1", CONFIG.features)
    v2 = FeatureSpec("egocentric_v2", CONFIG.features)
    episode = make_episode(11)
    for _ in range(60):
        if episode.finished:
            break
        a, b = v1(episode), v2(episode)
        assert np.array_equal(a[0:3], b[0:3])   # danger
        assert np.array_equal(a[6:], b[6:])     # food, length, starvation
        episode.step(Turn.STRAIGHT if episode.steps % 3 else Turn.RIGHT)


def test_time_aware_scan_sees_through_a_vacating_tail():
    """A snake following its own tail is not trapped, and v3 must know it."""
    scanner = SpaceScanner(9, 9)
    # Body coiled so the cell ahead is enclosed by the snake's own segments.
    body = [(4, 4), (4, 3), (4, 2), (3, 2), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5)]
    frozen = scanner.measure(body, [(3, 4)], limit=81)[0]
    timed = scanner.measure_over_time(body, [(3, 4)], limit=81, horizon=32)[0]
    assert frozen <= 2, "a frozen body makes the coil look like a dead end"
    assert timed > 40, "the tail vacates, so the board is still open"


def test_time_aware_scan_still_reports_a_real_trap():
    """A pocket sealed by segments near the head stays sealed.

    What decides a trap is how soon the sealing cells clear, which depends on
    their distance from the tail. Here the wall is the head end of a long snake,
    so nothing on it moves within the horizon.
    """
    scanner = SpaceScanner(9, 9)
    wall = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]     # seals the top-left 2x2
    tail = [(0, y) for y in range(3, 9)] + [(x, 8) for x in range(1, 9)]
    body = wall + tail                                   # head first, 19 segments

    # A snake must move every tick, so it cannot loiter in the pocket waiting for
    # the wall to clear. The count stays at the pocket size however far ahead the
    # search is allowed to look.
    for horizon in (6, 20, 40):
        timed = scanner.measure_over_time(body, [(1, 1)], limit=81, horizon=horizon)[0]
        assert timed == 4, f"horizon {horizon}: expected the 4-cell pocket, got {timed}"


def test_space_horizon_bounds_the_search():
    scanner = SpaceScanner(15, 15)
    body = [(0, 0)]
    near = scanner.measure_over_time(body, [(7, 7)], limit=225, horizon=3)[0]
    far = scanner.measure_over_time(body, [(7, 7)], limit=225, horizon=40)[0]
    assert near < far
    # Within h moves a walker covers at most a diamond of 2h^2+2h+1 cells.
    assert near <= 2 * 3 ** 2 + 2 * 3 + 1


def test_v3_matches_v2_outside_the_space_columns():
    from evolvarium.game.board import Turn

    v2 = FeatureSpec("egocentric_v2", CONFIG.features)
    v3 = FeatureSpec("egocentric_v3", CONFIG.features)
    episode = make_episode(13)
    for _ in range(80):
        if episode.finished:
            break
        a, b = v2(episode), v3(episode)
        assert np.array_equal(a[0:3], b[0:3])
        assert np.array_equal(a[6:], b[6:])
        episode.step(Turn.STRAIGHT if episode.steps % 4 else Turn.LEFT)


def test_v4_appends_to_v3_without_disturbing_it():
    """The first eleven columns must mean the same thing in both specs."""
    from evolvarium.game.board import Turn

    v3 = FeatureSpec("egocentric_v3", CONFIG.features)
    v4 = FeatureSpec("egocentric_v4", CONFIG.features)
    assert v4.size == v3.size + 3

    episode = make_episode(17)
    for _ in range(80):
        if episode.finished:
            break
        assert np.allclose(v4(episode)[:11], v3(episode), atol=1e-6)
        episode.step(Turn.STRAIGHT if episode.steps % 3 else Turn.RIGHT)


def test_v4_fit_scaling_puts_break_even_at_one_body_length(monkeypatch):
    """0.5 must mean reachable space exactly equals the body, whatever the geometry.

    The scanner is stubbed so the scaling is tested on its own; building a board
    whose reachable count lands on an exact multiple of the body is fiddly and
    would test the geometry rather than the contract.
    """
    from evolvarium import features

    episode = Episode(CONFIG.board, CONFIG.episode, seed=5)
    episode.snake = Snake((7, 7), Direction.UP, 6)
    episode.food = (0, 0)
    length = len(episode.snake)

    class Stub:
        def __init__(self, counts):
            self.counts = counts

        def measure_over_time(self, body, cells, limit, horizon):
            return self.counts

    monkeypatch.setattr(
        features, "scanner_for", lambda w, h: Stub((length // 2, length, length * 3))
    )
    fit = FeatureSpec("egocentric_v4", CONFIG.features)(episode)[11:14]

    assert fit[0] == pytest.approx(0.25)   # half a body of room
    assert fit[1] == pytest.approx(0.5)    # exactly break-even
    assert fit[2] == pytest.approx(1.0)    # three bodies reads the same as two


def test_v4_fit_inputs_saturate_when_the_board_is_roomy():
    """Above two body lengths the input holds at its ceiling, by design."""
    from evolvarium.game.board import Direction
    from evolvarium.game.engine import Episode

    spec = FeatureSpec("egocentric_v4", CONFIG.features)
    episode = Episode(CONFIG.board, CONFIG.episode, seed=3)
    episode.snake = Snake((7, 7), Direction.UP, 3)
    episode.food = (0, 0)

    fit = spec(episode)[11:14]
    # A short snake on an open board has far more than twice its length of room,
    # so the input saturates at its ceiling.
    assert np.allclose(fit[fit > 0], 1.0), fit


def test_v4_fit_inputs_fall_as_the_snake_fills_its_region():
    """The survival margin must drop as the body grows into the space it has."""
    from evolvarium.game.board import Direction
    from evolvarium.game.engine import Episode

    spec = FeatureSpec("egocentric_v4", CONFIG.features)
    readings = []
    for length in (4, 20, 60):
        episode = Episode(CONFIG.board, CONFIG.episode, seed=3)
        # A straight body laid along the rows, so the free region stays connected.
        cells = []
        x = y = 0
        for _ in range(length):
            cells.append((x, y))
            x += 1
            if x >= CONFIG.board.width:
                x = 0
                y += 1
        episode.snake = Snake(cells[0], Direction.DOWN, 1)
        episode.snake._body.clear()
        episode.snake._body.extend(cells)
        episode.snake._occupied = set(cells)
        episode.snake.heading = Direction.DOWN
        episode.food = (CONFIG.board.width - 1, CONFIG.board.height - 1)
        readings.append(float(spec(episode)[11:14].max()))

    assert readings[0] >= readings[1] >= readings[2], readings
    assert readings[2] < readings[0]


# ------------------------------------------------------------------ egocentric_v5

V5 = FeatureSpec("egocentric_v5", CONFIG.features)
V5_RAY_NAMES = (
    "forward", "forward_left", "left", "back_left", "back_right", "right", "forward_right",
)

# A body that folds back across the head's path. Heading up from (7, 7), the cell
# (7, 5) is two cells ahead at body position 8, and (5, 5) is two steps up the
# forward-left diagonal at body position 6.
COIL = [(7, 7), (7, 8), (6, 8), (5, 8), (5, 7), (5, 6), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]


def place_snake(episode, cells, heading):
    """Lay an exact body on the board, head first."""
    episode.snake = Snake(cells[0], heading, 1)
    episode.snake._body.clear()
    episode.snake._body.extend(cells)
    episode.snake._occupied = set(cells)
    episode.snake.heading = heading


def v5_rays(episode):
    values = V5(episode)[11:]
    return {
        name: (float(values[2 * i]), float(values[2 * i + 1]))
        for i, name in enumerate(V5_RAY_NAMES)
    }


def test_v5_appends_to_v3_without_disturbing_it():
    """The first eleven columns must be v3's, bit for bit."""
    v3 = FeatureSpec("egocentric_v3", CONFIG.features)
    assert V5.size == v3.size + 14
    assert V5.names[:11] == v3.names

    episode = make_episode(23)
    for _ in range(120):
        if episode.finished:
            break
        assert np.array_equal(V5(episode)[:11], v3(episode))
        episode.step(Turn.STRAIGHT if episode.steps % 4 else Turn.LEFT)


@pytest.mark.parametrize("heading, cells", [
    (Direction.UP, (11, 4, 4, 4, 4, 11, 11)),
    (Direction.RIGHT, (11, 11, 11, 4, 4, 4, 4)),
])
def test_v5_wall_closeness_matches_the_board(heading, cells):
    """Each ray's distance to the edge, worked out by hand for a head at (3, 10)."""
    episode = Episode(CONFIG.board, CONFIG.episode, seed=1)
    back = heading.opposite().delta
    place_snake(episode, [(3 + back[0] * i, 10 + back[1] * i) for i in range(3)], heading)
    episode.food = (13, 0)

    rays = v5_rays(episode)
    for name, distance in zip(V5_RAY_NAMES, cells, strict=True):
        wall, blocked = rays[name]
        assert wall == pytest.approx(1.0 / distance), name
        # Nothing but the edge lies along any ray from here.
        assert blocked == pytest.approx(wall), name


@pytest.mark.parametrize("length, still_there", [
    (9, False),    # both cells clear a move or more before the head could arrive
    (10, False),   # both clear on exactly the move it could arrive, like a tail
    (11, True),    # both are still occupied when it arrives
])
def test_v5_rays_look_past_segments_that_will_have_moved(length, still_there):
    episode = Episode(CONFIG.board, CONFIG.episode, seed=1)
    place_snake(episode, COIL[:length], Direction.UP)
    episode.food = (0, 13)

    rays = v5_rays(episode)
    # (7, 5) is two moves away and clears after length - 8 moves.
    # (5, 5) is four moves away along the diagonal and clears after length - 6.
    for name in ("forward", "forward_left"):
        wall, blocked = rays[name]
        assert wall == pytest.approx(1.0 / 8), name
        assert blocked == pytest.approx(0.5 if still_there else wall), name


def test_v5_rays_are_the_same_in_every_orientation():
    """Turning the whole board a quarter turn must not change what the snake sees."""
    n = CONFIG.board.width
    assert CONFIG.board.height == n

    def turn(cell):
        x, y = cell
        return (n - 1 - y, x)

    original = Episode(CONFIG.board, CONFIG.episode, seed=1)
    place_snake(original, COIL, Direction.UP)
    original.food = (0, 13)

    rotated = Episode(CONFIG.board, CONFIG.episode, seed=1)
    place_snake(rotated, [turn(c) for c in COIL], Direction.RIGHT)
    rotated.food = turn((0, 13))

    assert np.allclose(V5(original)[11:], V5(rotated)[11:])


def test_v5_blocked_is_never_beyond_the_wall():
    episode = make_episode(29)
    for _ in range(300):
        if episode.finished:
            break
        values = V5(episode)[11:]
        assert np.all(values[1::2] >= values[0::2] - 1e-6)
        assert np.all((values > 0.0) & (values <= 1.0))
        episode.step(Turn.STRAIGHT if episode.steps % 5 else Turn.RIGHT)
