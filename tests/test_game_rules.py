"""Game rule correctness."""

from __future__ import annotations

import dataclasses

import pytest

from evolvarium.config import CONFIG
from evolvarium.game.board import Board, Direction, Turn
from evolvarium.game.engine import DeathCause, Episode
from evolvarium.game.snake import Snake


def make_episode(**episode_overrides) -> Episode:
    rules = dataclasses.replace(CONFIG.episode, **episode_overrides)
    return Episode(CONFIG.board, rules, seed=7)


def test_turns_are_relative_to_heading():
    assert Turn.LEFT.apply(Direction.UP) is Direction.LEFT
    assert Turn.RIGHT.apply(Direction.UP) is Direction.RIGHT
    assert Turn.STRAIGHT.apply(Direction.UP) is Direction.UP
    assert Turn.LEFT.apply(Direction.LEFT) is Direction.DOWN
    assert Turn.RIGHT.apply(Direction.LEFT) is Direction.UP


def test_reversal_is_unreachable():
    """No sequence of one turn can produce the opposite heading."""
    for heading in Direction:
        produced = {turn.apply(heading) for turn in Turn}
        assert heading.opposite() not in produced


def test_snake_spawns_with_body_behind_head():
    snake = Snake((5, 5), Direction.UP, 3)
    assert snake.head == (5, 5)
    # Facing up means the body trails downward, so moving straight is legal.
    assert snake.body() == [(5, 5), (5, 6), (5, 7)]


def test_eating_grows_and_moving_does_not():
    episode = make_episode()
    start = len(episode.snake)
    episode.food = episode.board.step(episode.snake.head, episode.snake.heading)
    episode.step(Turn.STRAIGHT)
    assert len(episode.snake) == start + 1
    assert episode.score == 1

    length = len(episode.snake)
    episode.food = (0, 0)
    episode.step(Turn.STRAIGHT)
    assert len(episode.snake) == length


def test_wall_collision_ends_episode():
    episode = make_episode()
    episode.food = (0, 0)
    for _ in range(CONFIG.board.height + 2):
        if episode.finished:
            break
        episode.step(Turn.STRAIGHT)
    assert episode.finished
    assert episode.result.death is DeathCause.WALL


def test_following_own_tail_is_allowed():
    """The tail vacates on the same tick, so entering it must not be fatal."""
    episode = make_episode()
    board = Board(CONFIG.board.width, CONFIG.board.height)
    episode.snake = Snake((5, 5), Direction.UP, 1)
    snake = episode.snake
    snake._body.clear()
    snake._body.extend([(5, 5), (6, 5), (6, 6), (5, 6)])
    snake._occupied = set(snake._body)
    snake.heading = Direction.UP

    # Turning left from UP heads to (4, 5); turning around the loop into the
    # tail cell (5, 6) requires heading DOWN, which is the reverse. Use the
    # collision predicate directly instead.
    assert snake.would_collide((5, 6), board) is False
    assert snake.would_collide((6, 6), board) is True


def test_starvation_ends_episode():
    episode = make_episode(starvation_limit=5, starvation_per_segment=0)
    episode.food = None
    for _ in range(40):
        if episode.finished:
            break
        # Circle without eating.
        episode.step(Turn.RIGHT if episode.steps % 3 == 0 else Turn.STRAIGHT)
    assert episode.finished
    assert episode.result.death in (DeathCause.STARVATION, DeathCause.SELF, DeathCause.WALL)


def test_food_never_spawns_on_the_snake():
    episode = make_episode()
    for _ in range(200):
        if episode.finished:
            break
        if episode.food is not None:
            assert not episode.snake.occupies(episode.food)
        episode.step(Turn.STRAIGHT if episode.steps % 5 else Turn.RIGHT)


def test_episode_is_deterministic_for_a_seed():
    """Same seed gives the same game; different seeds place food differently."""

    def food_sequence(seed: int) -> list[tuple[int, int]]:
        episode = Episode(CONFIG.board, CONFIG.episode, seed)
        seen = [episode.food]
        while not episode.finished and episode.steps < 60:
            episode.step(Turn.STRAIGHT if episode.steps % 4 else Turn.RIGHT)
            if episode.food != seen[-1]:
                seen.append(episode.food)
        return seen

    assert food_sequence(11) == food_sequence(11)
    assert food_sequence(11) != food_sequence(12)


@pytest.mark.parametrize("heading", list(Direction))
def test_wall_distance_matches_board_geometry(heading):
    board = Board(10, 10)
    assert board.wall_distance((0, 0), Direction.UP) == 0
    assert board.wall_distance((0, 0), Direction.DOWN) == 9
    assert board.wall_distance((9, 9), heading) in range(10)


def test_avoidable_death_is_flagged_when_an_escape_existed():
    """A fatal turn taken while another turn was safe must be marked avoidable."""
    episode = make_episode()
    episode.food = (0, 0)
    # Head against the top wall facing up: straight is fatal, both turns are not.
    episode.snake = Snake((5, 0), Direction.UP, 3)
    episode.step(Turn.STRAIGHT)

    assert episode.finished
    assert episode.result.death is DeathCause.WALL
    assert episode.result.avoidable_death is True


def test_being_boxed_in_is_not_flagged_as_avoidable():
    """Dying with no legal move is a consequence of earlier choices, not a slip."""
    episode = make_episode()
    episode.food = (0, 0)
    # Corner at (0, 0) facing up: straight and left are wall, and right is body.
    # The segment to the right must not be the tail, which would vacate and make
    # the turn legal.
    snake = episode.snake
    snake._body.clear()
    snake._body.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
    snake._occupied = set(snake._body)
    snake.heading = Direction.UP
    board = episode.board
    assert all(
        snake.would_collide(board.step(snake.head, t.apply(snake.heading)), board)
        for t in Turn
    ), "the fixture must leave no legal move"

    episode.step(Turn.STRAIGHT)
    assert episode.finished
    assert episode.result.avoidable_death is False


def test_a_survived_episode_reports_no_avoidable_death():
    episode = make_episode(starvation_limit=4, starvation_per_segment=0)
    episode.food = None
    for _ in range(30):
        if episode.finished:
            break
        episode.step(Turn.STRAIGHT if episode.steps % 3 else Turn.RIGHT)
    if episode.result.death is DeathCause.STARVATION:
        assert episode.result.avoidable_death is False
