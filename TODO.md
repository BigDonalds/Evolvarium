# Backlog

Open work only. Settled changes and their measurements go in
[HISTORY.md](HISTORY.md).

Items still open at the baseline are in Snake-AI's
[TODO.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/TODO.md).

## Games

- **`scored%`** counts genomes whose best episode scored above zero, which says
  nothing about a game that wants its score low.
- **No cap on an episode's steps.** A game with no rule ending its episodes is
  stopped by `limits.episode_seconds`, which ends the run with a hint to add
  one. The engine could cap the steps itself instead, but what the cap should
  be depends on the game, which is for the probe to work out from the game's
  own episodes.

## Engine

- **One worker in the same process.** `train` with `evolution.workers=1` scores
  in the trainer's process, where a step that never returns cannot be ended and
  a game that exits the interpreter takes the run with it. Scoring in a worker
  process even then would cost a second or two at start-up on Windows.
