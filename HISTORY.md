# History

What changed after the baseline, why, and what it measured. Newest first.

The baseline is Snake-AI v0.2.0. Everything before it, including the held-out
figures the code refers to, is in Snake-AI's
[HISTORY.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/HISTORY.md).
Figures here use the same held-out seeds, 900000–900059, so the two compare
directly.

---

## Pinning down the baseline - the 81.5 model as a test, CI on three systems, and starting measurements

Most of the planned work rewrites what the 81.5 model depends on: the game
loop, the inputs, the network and the training loop. Before any of it starts,
the model's held-out games are part of the test suite. A change that alters how
the model plays now fails a test, instead of showing up later as a held-out
figure a few points off that is easy to put down to seed noise.

**The record.** `tests/data/held_out_81.json` holds the model's 435 weights, the
board, episode, input and network settings it was trained under, and all 60 of
its held-out games: score, number of moves and cause of death. Replayed, they
come back the same, game for game, at a mean of 81.45 over 194,035 moves. The
model runs into itself in 57 games, into a wall in 2 and starves in 1. Checking
all three numbers per game is strict. Once one move differs, the snake is
somewhere else for the rest of the game, and these games run from 433 to 5,618
moves with a median near 3,200, so a game that changes path has little chance
of ending on the same score, length and death.

**Quick and slow.** The full block replays in about 35 seconds here, too long
for every run of the suite, so it is marked `slow` and runs with
`pytest -m slow`. Six games run with the ordinary suite and take a few seconds:
the four shortest, the one starvation and the shorter of the two wall deaths,
so every way this model dies is still covered. A third test, skipped when
PyTorch is not installed, checks the weights in the record against the exported
model in `models/`, so the record cannot drift from the file it describes.

**CI.** GitHub Actions runs the suite and the slow replay on Linux, Windows and
macOS, each on Python 3.10 and 3.14, the oldest supported version and the
newest, and runs `ruff check` once. The replay runs on every system on purpose.
The network's matrix products go through different BLAS builds and CPU
instructions on each, and a sum rounded differently in the last place can flip
a close choice between two moves. On the first run all six passed: the 60 games
came back the same on every system and both versions of Python.

**Lint.** At the baseline `ruff check` had one complaint: the rename pushed a
line in `cli.py` from 99 characters to 101. It is wrapped.

### Starting measurements

Taken at the baseline so later work has something to be compared with. One
machine throughout: Windows 11, 8 logical cores, Python 3.14.2, NumPy 2.4.2.

| | |
| --- | --- |
| test suite | 108 tests in 51 s; the slow replay another 35 s |
| line coverage | 75.5% of 3,257 lines |
| fresh clone to passing tests | 99 s: clone 1, virtual environment 11, install 34, tests 53 |
| play speed, one process | 4,800 to 5,600 moves a second, 81.5 model |
| one generation, late in a run | 71 to 208 s, median 181 s |

**Coverage** was measured with a short line tracer built on `sys.monitoring`.
It counts only code that runs in the test process, so the parts of training
that run in worker processes are undercounted, and the dashboard's JavaScript is
not measured at all. The thinnest modules are the dashboard server (16%),
speciation (19%), the console reporter (38%), mutation and crossover (42%), the
training loop (43%), selection (55%) and the command line (56%).

**Generation time** is generations 140 to 157 of the 81.5 model's run: 150
genomes, 4 games each, 8 worker processes, mean game length between 1,926 and
3,519 moves. It varies a lot even at similar game lengths. Generations 155 and
156 averaged 3,354 and 3,417 moves a game and took 115 s and 208 s.

The fresh clone was the baseline commit, timed from `git clone` through
`pip install -e ".[dev]"` to its 106 tests passing.
