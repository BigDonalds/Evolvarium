# History

What changed after the baseline, why, and what it measured. Newest first.

The baseline is Snake-AI v0.2.0. Everything before it, including the held-out
figures the code refers to, is in Snake-AI's
[HISTORY.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/HISTORY.md).
Figures here use the same held-out seeds, 900000–900059, so the two compare
directly.

---

## Where a move's time goes - a profiler, and what it means for batching

One of the larger pieces of planned work is an evaluator that steps a whole
population's games together and runs the network once a step for all of them.
How much that can gain depends on how much of a move the network takes, so
before any of it is built, `python -m evolvarium profile` measures it on a
checkpoint. It plays held-out games with the clock read around the three calls
in a move, splits the input encoder's time with cProfile, times the network on
its own one state at a time and batched, and scores a population on the worker
pool while accounting for every worker's time.

The timed games are each played twice, once untouched and once with the clock
reads, and the two have to end the same way or the profile stops. The rate
comes from the untouched games and the split from the timed ones. The figures
below are the 81.5 model, member 24 of its run's last checkpoint, on all 60
held-out games: 194,094 moves, which is the record's 194,035 plus the 59 moves
that ended a game in a collision and are not counted as steps.

The machine is a Snapdragon X with 8 cores, running Windows 11 on ARM64. The
Python used up to now is the x86-64 build, which Windows runs under emulation,
so the profile was run on it and on the native ARM64 build of the same version,
3.14.2, with NumPy 2.4.2 on both. The profile's system line reports which build
it ran on. Everything below was measured plugged in. The first runs were on
battery, and what that did is under the machine.

**A move.** 5,590 moves a second on x86-64, 178.9 µs a move, and 7,313 on
ARM64, 136.7 µs. The clock reads add 1.6% and 0.2%.

| | x86-64, µs a move | share | ARM64, µs a move | share |
| --- | --- | --- | --- | --- |
| inputs | 158.1 | 87.0% | 124.9 | 91.2% |
| network | 15.6 | 8.6% | 7.9 | 5.8% |
| game | 7.0 | 3.8% | 3.6 | 2.6% |
| loop and clock reads | 1.1 | 0.6% | 0.6 | 0.4% |

The inputs get cheaper as the snake grows, on x86-64 from 191 µs a move below
length 25 to 105 µs above 100, because there are fewer free cells left to
search. The network and the game cost the same at every length.

**Inside the inputs.** `measure_over_time` takes 89% of the encoder's time on
x86-64 and 90% on ARM64. It is the search that counts how much of the board
each of the three moves leads to while the tail clears, and it comes to 77% and
82% of a whole move. It searches breadth first in Python, separately for each
candidate move, up to 32 moves out. Nothing else in the encoder takes more than
5%.

**The network.** One state at a time, the way games are played now, costs
11.1 µs on x86-64 and 7.0 µs on ARM64. Stepping a population together needs a
batch with a different genome on every row, and at 150 rows that costs 0.72 and
0.60 µs a state.

| batch | x86-64, one genome | x86-64, own genome per row | ARM64, one genome | ARM64, own genome per row |
| --- | --- | --- | --- | --- |
| 1 | 10.92 | 9.63 | 6.96 | 6.24 |
| 8 | 1.88 | 2.17 | 1.03 | 1.47 |
| 64 | 0.54 | 0.86 | 0.23 | 0.68 |
| 150 | 0.41 | 0.72 | 0.16 | 0.60 |
| 600 | 0.33 | 0.64 | 0.13 | 0.57 |

µs a state. On all 600 rows of the widest batch the logits came out bit for bit
the same as `forward` on each row alone, on both builds, so batching would not
change a move.

**What batching the network buys.** At 150 rows it would take a move from
181.8 µs to about 166.9 on x86-64, 8.2% less, and from 137.0 to 129.7 on ARM64,
5.3% less. Moving to the native build saved three to four times as much. On
battery, two runs of the same profile half an hour apart differed by 20% in
move time. A batched evaluator is worth building only if the space search is
batched with it, run for every game at once as array operations, or made much
cheaper first.

**The pool.** 150 genomes, 4 games each, 7 workers, two rounds. Shares are of
worker time.

| build | round | wall | moves a second | starting | scoring | between tasks | end of round |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x86-64 | 1 | 80 s | 25,015 | 0.5% | 90.5% | 0.0% | 9.0% |
| x86-64 | 2 | 87 s | 24,649 | 0.0% | 92.0% | 0.0% | 7.9% |
| ARM64 | 1 | 54 s | 37,172 | 0.6% | 90.5% | 0.0% | 8.9% |
| ARM64 | 2 | 54 s | 39,548 | 0.0% | 91.9% | 0.0% | 8.1% |

Starting the workers took under a second, handing out one task at a time cost
nothing measurable, and a result is under 6 KiB. The only loss inside the pool
is the end of each round, 8 to 9% of worker time, when workers wait for the
longest games to finish. Starting the tasks expected to run longest first should
recover most of it.

**The machine.** The first seven tasks of the second round, 110,806 moves, were
also played one at a time in the main process before the pool started and after
it closed.

| moves a second | one process before | a worker in the pool | one process after |
| --- | --- | --- | --- |
| x86-64 | 5,185 | 4,140 | 4,254 |
| ARM64 | 6,515 | 6,175 | 6,229 |

Seven workers gave 5.8 and 6.3 times one process after the pool. Next to one
process on a machine that has already been busy, running seven side by side
costs a worker 3% on x86-64 and 1% on ARM64. A single process slows as the
laptop warms up, though: the emulated build by 18% from before the pool to
after it, the native build by 4%. The ARM64 run came
straight after the x86-64 one, so even its first figure is from a warm machine.

On battery the laptop throttles under load. The first two runs, both x86-64 on
battery, took 120 to 231 s a round, 9,248 to 16,698 moves a second. In the
second, one process ran at 4,300 moves a second before the pool and 2,359 after,
and a worker in the pool at 1,347. A single process on a cool machine can run
as fast on battery as plugged in, 177.0 µs a move in that run against 178.9, so
a short benchmark need not show it, but a generation does. Generation times
from a run on battery do not compare with ones from a run plugged in.

**Native Python.** The ARM64 build plays all 60 held-out games the same as the
record, move for move, and passes the suite. On a cool machine it plays a move
1.31 times as fast as the x86-64 build. Under steady load the gap widens, since
the emulated build slows more. Replaying the 60 games three times on each,
alternating between the two, took 28.1 to 28.3 s on ARM64 and 42.5 to 44.8 s on
x86-64, 1.55 times as fast, and the pool ran 1.5 to 1.6 times as fast. Windows
on ARM was added to CI after this, on Python 3.11 and 3.14, the oldest and
newest with an ARM64 build.

## Exact resume - the random state and species go into checkpoints

A checkpoint held the genomes and the run's counters but not the random
generator. Resuming built a new generator from the seed given on the command
line, or a random one when none was given, so the resumed run drew different
tournaments, mutations and game seeds from the run it was meant to continue.
Two runs resumed from the same checkpoint did not agree with each other either.
Any comparison that went through a resume mixed the effect being measured with
a reshuffle at the resume point.

Checkpoints are now format 3. They add the generator's state, the species with
their representatives, best fitness and stagnation counts, the speciation
threshold, and each member's id and parents. `tests/test_resume.py` plays six
generations straight through, plays them again with a save and a load after the
third, and requires every generation's report and the final genomes to agree
bit for bit. It does this with speciation off and on, and once more through the
trainer, which also writes history and checkpoints on the way. The load is
handed a different seed on purpose and the saved state has to win. Before the
test was trusted, the random state and then the species were stripped from a
checkpoint, and both times the comparison failed. The runs use an 8 by 8 board
and a 40-move starvation limit, which brings the five tests down from two
minutes on the full board to about 14 seconds.

Restored members keep their ids, so parent links in the history carry across a
resume, and new members are numbered above them. Format 2 checkpoints still
load, and the trainer says that the resumed run will not match the original.

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
machine throughout: Windows 11 on ARM64, 8 logical cores, the x86-64 build of
Python 3.14.2 under emulation, NumPy 2.4.2.

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

**Generation time** is generations 140 to 154 of the 81.5 model's run: 150
genomes, 4 games each, 4 worker processes, mean game length between 1,926 and
3,021 moves. It varies a lot even at similar game lengths. Generations 151 and
153 averaged 2,428 and 2,460 moves a game and took 181 s and 76 s.

The fresh clone was the baseline commit, timed from `git clone` through
`pip install -e ".[dev]"` to its 106 tests passing.
