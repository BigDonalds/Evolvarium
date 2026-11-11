# Backlog

Open work only. Settled changes and their measurements go in
[HISTORY.md](HISTORY.md).

Items still open at the baseline are in Snake-AI's
[TODO.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/TODO.md).

## Games

- **`scored%`** counts genomes whose best episode scored above zero, which says
  nothing about a game that wants its score low.
- **No cap on an episode's steps with the probe off.** A game with no rule ending
  its episodes is stopped by `limits.episode_seconds`, which ends the run with a
  hint to add one or turn the probe on.

## Probe

- **Inputs that only move once a player is good.** The probe plays with players
  that have learned nothing, so an input it calls dead may only be quiet in weak
  play, and one it leaves alone may grow far out of range in strong play. A run
  could watch the inputs it dropped or left alone and say so when one moves.
- **A small count looks like a code.** Lives from 0 to 3 read the same as a kind
  of tile, and get one input per value. That still trains, but a value past the
  ones weak play showed reads as the largest of them.
- **A code with more than 8 values is read as a number.**
- **No way to keep an input the probe would drop.** Turning the probe off is the
  only way to overrule it for now.
- **Episodes that differ between processes.** The determinism check compares
  episodes within one process, so a game whose episodes depend on the order of a
  set of strings looks fine here and differs between workers.
- **A low `probe.ceiling` on a game with no end.** Below the default the held
  players stop looking once three of them reach it, and on an endless arena that
  has called `hurt` dead. `probe.search_steps=0` does the same on the clean arena.

## Grid encoder

- **The player cannot build grid inputs.** A model trained through the grid
  encoder names the encoder's inputs, and a game outside Evolvarium has to work
  them out itself. A standard-library copy of the encoder beside `player.py`
  would do it.
- **One action, a choice of moves.** The encoder plays a game whose one action is
  where to move. A game that also picks something else each step, such as
  whether to strike, cannot use it.
- **The nearest of each kind only.** The way to the second-nearest food, or to a
  second rival, does not show until the nearest is gone.
- **Four neighbours.** Diagonal moves and other shapes of grid are not covered.
- **Turns in place of the arena's directions.** The arena trains on turns from
  the way each agent last moved when played through the grid. Whether that does
  better or worse than its own four directions is not measured.
- **The arena is about five times slower a generation on the grid.** 2.6 to 3.5 s
  against 0.56 to 0.58 s at the default size, nearly all of it in the searches,
  which cover the open floor for each of four agents every step.
- **Snake's own inputs search cell by cell.** The grid's searches on bit sets give
  the same space inputs, and all 18 of the grid's inputs cost 62 microseconds a
  step against 107 for Snake's 11.
- **Games that are not on a grid.** The encoder needs a board of cells, and the
  cyber estate is a network, where next to means connected to. A network encoder
  that reads which hosts connect and what is happening on each would do for
  networks what the grid does for boards. It would want a comparison against
  cyber's own inputs, with its target set first, as E2 did for Snake.

## Cyber defence

- **Every incident has an intruder.** There are no quiet days, when any response
  is pure cost. A share of seeds without one would be more honest, and would
  make training longer.
- **Three scripted intruders, none of which adapts.** A defence can learn their
  habits. The evolving attacker is planned.
- **Every chance, rate, duration and cost is chosen.** None is measured, and the
  order the hand-written defences land in depends on them: making responses
  dearer pushed every rule toward doing nothing.
- **Records come from one incident.** The engine sets a record, and resets its
  count of generations without one, on a genome's best single incident. The
  tuned rule's cheapest incident costs about 10 against a mean near 80, so an
  early lucky one can hold mutation near its cap for much of a run, as in the
  arena.

## Sizing

- **Sized once, at the start.** A run that turns out slower than its sample
  suggested reaches fewer generations rather than shrinking. Changing the size
  part way belongs with self-diagnosis, logged and replayed like its other
  actions.
- **One allowance for every game.** Generations cost 1.2 to 8 times what the
  sample suggests depending on the game and the size, against the 3 assumed, so
  the generations a plan names are rough.
- **Breeding, history and dispatch go uncounted.** They cost about the same at any
  size, so they weigh most on the smallest populations, which measured the most
  growth on cart-pole even though their episodes grew the least.
- **Measured at a few minutes only.** The sizes were compared at 15 s on
  cart-pole, 1 and 2 minutes on the arena and 3 and 5 on Snake, three seeds each
  and two on Snake at 5.
- **Snake at 5 minutes.** Sized to 100 × 3, it lost to the default size on the
  pick on both clean seeds, by 9 and 6 points. More seeds and longer times before
  trusting it on Snake.

## Engine

- **One worker in the same process.** `train` with `evolution.workers=1` scores
  in the trainer's process, where a step that never returns cannot be ended and
  a game that exits the interpreter takes the run with it. Scoring in a worker
  process even then would cost a second or two at start-up on Windows.
