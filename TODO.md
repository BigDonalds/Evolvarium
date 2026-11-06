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
- **No way to keep an input the probe would drop.** Turning the probe off is the
  only way to overrule it for now.
- **Episodes that differ between processes.** The determinism check compares
  episodes within one process, so a game whose episodes depend on the order of a
  set of strings looks fine here and differs between workers.
- **A low `probe.ceiling` on a game with no end.** Below the default the held
  players stop looking once three of them reach it, and on an endless arena that
  has called `hurt` dead. `probe.search_steps=0` does the same on the clean arena.

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
