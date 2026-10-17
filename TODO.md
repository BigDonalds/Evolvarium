# Backlog

Open work only. Settled changes and their measurements go in
[HISTORY.md](HISTORY.md).

Items still open at the baseline are in Snake-AI's
[TODO.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/TODO.md).

## Games

- **`scored%`** counts genomes whose best episode scored above zero, which says
  nothing about a game that wants its score low.
- **The dashboard draws Snake.** A game's `scene` gives its kind and size, and
  the page only knows the grid.

## Engine

- **`layer_uniform` crossover is `uniform`.** Its comment says it swaps whole
  rows so each unit keeps its input weights together, but it draws a mask per
  gene exactly as `uniform` does. It needs the network's layer shapes to swap
  rows, and a run that chose it so far has trained under `uniform`.
