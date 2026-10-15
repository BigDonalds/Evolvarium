# Backlog

Open work only. Settled changes and their measurements go in
[HISTORY.md](HISTORY.md).

Items still open at the baseline are in Snake-AI's
[TODO.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/TODO.md).

## Games

- **The command line plays Snake only.** `train` sets Snake's settings from its
  own flags, `--board`, `--spec`, `--avoidable-penalty` and `--death-penalty`,
  and `cli.py` imports Snake's list of encodings for `--spec`. It needs `--game`
  and one way to set any game's settings by name, which should replace those
  flags.
- **Unknown setting names.** `contract.override` raises AttributeError for a
  name the settings do not have, without saying which names exist. It should
  suggest the closest one. A setting that has no effect under the chosen options
  should be refused rather than ignored.
- **Model files hold a single choice.** The file and `player.py` pick the
  highest output, so a game with a range or with several action parts trains
  but cannot be exported. The file needs each part's kind and its slice of the
  outputs, and the player has to decode them the way `contract.Actions` does.
- **`scored%`** counts genomes whose best episode scored above zero, which says
  nothing about a game that wants its score low.
- **The dashboard draws Snake.** A game's `scene` gives its kind and size, and
  the page only knows the grid.
