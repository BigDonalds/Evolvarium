# Evolvarium

[![tests](https://github.com/BigDonalds/Evolvarium/actions/workflows/tests.yml/badge.svg)](https://github.com/BigDonalds/Evolvarium/actions/workflows/tests.yml)

A platform that trains game-playing agents and NPC behaviour by neuroevolution,
from a user's own environment, for people with no background in machine learning.

Evolvarium is a student-initiated undergraduate capstone project at the
University of Windsor, carried out over two semesters. The first turns a
single-game engine into one that trains any environment written to a common
contract. The second builds the platform around it. It is in development.

## Baseline

The code starts as an exact copy of
[Snake-AI v0.2.0](https://github.com/BigDonalds/Snake-AI/tree/v0.2.0), where that
project stopped. Only the name differs: the package `snake_ai` is now
`evolvarium`, and the command, dashboard title, default export name and model
format string follow it. The first commit is the baseline the project is
measured against, and everything after it is new work.

At the baseline the engine trains Snake and nothing else. Its best model, in
`models/`, averages 81.5 over 60 held-out games.

## Running

Python 3.10 or newer. On Windows on ARM, use the ARM64 build of Python. The
x86-64 build runs under emulation and trains about a third slower.

```bash
pip install -e ".[dev]"
python -m evolvarium train --open
python -m pytest
```

`python -m pytest -m slow` replays all 60 held-out games of the model in
`models/` and checks each game's score, length and cause of death against the
record in `tests/data`. It takes under a minute. CI runs both on Linux, Windows,
Windows on ARM and macOS.

`python -m evolvarium --help` lists the other commands. The full reference for
the baseline is Snake-AI's
[COMMANDS.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/COMMANDS.md),
reading `snake_ai` there as `evolvarium`.

## Exported models

`python -m evolvarium export`, or the export panel on the dashboard, writes a
trained model as one JSON file: its weights, the names of its inputs in the
order the network reads them, its actions and the game rules it learned under.
`evolvarium/player.py` runs that file with the standard library alone, so it can
be copied next to a game that never installs Evolvarium:

```python
from player import Player

player = Player.load("v3-seed99-gen155-81.json")
move = player.action(inputs)  # "left", "straight" or "right"
```

The player's docstring describes the format. `python -m evolvarium evaluate`
scores a model file on the held-out games under the rules it records. With
torch installed, an export also writes a `.pt` copy.

## Documents

- [HISTORY.md](HISTORY.md): what changed after the baseline, and what it measured
- [TODO.md](TODO.md): open work
