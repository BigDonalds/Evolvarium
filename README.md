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

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
python -m evolvarium train --open
python -m pytest
```

`python -m pytest -m slow` replays all 60 held-out games of the model in
`models/` and checks each game's score, length and cause of death against the
record in `tests/data`. It takes under a minute. CI runs both on Linux, Windows
and macOS.

`python -m evolvarium --help` lists the other commands. The full reference for
the baseline is Snake-AI's
[COMMANDS.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/COMMANDS.md),
reading `snake_ai` there as `evolvarium`.

## Documents

- [HISTORY.md](HISTORY.md): what changed after the baseline, and what it measured
- [TODO.md](TODO.md): open work
