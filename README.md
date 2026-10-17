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

At the baseline the engine trained Snake and nothing else. Its best model, in
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
`models/` and checks each game's score, length and how it ended against the
record in `tests/data`. It takes under a minute. CI runs both on Linux, Windows,
Windows on ARM and macOS.

Every setting has a name. A game's are named by their path in its settings,
such as `board.width` for Snake, and the engine's by section, such as
`evolution.population_size`. `python -m evolvarium settings` lists them with
their values and the choices each one allows, and `train` sets any of them:

```bash
python -m evolvarium train --set board.width=12 network.hidden_layers=32,16 network.hidden_activations=relu,tanh
```

A run checks its settings before it starts and reports every problem at once: a
name that does not exist, with the closest one that does, a value of the wrong
type, out of range or at odds with another setting, and a setting given by name
that the others leave without effect.

Ctrl+C stops training after the generation in progress and saves it. A second
Ctrl+C drops that generation and quits. The dashboard follows the run through
`GET /api/events`, a server-sent event stream of each generation, record, hold
and checkpoint, which any other client can read as well:

```bash
curl -N http://127.0.0.1:8765/api/events
```

`python -m evolvarium --help` lists the other commands. The full reference for
the baseline is Snake-AI's
[COMMANDS.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/COMMANDS.md),
reading `snake_ai` there as `evolvarium`. Snake's own flags there, such as
`--board` and `--spec`, are settings now, set with `--set`.

## Games

The engine trains any game written to the contract in `evolvarium/contract.py`.
A game declares the inputs each agent sees, as numbers or categories, and the
actions it can take, as named options or numbers within a range. It resets from
a seed, takes an action for every agent each step, and says how each episode
ended and whether a higher or a lower score is better. The engine sizes the
network from that and turns its outputs back into the game's own actions.

Snake, in `evolvarium/games/snake`, is the one built in. Any other game is named
`module:Class` and imported from wherever Python finds it, so
`train --game package.module:Class` trains it and
`settings --game package.module:Class` lists its settings. A game can declare
the values a setting allows, check its settings against each other, and name
the ones that others leave without effect, and the run uses all three before it
starts. `tests/test_contract.py` trains one with two agents and a score that is
better lower.

## Exported models

`python -m evolvarium export`, or the export panel on the dashboard, writes a
trained model as one JSON file: its weights, the names of its inputs in the
order the network reads them, each part of its actions and the game rules it
learned under.
`evolvarium/player.py` runs that file with the standard library alone, so it can
be copied next to a game that never installs Evolvarium:

```python
from player import Player

player = Player.load("v3-seed99-gen155-81.json")
move = player.action(inputs)  # "left", "straight" or "right"
```

A game whose actions have several parts, such as a direction and a stride, gets
a dict from part name to value, as the game received them in training.

The player's docstring describes the format. `python -m evolvarium evaluate`
scores a model file on the held-out games under the rules it records. With
torch installed, an export also writes a `.pt` copy.

## Documents

- [HISTORY.md](HISTORY.md): what changed after the baseline, and what it measured
- [TODO.md](TODO.md): open work
