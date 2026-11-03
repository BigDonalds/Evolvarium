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
record in `tests/data`, then probes every game, broken or not, on the default
budget. It takes under six minutes. CI runs both on Linux, Windows, Windows on
ARM and macOS.

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

`python -m evolvarium serve --open` opens the dashboard on every run in the runs
directory without training one. Runs are started there from a form of the
game's settings, checked as `train` checks them, and can be stopped, set side by
side and exported, choosing the exported genome by its average as `export` does.
`--game` picks the game the form starts on, and `--set` the values it starts
from:

```bash
python -m evolvarium serve --game package.module:Class --set evolution.population_size=120 --open
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

Three games are built in and go by their short names, as in `train --game arena`.
Snake is in `evolvarium/games/snake`. The arena, in `evolvarium/games/arena`,
puts a team of agents with health on one floor to eat, heal and chase a scripted
rival that turns on them when they are hurt, and every agent plays the same
network. `python -m evolvarium.games.arena.scripted` plays hand-written teams on
the held-out seeds, the figures a trained team is measured against. Cart-pole,
in `evolvarium/games/cartpole`, keeps a pole upright on a cart by pushing the
cart along a track, and the network gives the push as a force rather than
picking a move. Its score is the number of steps the pole stayed up, out of 500
by default, and `python -m evolvarium.games.cartpole.scripted` does the same for
hand-written controllers.

Any other game is named `module:Class` and imported from wherever Python finds
it, so `train --game package.module:Class` trains it and
`settings --game package.module:Class` lists its settings. A game can declare
the values a setting allows, check its settings against each other, and name
the ones that others leave without effect, and the run uses all three before it
starts. `tests/test_contract.py` trains one with two agents and a score that is
better lower.

A game's code plays under two time limits, `limits.step_seconds` for a step and
`limits.episode_seconds` for a whole episode, 10 s and 120 s by default. An
episode that runs over is played once more, since a busy machine can hold up a
step that is fine, and running over again ends the run with the step and seed
it happened on. An exception from the game ends it the same way, naming the
call it came from. With more than one worker, a worker that dies or stops
reporting while it scores is replaced and the genome played again on the new
one. A second worker lost on the same genome ends that run, and the dashboard
and any other runs carry on. `train` with `evolution.workers=1` plays the game
in its own process and can only check the time after a step, so a step that
never returns hangs it there.

`evolvarium/games/broken.py` has each built-in game five more times, each with
one fault of the kind a game written in a hurry has: inputs far from zero in
units of their own, inputs that never change, inputs that copy others, no limit
on an episode's length, and episodes that ignore their seed. None is built in,
so a run names one in full, as in
`train --game evolvarium.games.broken:DeadInputSnake`.

The probe looks for those faults in any game before anything trains on it.
Players that have learned nothing play the game, one at random, one repeating a
single action, one holding each action for a while and untrained networks, on
seeds from 600000 up. It takes about a second on cart-pole, under 10 s on the
arena and under 35 s on Snake, and says what it measured and what it would do
about it:

```bash
python -m evolvarium probe --game evolvarium.games.broken:DeadInputCartPole
```

`--set probe.enabled=true`, on `train` or the dashboard's form, probes the game
before a run and trains on the game as fixed. It is off by default.

- An input reading beyond 8 either way, or barely moving for where it sits, is
  moved and scaled to read about -1 to 1. Inputs in range are left alone.
- An input that never changes is dropped, and so is one that only repeats
  another, even flipped or scaled. The network is sized from what is left.
- A game with an episode still going at `probe.ceiling` steps, 5,000 by default,
  gets a limit on steps with the score standing still, if its score moves often
  enough to tell, and a limit on all the steps of an episode. A game whose
  episodes all end by themselves gets no limit at all.
- A game whose episodes a seed does not decide has its random generators seeded
  at the start of each one, wherever it keeps them. If that does not settle it,
  the report says so and the run goes ahead.
- Rewards in under 2% of episodes, and steps too slow for the time limits, are
  reported and left for the game's author.

A game with nothing wrong is played as it was written and trains the same as with
the probe off. The report is kept in the run's directory as `probe.json`, and the
fixes go with its checkpoints and exported models, so a resumed run or a replay
reads the game as the run did. A game can define `stop(stalled)` to settle an
episode the probe's limits cut short, as the built-in games do. The players being
weak is also the probe's limit: an input that only moves or grows once a player
is good looks dead or well scaled to them.

A game that declares a `scene` can be watched on the dashboard. The scene lists
layers of cells, dots, boxes or lines and labels such as the score, and `draw`
returns each layer's items and each label's value after every step of a recorded
episode. The dashboard draws any game from that without running its code.

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

A model of a game the probe changed still takes every input the game has. The
file says which of them the network reads, with the offset and scale of each, and
the player does the sums itself.

The player's docstring describes the format. `python -m evolvarium evaluate`
scores a model file on the held-out games under the rules it records. With
torch installed, an export also writes a `.pt` copy.

## Documents

- [HISTORY.md](HISTORY.md): what changed after the baseline, and what it measured
- [TODO.md](TODO.md): open work
