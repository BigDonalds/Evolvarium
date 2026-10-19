# History

What changed after the baseline, why, and what it measured. Newest first.

The baseline is Snake-AI v0.2.0. Everything before it, including the held-out
figures the code refers to, is in Snake-AI's
[HISTORY.md](https://github.com/BigDonalds/Snake-AI/blob/v0.2.0/HISTORY.md).
Figures here use the same held-out seeds, 900000–900059, so the two compare
directly.

---

## Replays - the best episode, not the first

Every agent plays `evolution.evaluation_episodes` episodes a generation, four by
default, and the dashboard puts the best one's score on the agent's tile and in
its sheet. The replay beside that score was the first episode's, as it had been
since the baseline. With one snake the two were easy to take for each other. In
the arena a tile could read 10 over a replay in which nobody scored, because the
catch worth 10 came in another match. In a two-generation arena run, 24 of the
60 replays ended on a score other than their tile's.

Every episode of a recorded generation is now recorded, and the replay kept is
the best episode's, the earliest of any that tie. A game that counts lower as
better keeps its lowest. The step count on the tile was the mean over all four
episodes, and is now the replayed episode's own, kept with each member as
`best_episode_steps`. Runs archived before it show the mean as they did. The
agent sheet still gives the mean steps and the worst episode's end, which are
about all four.

**Speed.** Drawing and packing cost about 7 µs a step in either game, 17% of an
arena step and 4% of a Snake step. Recording all four episodes rather than the
first made scoring 60 random arena genomes 12% slower, and four games of the
81.5 model 5% slower.

**Checking it.** `tests/test_replays.py` plays the 81.5 model on three seeds
where it scores 56, 100 and 100, and checks that the replay kept is the second
game's, frame for frame, and not the third's, with that game's 3,623 steps.
Herd, which counts lower as better, keeps its third and lowest. In a two-generation arena run started from
the page, all 60 replays ended on their tile's score, and the best tile's 13
played as three bites of food and a catch.

---

## Runs on the page - start, stop, compare and choose for any game

The dashboard came with `train` and showed the one run that process trained. It
knew Snake by name: the title read snake-ai, one figure was "Feeding", and a
curve was the share of genomes that scored, which says nothing about a game that
wants its score low. Starting a run, changing its settings or comparing two
meant the command line, and the export panel wrote the stored champion while
`export --pick best-average` could choose the steadiest genome instead.

**Serve.** `python -m evolvarium serve` opens the dashboard on every run in the
runs directory and trains nothing until asked. A list at the top picks the run
shown, the one training here or any saved one. A saved run is read from its
history and checkpoints and answers with the same fields as a live one, so the
page draws both alike, and a run whose game no longer loads is still listed with
the reason. Events carry the id of the run they came from. `train` still opens a
dashboard on its own run, and that one cannot start others.

**Starting and stopping.** New run opens a form built from the game's settings:
the names `train --set` takes, grouped by section, with each value's kind and,
where the set is fixed, its choices. Changed settings are marked and counted.
The server checks the form as `train` checks its command line and returns every
problem at once, and the page marks each row named. The server, paths and
held_out sections are not in the form, since every run started from one
dashboard shares them. A run started without a seed is given one and records
it. One run trains at a time. Stop takes two clicks and ends the run after the
generation in progress, and a run that has ended is read back from disk.

**What any game reports.** The main chart plots the best and mean score with a
dot at each record, and the pointer reads out both for the generation under it.
Three small charts follow mean episode length, diversity and the
mutation rate, which every game has, and the page says whether a higher or a
lower score is better. No file of the page names a game. `app.js` grew from 749
lines to 1,951, `index.html` from 163 to 276 and `style.css` from 488 to 791. The
server answers 19 routes, up from 10.

**Compare.** Up to eight runs are set side by side. Their best-so-far curves
share one chart, each in its own colour, and a table gives each run's settings,
game, seed, generations, best score and the generation it came at, the mean of
its last ten generations, episodes, speed and time. The best of each game is
marked, since scores of different games do not compare. A run still training is
fetched again as it moves.

**Choosing by average.** The export panel picks as `export --pick best-average`
does. The stored champion, every member of the population and every past
champion play 8 games, then the best five and the stored champion play 40 more,
and the best mean over all 48 is exported with a note of how it was chosen. The
games run as a job that reports each one, can be cancelled, and runs one at a
time. On a population of 60 on a 10 by 10 board with 4 workers, 65 candidates
took 43 s for 520 screening games and 240 final games. Population member 41 won
with a mean of 18.10, against 15.35 for the stored champion.

**Past champions.** Each archived generation holds the champion as it stood
then, and the champion changes only on a record. The choice opens the first
generation still on disk and the records, not every archived generation. On a
150-generation run that is 15 files instead of 150 and 0.02 s instead of
0.18 s, for the same 13 distinct champions.

**Another process.** A saved run written to in the last two minutes may be
training in another process, so the page keeps reading it and says when it was
written. A run that ended in this dashboard is known to be done unless something
writes to it afterwards.

**Checking it.** `tests/test_server.py` asks the server what the page asks:
the list of runs, the form and the sections it leaves out, every problem with a
new run at once, a run started from the page training and then read from disk
with its seed and settings, the dashboard of one run starting no others, only
the live run being held and stopped, runs side by side, a choice by average
matching the command line's, cancelling a choice and refusing a second, an
export as of a generation, a saved run answering with a live run's fields, and a
run whose game no longer loads. `tests/test_held_out.py` checks that the records
alone give every past champion and that progress is told of every game and can
end a choice. In the browser every part above was used on Snake runs started
from the page, with no errors in the console.

## Scenes - a game draws its own frames, the dashboard draws any game

At the baseline the dashboard drew Snake and nothing else. A replay was Snake's
own: a start cell, one heading per step, and the steps where the food moved or
the snake grew. The page rebuilt the body from the path and painted the head,
body and food itself, and `scene()` gave only the board's kind and size. Another
game trained, but watching it needed a packer of its own in Python and drawing
code of its own in the page.

**Scenes.** A game now declares a `Scene` once: its size and origin in its own
units, which way y grows, its layers and its labels. A layer is a list of cells,
dots, boxes or lines drawn in one tone, such as agent, goal or hazard. Tones name
roles rather than colours, so the page's theme picks the colours. A label is a
number shown beside the drawing, kept to a set number of decimals. While an
episode is recorded the engine calls `draw` after the reset and after every
step, and the game returns each layer's items and each label's value. Snake's
scene is its board, with layers for the food, the body and the head, and the
score as a label. The body is listed from the tail and fades toward it. Snake no
longer keeps frames or packs its replay, and `reset` no longer takes `record`.
Herd, the test game, draws its two agents as dots on a line where y grows
upward, kept to a hundredth of a unit.

**Replays.** `evolvarium/replays.py` packs the frames in the worker as the
episode plays. Each frame gives every layer and label one operation: nothing
changed, a shift that drops items from the front of the list and appends new
ones, a delta that stores each item's change, or the whole list again. A snake's
move is one shift whatever its length, its tail dropped and the cell the head
left appended. Something that drifts a little each step is a delta of small
numbers. Positions are stored as whole multiples of the scene's unit and labels
to their decimals, so every stored number is an integer. Operations and numbers
go in two streams of variable-length integers, and frames that repeat the
operations of the frame before share one run. A replay is plain JSON data, and
`unpack` reads it back into frames.

**Size.** The format cannot know that a snake moves one cell at a time, so it is
larger than Snake's own. Twelve held-out games of the 81.5 model came to 66.4 KiB
before and 217.8 KiB after, or 16.4 and 22.8 KiB compressed. Generation 154 of a
real run came to 609 KiB before and 2,135 KiB after, or 108 and 151 KiB
compressed. Replays are kept on disk compressed, so a run's history takes about
1.4 times the space it did.

**Sending it.** The server compresses any response of 1 KiB or more for a
browser that accepts gzip. At level 5, generation 154's 2,178 KiB of JSON came to
188 KiB in 24 ms. Level 1 took 14 ms and left 515 KiB, and level 6 took 38 ms
for 163 KiB.

**Speed.** Drawing and packing cost about 10 µs a step, 8 to 9% of a recorded
episode. Only recorded episodes pay it, so it comes to about 2% of a
generation's time, against about 0.5% for Snake's own packing.

**The page.** The dashboard reads the scene sent with each generation and draws
every layer as SVG from it, so nothing that draws a replay knows the game. A
replay is read with a cursor that keeps its state every 64 frames, and moving to
any frame starts from the nearest one kept. Reading a whole 4,007-frame Snake
replay takes 18 ms and a random seek about 20 µs. Moving all 60 agents on the
grid to their last frame takes 7.8 ms the first time and 2.7 ms after. The agent
sheet plays the same frame larger, with the frame number and the labels beside
it. A replay that cannot be read stays blank rather than stopping playback.
Replays archived in the old format are not drawn.

**Checking it.** `tests/test_replays.py` has 24 tests: every kind of operation
and the counts too large for an operation byte, random edits, rounding to the
unit and the decimals, a moving body costing a few bytes a frame whatever its
length, a limited recorder, frames and scenes that do not fit, recording that
leaves the game as it was, Snake's replays against the games they recorded,
truncation, and a game in continuous space. `tests/test_server.py` checks when
the server compresses, that a generation's replays unpack against its scene, and
that the status describes the scene. In the browser, the page's reader was
compared with `unpack` frame by frame on four replays: 700 frames using every
shape, operation and escape, and three Snake games of 2,238 to 4,007 frames.
Every frame matched, and 400 random seeks on each landed on the right frame.
The 60 held-out games replay unchanged.

## Model files for any game's actions - parts, ranges and version 2

The model file held a list of action names and took the highest output, so only
a game whose actions were a single choice could export. Herd, the test game that
moves by a choice and a stride, trained, and exporting it said "the model file
holds actions that are a single choice".

**The file.** Version 2 records the actions as the game describes them: a list
of parts, each a choice with its options or a range with its low and high. The
parts read the network's outputs in order, a choice one per option and a range
one. `player.py` decodes them as the engine does. A choice takes its highest
output, the earliest on a tie, and a range stretches the tanh of its output onto
its bounds. A model with one part gives that part's value and one with several
a dict by part name, which prints as JSON from a shell. A choice gives its
option by name, so a choice of 4 gives "3" where the game was handed 3. The
player reads a version 1 file as a single choice, so the shipped 81.5 model
loads, plays and scores as before.

**The game's name.** The file recorded the game's short name, which finds only a
built-in game. It now records the name the registry finds the game by, such as
`test_contract:Herd` for a game outside the package, so `evaluate` can score an
export of any game. Scoring a model also checks that its actions are the
game's. A model whose parts differ, such as a stride with other bounds, is
refused rather than played with its outputs read as something else.

**Checking it.** The player is compared with the engine's decoding on 300 random
inputs each for a single choice, a single range, a choice and a range, and
three parts mixing both. Choices agree exactly. Over 120,000 range decodes from
random networks the largest gap was 2.5e-7 of the range's width, from float64
against float32. A Herd model exported after one generation plays two games
with the player deciding beside the engine at every step, and scores the same
through `evaluate` as through the engine. Broken parts are refused with the part
named. The 81.5 model still makes the engine's choice on all 194,094 held-out
states.

## Settings by name - checked before a run starts

The command line had flags for some settings and none for the rest: four of
Snake's, `--board`, `--spec` and the two death penalties, and fourteen of the
engine's. Anything else meant editing `config.py`. Nothing checked the values.
Some failed once the run was under way: a starting snake longer than half the
board's height stopped the first generation with an IndexError from inside the
episode. Others were quietly bent into something that could run: a tournament
of 1 drew 2, and a mutation rate above its ceiling ran at the ceiling. A setting
that the others switch off, such as the tournament size under rank selection,
was accepted and ignored, so a sweep over it would compare runs that were all
the same.

**Names.** Every setting now has one name. A game's are named by their path in
its settings, such as `board.width`, and the engine's by section, such as
`evolution.population_size`. `train --set NAME=VALUE` sets any of them and reads
the value as the setting's type, with lists written comma-separated.
`--game` picks the game, built in or `module:Class`. The flags for the settings
a sweep changes most, such as `--population` and `--mutation-rate`, stay as
short forms of the same thing. Snake's own flags are gone, so `cli.py`
imports nothing from Snake, and so is `--no-speciation`, which turned off what
is already off by default. `settings` lists every setting a game trains under,
its value and its choices. It replaces `specs`, which listed Snake's encodings.

**Checks.** `evolvarium/settings.py` checks a configuration as a whole and
reports every problem in one list:

- a name that is not a setting, with the closest one that is. Dotted names are
  also compared by their last part, so `width` finds `board.width`;
- a value that cannot be read, or does not have the type of the default;
- a value outside the setting's choices, with the closest one;
- a value out of range, or at odds with another: elites and immigrants that
  leave no room for offspring, a mutation rate above `max_mutation_rate`,
  activations that do not match the hidden layers one for one, a tournament
  outside 2 to the population;
- a setting given by name that the others leave without effect, with the reason,
  such as "evolution.tournament_size has no effect, since evolution.selection is
  rank". Left at its default it is not an error and is not range-checked, since
  nothing reads it.

A game adds its own through three hooks on the contract. `setting_choices` gives
the values a setting allows, `setting_problems` checks its settings against each
other, and `unused_settings` names the ones the others switch off. Snake checks
its board and that the starting snake fits below the centre cell, and refuses a
negative fitness weight, since the formula gives each weight its sign. The flood
fill limit is unused unless the encoding is `egocentric_v1`, the space horizon
unless it is v3 to v5, and the shape of each fitness term unless its weight is
above 0. Under `--headless` the server settings and `--open` have no effect
either. The trainer runs the same check before it creates the run's directory,
so a configuration that cannot train leaves nothing behind.

**Removed settings.** `elite_mutation_rate`, `elite_mutation_strength` and
`hall_of_fame_size` were read by nothing. `min_mutation_rate` was a floor under
a rate that only ever widens from `mutation_rate`, so it only acted on a
`mutation_rate` set below it, which it raised without saying so. The rate now
starts where it is set. Manifests that record these still load, and `inspect`
and the held-out grouping leave them out.

**Checking it.** `tests/test_settings.py` has 67 tests: reading each type from
text, each kind of problem for the engine and for Snake, each rule for a setting
without effect, the game in `tests/test_contract.py` under its own settings, the
registry refusing a game whose setting shares a name with an engine section, the
trainer stopping before it writes anything, and the command line. The defaults
pass for both games. The 60 held-out games replay unchanged and resumed runs
still match uninterrupted ones.

## Games behind a contract - Snake moves out of the engine

At the baseline the engine was Snake. The population built a Snake episode to
score a genome, the evaluator called Snake's input encoder, fitness was Snake's
formula, checkpoints and the model file recorded the board and the feature spec,
the profiler grouped its timings by snake length, and the export panel's
achievements counted food. Training anything else meant changing all of them.

`evolvarium/contract.py` now says what a game gives the engine. It resets from a
seed, and the same seed gives the same episode. It names the agents that share
an episode, one unless it says otherwise. It declares the inputs each agent sees
by name, as numbers or as categories, and the engine gives a category one column
per value so no value reads as larger than another. It declares its actions in
its own terms: a choice among named options, a number within a range, or several
of these at once, which arrive as a dict by part name. Each step it takes an
action for every agent together. It says when the episode is over and why, gives
the score and whether a higher or a lower score is better, and may report named
measurements beside the score. The engine sizes the network from the inputs and
actions, encodes what the game observes and decodes the outputs, so a game never
handles the network's arrays.

Snake moved to `evolvarium/games/snake`: its board, episode, inputs, fitness,
replay packing and achievements, and its settings. The board, episode, features
and fitness sections of the engine's configuration are now `SnakeSettings`.
Nothing in the engine imports from that package. Games are found by name,
"snake" for the built-in one and "module:Class" for one that lives anywhere
Python can import it, and worker processes find a game the same way. The
configuration has `game` and `settings`, and the run manifest records both.

**Lower is better.** A game can ask for a lower score. Fitness is then the score
negated, unless the game defines its own. Records and a generation's best
episode follow the game's direction, and a genome's worst ending is its highest
score. The best score so far starts as none rather than 0, so the first
generation is a record whichever way the game counts. Roulette selection clipped
negative fitness to 0, which would give every member of such a game the same
weight. When any fitness is negative it now measures every member up from the
lowest. Fitness sharing divided fitness by the size of the species, which lifts
a negative figure toward zero and would favour the largest species, so it does
the same. Neither changes anything for a run where no fitness is negative.

**Checking it.** `tests/test_contract.py` defines a second game in the test
module itself: two agents on a line, who score lower the closer they finish to
the middle, with a category among their inputs and an action made of a choice
and a range. It trains through the population and gets better, round-trips
through a checkpoint, scores the same in worker processes with exploration on,
and is refused by a configuration that plays Snake. The same file tests input
encoding, action decoding, settings and the registry's errors, 30 tests in all.
The 60 held-out games replay unchanged, the player makes the engine's choice on
every one of their moves, and resumed runs still match uninterrupted ones.

**Speed.** The loop now builds a dict of actions every step and goes through the
game's encode and decode. On the ARM64 build, 15 held-out games, 50,056 moves,
took 112.0 µs a move before the change and 111.9 to 113.0 after, so it costs
nothing measurable next to the inputs.

**Checkpoints** are format 4. They record the game, the name of its inputs, the
input names themselves and the settings. Loading refuses a checkpoint whose
game, encoding or input names differ from the configuration's. Older
checkpoints still load and are read as Snake, and `inspect` and `evaluate` take
their settings from the run's manifest.

**Names.** How an episode ended is its end rather than its death: `ends` in the
generation summaries and the reports, and summaries that say `deaths` are still
read. The console's `eat%` column is `scored%`. The held-out report drops the
`50+` column, which only meant something for Snake, and its worst, best and
quartiles follow the game's direction. `inspect` prints the game, its encoding
and each setting changed from the default as a dotted path. The profiler groups
timings by whatever the game says drives the cost of a step, snake length for
Snake, and adds groups as the games reach them instead of sizing them from the
board.

**The model file** already named its game. It picks an action as the highest
output, so it holds games whose actions are a single choice. Other games train,
and exporting one says why it cannot.

## Training events - the loop announces, everything else listens

The trainer used to call everything that cared about a generation itself. From
inside the loop it built the history record, decided when to checkpoint and told
the console what to print. The dashboard was never told anything: it asked for
the status and the generation list every 1.2 seconds and fetched the whole chart
series again for each new generation. Every new view meant another call wired
into the loop.

The trainer now publishes events as the run proceeds: run started, generation
started, generation evaluated, record set, held, resumed, checkpoint saved, a
note, and run finished with the reason it ended. It no longer refers to history,
checkpoints, the console or the dashboard. Each of them subscribes to the events
it needs, in `evolvarium/events.py`.

Subscribers are of two kinds. History and checkpoints keep the run's record.
They hear each event first, so the console and the dashboard only hear about a
generation once it is written down. If one of them fails, the run stops, since a
run that carries on without its record leaves nothing to resume from. The
console and the dashboard are views. A view that raises is dropped, the failure
is announced as a note, and training carries on. Handlers run on the trainer
thread and may not draw from the population's random generator, so a run evolves
the same with or without listeners. The held-out replay and the resume tests
pass unchanged, and the trainer's resume test now resumes from the checkpoint
the run left behind by itself rather than one saved afterwards.

**Stopping.** After a Ctrl+C the command line used to write the last checkpoint
from the main thread. `stop()` waited 10 seconds for the generation in progress
and then closed the workers regardless, so a slow generation could be saved part
way through, after it had already drawn its seeds. A checkpoint taken there does
not continue the run it came from. The workers also received the Ctrl+C
themselves, since the console sends it to every process attached to it, and
could die mid-generation. Workers now ignore it. The first Ctrl+C asks the
trainer to stop after the current generation, and the trainer saves
`latest.npz` on its own thread at that boundary before printing the closing
summary. A second closes the workers under the generation. The run ends as
abandoned, nothing is saved from it, and the summary names the last checkpoint
that was written and its generation.

**Dashboard.** `GET /api/events` streams the events as server-sent events. Each
one is formatted once into a log of the last 1,024, which every open page reads
at its own pace, so the trainer never waits on a browser. Ids carry the server's
start time. A page that reconnects continues after the last id it saw, and when
the log no longer reaches back that far, or the id came from an earlier server,
it is told to reload from the regular endpoints instead. The page extends the
chart from each generation event rather than fetching the series again, shows a
record or a note as it happens, and polls status every 5 seconds for the clocks
and rates. While the stream is down it polls at the old 1.2 seconds. A quiet
stream sends a comment every 15 seconds, so a closed tab shows up as a failed
write rather than a thread left waiting.

**Not announced yet.** Automatic adjustments. The only ones so far are the
mutation-rate schedule, which moves on every stagnant generation, and the drift
of the speciation threshold. Announcing those would fill the feed with routine,
so they get an event once the engine's adjustments are decisions rather than
schedules.

## A model file of its own - JSON export and a player that needs nothing installed

An exported model used to be a `.pt` file, which takes PyTorch to write and to
read, with a JSON file beside it describing the run. The JSON named the inputs
and the actions but held no weights, and the `.pt` held the weights but not the
board, the episode limits or the encoder's settings, so neither file was enough
on its own to play the model again under the rules it learned.

Exporting now writes one JSON file that is the model. It holds every weight and
bias, the names of the inputs in the order the network reads them, the actions,
the encoder and its settings, the board and episode rules, the run and
generation it came from and the training report. `evolvarium/player.py` reads it
and plays it with the standard library alone. The player imports nothing from
the rest of the package, so it can be copied next to a game that never installs
Evolvarium, and its docstring is the format's specification. The file carries a
version number. The player refuses a version it does not know, and for a file it
cannot run it says what is wrong, down to the layer and the row.

Layers store one row of weights per output, the layout `torch.nn.Linear` uses,
so the file maps onto other frameworks without transposing anything. Each weight
is written as the exact decimal value of its float32. That costs about 24 bytes
a weight, but reading the file back gives the same genome bit for bit, which a
test checks with weights spread over 60 orders of magnitude. The 81.5 model's
file is 13,745 bytes with its training report.

**Float64 against float32.** The engine computes in float32 with NumPy and the
player in float64 with plain Python, from the same numbers. They can only choose
differently where the top two outputs come within rounding of each other, so
what matters is how close the model's calls get. Over all 194,094 moves of its
held-out games the closest is 2.7e-7 apart, at move 542 of seed 900030. The
float64 outputs are never more than 7.3e-7 from the float32 ones, and at that
move they are 8.1e-8 away, so the player makes the engine's choice on every
move. The slow suite now also plays the whole block with the player choosing,
asks the engine at every move, and checks that the two never differ and that
every game ends as recorded. The quick suite does the same for its six games.

**Speed.** The player takes 28 µs a move on the ARM64 build of Python, against
5 µs for the engine's network, and 51 µs against 11 µs on the x86-64 build.
That is still small next to the 125 µs the inputs take on ARM64, and a game that
only plays a trained model calls it once a move.

**Torch.** Exporting no longer needs it. With torch installed an export also
writes a `.pt` copy holding the same contents, the genome, and each layer's
weights and biases as tensors. It loads with `torch.load`'s default safe mode,
without `weights_only=False`. The dashboard's export panel says which files will
be written.

**Scoring a file.** `evaluate` takes a model file as well as a checkpoint or a
run, and scores it under the rules the file records. The shipped model,
`models/v3-seed99-gen155-81.json`, is rewritten in the new format with the same
weights, export time, selection details and training report. On its own it
scores 81.45 over the held-out block, as the run's checkpoint does. The `.pt`
beside it stays as it was at the baseline, since the record in `tests/data`
names it, and a test that runs without torch now checks the record's weights and
rules against the JSON file too.

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
