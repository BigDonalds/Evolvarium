"""The JSON model file and the standalone player that runs it.

The player computes in float64 from the same numbers the engine holds as
float32, so the two could part ways where two outputs come within rounding of
each other. Over the 81.5 model's whole held-out block the closest call is
about 3e-7 apart and float64 lands within 1e-7 of float32 there, so the player
must make the engine's choice on every one of the 194,094 states.
"""

from __future__ import annotations

import ast
import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest
from test_golden import GOLDEN, QUICK_SEEDS, ROOT, golden_settings, recorded

from evolvarium.contract import Actions, Choice, Range
from evolvarium.games.snake import SnakeGame
from evolvarium.nn.activations import ACTIVATIONS as ENGINE_ACTIVATIONS
from evolvarium.nn.network import Architecture, PolicyNetwork, random_genome
from evolvarium.persistence import model_file
from evolvarium.player import ACTIVATIONS, ModelError, Player, check
from evolvarium.player import main as player_main

MODEL = ROOT / "models" / "v3-seed99-gen155-81.json"
PLAYER = ROOT / "evolvarium" / "player.py"
STATES_IN_BLOCK = 194_094
MOVE = [{"name": "move", "kind": "choice", "options": ["a", "b", "c"]}]
MOVE_AND_STRIDE = Actions(Choice("move", ("left", "stay", "right")), Range("stride", 0.5, 1.5))


def play_both(seeds) -> tuple[list[dict], int, list[tuple]]:
    """Play each seed with the player choosing, asking the engine at every state.

    Returns the games, the number of states and every state where they differ.
    """
    player = Player.load(MODEL)
    network = PolicyNetwork(Architecture.from_dict(GOLDEN["architecture"]))
    network.load(np.array(GOLDEN["genome"], dtype=np.float32))
    game = SnakeGame(golden_settings())
    assert game.observation.names == player.input_names
    decide = game.actions.decode

    games, states, differ = [], 0, []
    for seed in seeds:
        game.reset(seed)
        while not game.over:
            inputs = game.observation.encode(game.observe("agent"))
            choice = player.action(inputs.tolist())
            engine = decide(network.forward(inputs))
            if choice != engine:
                differ.append((seed, game.steps, choice, engine))
            states += 1
            game.step({"agent": choice})
        games.append({"seed": seed, "score": game.score, "steps": game.steps,
                      "death": game.end})
    return games, states, differ


def document_for(genome, architecture, actions=MOVE) -> dict:
    return model_file.build(
        genome=genome, architecture=architecture,
        input_names=[f"x{i}" for i in range(architecture.input_size)], encoding="test",
        settings={}, actions=actions, game={"name": "test"}, source={},
    )


def version_1(document):
    """The same model laid out as version 1, which names the outputs of one choice."""
    document["version"] = 1
    document["actions"] = {"names": ["a", "b", "c"], "choice": "argmax"}


def small_model(activations, seed=0, scale=1.0,
                actions=None) -> tuple[dict, np.ndarray, Architecture]:
    """A random model with 5 inputs, acting by `actions` or a choice of a, b and c."""
    architecture = Architecture(
        input_size=5, hidden_layers=(7,) * len(activations),
        hidden_activations=tuple(activations),
        output_size=3 if actions is None else actions.size,
    )
    rng = np.random.default_rng(seed)
    genome = random_genome(architecture, rng, 1.0, 1.0) * np.float32(scale)
    parts = MOVE if actions is None else actions.describe()
    return document_for(genome, architecture, parts), genome, architecture


def test_player_makes_the_engines_choice_in_the_quick_games():
    games, _, differ = play_both(QUICK_SEEDS)
    assert not differ
    assert games == recorded(QUICK_SEEDS)


@pytest.mark.slow
def test_player_makes_the_engines_choice_on_every_held_out_state():
    seeds = [game["seed"] for game in GOLDEN["games"]]
    games, states, differ = play_both(seeds)
    assert not differ, f"{len(differ)} states differ, first {differ[0]}"
    assert games == recorded(seeds)
    assert states == STATES_IN_BLOCK


@pytest.mark.parametrize("activations", [
    ("tanh", "tanh"), ("relu",), ("leaky_relu", "sigmoid"), ("linear", "relu", "tanh"),
])
def test_round_trip_is_exact(activations, tmp_path):
    # Weights spread over many orders of magnitude, so a lossy number format shows.
    _, genome, architecture = small_model(activations, seed=len(activations))
    exponents = np.random.default_rng(1).integers(-30, 30, genome.size)
    genome = (genome * 10.0 ** exponents).astype(np.float32)
    path = tmp_path / "model.json"
    model_file.write(document_for(genome, architecture), path)

    restored, restored_architecture = model_file.to_genome(model_file.read(path))
    assert restored_architecture == architecture
    assert restored.dtype == np.float32
    assert restored.tobytes() == genome.tobytes()


@pytest.mark.parametrize("activations", [("relu", "relu"), ("leaky_relu",), ("sigmoid", "tanh")])
def test_player_follows_the_engine_with_every_activation(activations):
    document, genome, architecture = small_model(activations, seed=3, scale=2.0)
    player = Player(document)
    network = PolicyNetwork(architecture).load(genome)
    rng = np.random.default_rng(4)
    for _ in range(300):
        x = rng.normal(0.0, 2.0, 5).astype(np.float32)
        engine = network.forward(x)
        assert player.outputs(x.tolist()) == pytest.approx(engine.tolist(), rel=1e-5, abs=1e-5)


def test_player_activations_match_the_engines():
    assert set(ACTIVATIONS) == set(ENGINE_ACTIVATIONS)
    points = [-80.0, -3.0, -0.5, 0.0, 0.25, 2.0, 80.0]
    for name, function in ACTIVATIONS.items():
        engine = ENGINE_ACTIVATIONS[name](np.array(points)).tolist()
        assert [function(z) for z in points] == pytest.approx(engine, rel=1e-12), name


@pytest.mark.parametrize("actions", [
    Actions(Choice("turn", ("left", "straight", "right"))),
    Actions(Range("throttle", -2.0, 3.0)),
    MOVE_AND_STRIDE,
    Actions(Range("x"), Choice("lane", 4), Range("y", 0.0, 10.0)),
], ids=["choice", "range", "choice-and-range", "three-parts"])
def test_player_acts_as_the_engine_decodes(actions):
    architecture = Architecture(input_size=5, hidden_layers=(7,), hidden_activations=("tanh",),
                                output_size=actions.size)
    genome = random_genome(architecture, np.random.default_rng(5), 1.0, 1.0)
    player = Player(document_for(genome, architecture, actions.describe()))
    network = PolicyNetwork(architecture).load(genome)
    assert player.action_names == actions.names

    def by_part(value):
        return value if len(actions.parts) > 1 else {actions.parts[0].name: value}

    rng = np.random.default_rng(6)
    for _ in range(300):
        x = rng.normal(0.0, 2.0, 5).astype(np.float32)
        mine = by_part(player.action(x.tolist()))
        engine = by_part(actions.decode(network.forward(x)))
        assert list(mine) == list(engine)
        for part in actions.parts:
            if isinstance(part, Range):
                assert mine[part.name] == pytest.approx(engine[part.name], rel=1e-5, abs=1e-5)
            else:
                # The engine delivers the option itself and the player its name.
                assert mine[part.name] == str(engine[part.name])


def test_a_version_1_file_reads_as_one_choice():
    document, _, _ = small_model(("tanh",), seed=7, scale=2.0)
    old = json.loads(json.dumps(document))
    version_1(old)
    current, earlier = Player(document), Player(old)
    assert earlier.action_names == current.action_names == ("a", "b", "c")
    rng = np.random.default_rng(8)
    for _ in range(100):
        x = rng.normal(0.0, 2.0, 5).tolist()
        assert earlier.action(x) == current.action(x)


def test_ties_go_to_the_earliest_action():
    document, _, _ = small_model(("tanh",))
    last = document["network"]["layers"][-1]
    last["weights"] = [[0.0] * last["inputs"] for _ in range(3)]
    last["biases"] = [0.5, 1.0, 1.0]
    assert Player(document).action([0.0] * 5) == "b"


def test_player_describes_the_model():
    player = Player.load(MODEL)
    assert player.parameter_count == 435
    assert player.describe().splitlines()[1:] == [
        "action: one of left, straight, right",
        "network: 11 -> 16 tanh -> 12 tanh -> 3 linear, 435 parameters",
    ]
    with pytest.raises(ValueError, match="takes 11 inputs, got 2"):
        player.action([0.0, 1.0])


def test_player_describes_each_action_part():
    document, _, _ = small_model(("tanh",), actions=MOVE_AND_STRIDE)
    assert Player(document).describe().splitlines()[1:3] == [
        "action move: one of left, stay, right",
        "action stride: from 0.5 to 1.5",
    ]


def test_the_command_line_prints_several_parts_as_json(tmp_path, capsys):
    document, _, _ = small_model(("tanh",), actions=MOVE_AND_STRIDE)
    path = tmp_path / "model.json"
    model_file.write(document, path)

    assert player_main([str(path), "0", "1", "0", "0.5", "-1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == Player.load(path).action([0.0, 1.0, 0.0, 0.5, -1.0])
    assert set(printed) == {"move", "stride"}


def broken(change):
    document, _, _ = small_model(("tanh",))
    change(document)
    return document


def _set(path, value):
    def change(document):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


def _then(*changes):
    def change(document):
        for each in changes:
            each(document)
    return change


def _two_parts(**stride):
    """Actions of a choice of two and a range, the range's fields replaced by `stride`."""
    return _set(("actions", "parts"), [
        {"name": "move", "kind": "choice", "options": ["a", "b"]},
        {"name": "stride", "kind": "range", "low": 0.5, "high": 1.5, **stride},
    ])


def test_a_choice_and_a_range_fill_three_outputs():
    check(broken(_two_parts()))


@pytest.mark.parametrize("change, message", [
    (_set(("format",), "snake-ai-model"), "not an Evolvarium model"),
    (_set(("version",), 3), "reads version 1 or 2; the file is version 3"),
    (_set(("version",), True), "the file is version True"),
    (_set(("inputs", "names"), []), "inputs.names must be a list"),
    (_set(("actions", "parts"), []), "actions.parts must be a list"),
    (_set(("actions",), {"names": ["a", "b", "c"], "choice": "argmax"}),
     "actions.parts must be a list"),
    (_set(("actions", "parts", 0), "move"), "action part 0 is not an object"),
    (_set(("actions", "parts", 0, "name"), ""), "action part 0 needs a name"),
    (_set(("actions", "parts", 0, "kind"), "vector"), "has kind 'vector'"),
    (_set(("actions", "parts", 0, "options"), ["a"]), "at least two option names"),
    (_set(("actions", "parts", 0, "options"), ["a", 2, "c"]), "at least two option names"),
    (_set(("actions", "parts", 0, "options"), ["a", "b", "a"]), "lists an option twice"),
    (_set(("actions", "parts", 0, "options"), ["a", "b"]),
     "gives 3 outputs, but the actions read 2"),
    (_two_parts(name="move"), "two action parts are named 'move'"),
    (_two_parts(low=None), "range 'stride' holds None"),
    (_two_parts(high=0.5), "needs low below high"),
    (_then(version_1, _set(("actions", "names"), ["a", 3, "c"])), "actions.names must be a list"),
    (_then(version_1, _set(("actions", "choice"), "sample")), "chosen by 'sample'"),
    (_then(version_1, _set(("actions", "names"), ["a", "b"])),
     "gives 3 outputs, but the actions read 2"),
    (_set(("network", "kind"), "graph"), "network kind is 'graph'"),
    (_set(("network", "layers"), []), "at least one layer"),
    (_set(("network", "layers", 0, "inputs"), 0), "whole numbers of inputs and outputs"),
    (_set(("network", "layers", 0, "inputs"), 4), "takes 4 inputs, but the model names 5"),
    (_set(("network", "layers", 1, "inputs"), 6), "takes 6 inputs, but layer 0 gives 7"),
    (_set(("network", "layers", 0, "activation"), "swish"), "unknown activation 'swish'"),
    (_set(("network", "layers", 0, "biases"), [0.0]), "needs 7 biases"),
    (_set(("network", "layers", 0, "weights", 2), [1.0]), "weight row 2 needs 5 numbers"),
    (_set(("network", "layers", 0, "weights", 1, 3), math.nan), "not a finite number"),
    (_set(("network", "layers", 1, "biases", 0), True), "not a finite number"),
    (_set(("network", "layers", 1, "biases", 0), "0.5"), "not a finite number"),
])
def test_broken_files_are_refused(change, message):
    with pytest.raises(ModelError, match=message):
        check(broken(change))


def test_a_file_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"format": "evolvarium-model",', encoding="utf-8")
    with pytest.raises(ModelError, match="not valid JSON"):
        Player.load(path)


def test_the_engine_refuses_a_network_it_cannot_hold():
    document, _, _ = small_model(("tanh",))
    document["network"]["layers"][-1]["activation"] = "sigmoid"
    check(document)
    with pytest.raises(ModelError, match="end in a linear layer"):
        model_file.to_genome(document)


def test_unknown_keys_are_ignored():
    document = json.loads(MODEL.read_text(encoding="utf-8"))
    document["extra"] = {"anything": [1, 2, 3]}
    document["network"]["layers"][0]["note"] = "ignored"
    assert Player(document).parameter_count == 435


def test_player_imports_only_the_standard_library():
    tree = ast.parse(PLAYER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "player.py must not import from the package"
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names, imported


def test_a_copied_player_runs_without_numpy(tmp_path):
    bare = [sys.executable, "-I", "-S"]
    without_numpy = subprocess.run([*bare, "-c", "import numpy"], capture_output=True)
    if without_numpy.returncode == 0:
        pytest.skip("numpy is importable even without site-packages here")

    shutil.copy(PLAYER, tmp_path / "player.py")
    shutil.copy(MODEL, tmp_path / "model.json")
    inputs = ["0", "0", "1", "0.5", "0.8", "0.1", "0.3", "-0.2", "0.4", "0.1", "0.05"]

    def run(*args):
        return subprocess.run([*bare, "player.py", *args], cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)

    described = run("model.json")
    assert described.returncode == 0, described.stderr
    assert "435 parameters" in described.stdout

    acted = run("model.json", *inputs)
    assert acted.returncode == 0, acted.stderr
    assert acted.stdout.strip() == Player.load(MODEL).action([float(v) for v in inputs])

    short = run("model.json", "1", "2")
    assert short.returncode == 1
    assert "takes 11 inputs, got 2" in short.stderr
