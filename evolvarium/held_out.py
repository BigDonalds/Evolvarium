"""Scoring finished champions on episodes no run trained on.

A run's own best score comes from the seeds its population was selected on, so
it flatters the champion that won them. Every comparison in HISTORY.md scores
champions on one fixed block of held-out seeds instead: one episode per seed,
exploration off, the same block for every champion.

Two kinds of noise show up here and only one of them decides a comparison.
Spread across held-out episodes says how consistently one champion plays, and
over 60 episodes it averages down to a point or two. Spread across training
seeds says how much the champion itself changes when evolution is re-run, and it
is several times larger. Group statistics are therefore computed across runs,
never across episodes. A configuration run at a single seed has no measure of
that noise at all, and the report says so instead of printing an interval.
"""

from __future__ import annotations

import dataclasses
import glob
import gzip
import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import games
from .config import CONFIG, HeldOutConfig
from .contract import load_settings
from .evolution.evaluation import EvalContext, EvalResult, make_runner
from .evolution.pool import EvaluationPool, resolve_workers
from .persistence import checkpoint, model_file

# Engine sections that decide which champion a run ends with, beside the game
# and its settings. Paths, the dashboard, retention and the worker count change
# how a run is watched or how fast it goes, never its result, so runs that
# differ only there share a configuration.
SEARCH_SECTIONS = ("network", "evolution")
IGNORED_SETTINGS = {"evolution": {"workers"}}

# Two-sided 95% quantiles of Student's t, by degrees of freedom.
_T95 = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447),
    (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228), (12, 2.179), (15, 2.131),
    (20, 2.086), (30, 2.042), (60, 2.000), (120, 1.980),
)


class TargetError(ValueError):
    """A target that names no usable checkpoint."""


@dataclass
class Champion:
    """A run's best genome, the settings that produced it, and its held-out play."""

    run_id: str
    path: Path
    seed: int | None
    generation: int
    # None when no episode of the run was ever recorded as a best.
    train_best: float | None
    genome: np.ndarray
    context: EvalContext
    # Search settings from the run's manifest; None for runs that predate it.
    settings: dict | None
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ends: Counter = field(default_factory=Counter)
    # Whether the game counts a higher score as better.
    higher: bool = True

    @property
    def mean(self) -> float:
        return float(self.scores.mean())


@dataclass
class Group:
    """Champions of one configuration, differing only by training seed."""

    label: str
    generation: int | None
    champions: list[Champion]

    @property
    def title(self) -> str:
        if self.generation is None:
            return f"{self.label}, mixed generations"
        return f"{self.label}, generation {self.generation}"

    @property
    def means(self) -> np.ndarray:
        return np.array([c.mean for c in self.champions])

    @property
    def mean(self) -> float:
        return float(self.means.mean())

    @property
    def spread(self) -> float | None:
        """Standard deviation between runs, or None when there is only one."""
        return float(self.means.std(ddof=1)) if len(self.champions) > 1 else None

    def interval(self) -> tuple[float, float] | None:
        """95% interval on the configuration's mean champion."""
        if self.spread is None:
            return None
        n = len(self.champions)
        half = t_critical(n - 1) * self.spread / math.sqrt(n)
        return self.mean - half, self.mean + half


@dataclass(frozen=True)
class Comparison:
    label: str
    reference: str
    difference: float
    # Half-width of the 95% interval on the difference. None when either side
    # was run at a single seed, which leaves seed noise unmeasured.
    half_width: float | None

    @property
    def verdict(self) -> str:
        if self.half_width is None:
            return "one run on a side; cannot be told apart from seed luck"
        if abs(self.difference) > self.half_width:
            return "larger than seed noise"
        return "within seed noise"


@dataclass
class Report:
    held_out: HeldOutConfig
    groups: list[Group]
    comparisons: list[Comparison]

    def lines(self) -> list[str]:
        block = self.held_out
        out = [
            f"  held-out: {block.episodes} episodes per champion, seeds "
            f"{block.first_seed}-{block.first_seed + block.episodes - 1}, exploration off"
        ]
        if block != HeldOutConfig():
            out.append("  not the standard block, so not comparable with HISTORY.md")

        for group_ in self.groups:
            n = len(group_.champions)
            out += ["", f"  {group_.title}  ({n} run{'s' if n != 1 else ''})"]
            out.extend(_group_notes(group_))
            out.append(
                f"    {'run':<20} {'seed':>5} {'train best':>10} {'mean':>6} {'median':>7} "
                f"{'p25':>5} {'worst':>6} {'best':>5}  ends"
            )
            for c in group_.champions:
                total = sum(c.ends.values())
                ends = "  ".join(
                    f"{end} {count / total:.0%}" for end, count in c.ends.most_common(2)
                )
                seed = "?" if c.seed is None else str(c.seed)
                train_best = "-" if c.train_best is None else f"{c.train_best:g}"
                # The quartile is the one on the losing side, whichever way the
                # game counts.
                low, high = c.scores.min(), c.scores.max()
                worst, best = (low, high) if c.higher else (high, low)
                quartile = np.percentile(c.scores, 25 if c.higher else 75)
                out.append(
                    f"    {c.run_id:<20} {seed:>5} {train_best:>10} {c.mean:>6.1f} "
                    f"{np.median(c.scores):>7.1f} {quartile:>5.1f} "
                    f"{worst:>6g} {best:>5g}  {ends}"
                )
            interval = group_.interval()
            if interval is not None:
                out.append(
                    f"    mean of champions {group_.mean:.1f}, 95% interval "
                    f"{interval[0]:.1f} to {interval[1]:.1f}, "
                    f"spread between runs {group_.spread:.1f}"
                )

        if self.comparisons:
            out += ["", f"  against {self.groups[0].title}"]
            width = max(len(c.label) for c in self.comparisons)
            for c in self.comparisons:
                span = ""
                if c.half_width is not None:
                    span = (f"95% interval {c.difference - c.half_width:+.1f} "
                            f"to {c.difference + c.half_width:+.1f}   ")
                out.append(f"    {c.label:<{width}}  {c.difference:+6.1f}   {span}{c.verdict}")
        return out

    def to_dict(self) -> dict:
        return {
            "held_out": dataclasses.asdict(self.held_out),
            "groups": [
                {
                    "label": g.label,
                    "generation": g.generation,
                    "mean": g.mean,
                    "spread": g.spread,
                    "champions": [
                        {
                            "run_id": c.run_id,
                            "checkpoint": str(c.path),
                            "seed": c.seed,
                            "generation": c.generation,
                            "train_best": c.train_best,
                            "mean": c.mean,
                            "scores": [_plain_number(s) for s in c.scores],
                            "ends": dict(c.ends),
                        }
                        for c in g.champions
                    ],
                }
                for g in self.groups
            ],
            "comparisons": [
                {**dataclasses.asdict(c), "verdict": c.verdict} for c in self.comparisons
            ],
        }


def _plain_number(value) -> int | float:
    value = float(value)
    return int(value) if value.is_integer() else value


def _group_notes(group_: Group) -> list[str]:
    notes = []
    unrecorded = sum(c.settings is None for c in group_.champions)
    if unrecorded and unrecorded < len(group_.champions):
        notes.append(f"    includes {unrecorded} run(s) whose configuration was not recorded")
    seeds = Counter(c.seed for c in group_.champions if c.seed is not None)
    for seed, count in sorted(seeds.items()):
        if count > 1:
            notes.append(
                f"    seed {seed} appears {count} times; a configuration re-run at the "
                "same seed repeats itself exactly, so the copies add no information"
            )
    return notes


# ------------------------------------------------------------------- targets

def resolve_targets(targets: list[str], runs_dir: Path) -> list[Path]:
    """Turn run directories, run ids, checkpoint files, model files or patterns into files.

    Patterns are expanded here as well as by the shell, because PowerShell hands
    them to Python untouched.
    """
    found: list[Path] = []
    for target in targets:
        if any(ch in target for ch in "*?["):
            matches = sorted(glob.glob(target)) or sorted(glob.glob(str(runs_dir / target)))
            paths = [p for p in (_checkpoint_in(Path(m)) for m in matches) if p is not None]
            if not paths:
                raise TargetError(f"{target!r} matches no checkpoint or model file")
            found.extend(paths)
            continue
        path = _checkpoint_in(Path(target)) or _checkpoint_in(runs_dir / target)
        if path is None:
            raise TargetError(
                f"{target!r} is not a checkpoint, a model file, a run directory or a run id"
            )
        found.append(path)
    # A pattern and an explicit path can name the same run; score it once.
    return list(dict.fromkeys(p.resolve() for p in found))


def _checkpoint_in(path: Path) -> Path | None:
    if path.is_file() and path.suffix in (".npz", ".json"):
        return path
    latest = path / "checkpoints" / "latest.npz"
    return latest if latest.is_file() else None


def load_champion(path: Path) -> Champion:
    """Read a checkpoint's best genome and the conditions it was trained under."""
    if path.suffix == ".json":
        return _model_champion(path)
    try:
        meta = checkpoint.read_meta(path)
        with np.load(path, allow_pickle=False) as archive:
            genome = (
                np.array(archive["best_genome"], dtype=np.float32)
                if "best_genome" in archive.files else None
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise TargetError(f"cannot read {path}: {error}") from error
    if genome is None:
        raise TargetError(f"{path} holds no champion yet; no episode in that run has scored")

    run_dir = _run_dir(path)
    manifest = _read_manifest(run_dir / "run.json")
    recorded = manifest.get("config") if manifest else None
    try:
        context = run_context(path, meta)
    except ValueError as error:
        raise TargetError(f"cannot score {path}: {error}") from error
    return Champion(
        run_id=run_dir.name,
        path=path,
        seed=manifest.get("seed") if manifest else None,
        generation=int(meta.get("generation", 0)),
        train_best=meta.get("best_ever_score"),
        genome=genome,
        context=context,
        settings=search_settings(recorded) if recorded else None,
        higher=_higher(context),
    )


def _model_champion(path: Path) -> Champion:
    """An exported model, scored under the rules its file records."""
    try:
        document = model_file.read(path)
        genome, architecture = model_file.to_genome(document)
    except (OSError, model_file.ModelError) as error:
        raise TargetError(f"cannot read {path}: {error}") from error
    name = (document.get("game") or {}).get("name")
    encoding = (document.get("inputs") or {}).get("encoding")
    try:
        if not isinstance(name, str):
            raise ValueError("no game is named")
        settings = games.get(name).settings_from_model(document)
        _check_actions(document, games.create(name, settings))
    except ValueError as error:
        raise TargetError(
            f"{path} is a model of {name!r} with {encoding!r} inputs, "
            f"which this engine cannot play: {error}"
        ) from error
    source = document.get("source") or {}
    training = (document.get("report") or {}).get("training") or {}
    context = EvalContext(
        game=name,
        settings=settings,
        architecture=architecture.to_dict(),
        exploration_rate=0.0,
        replay_steps=CONFIG.history.replay_steps,
    )
    return Champion(
        run_id=str(source.get("run") or path.stem),
        path=path,
        seed=None,
        generation=int(source.get("generation") or 0),
        train_best=training.get("best_score"),
        genome=genome,
        context=context,
        settings=None,
        higher=_higher(context),
    )


def _check_actions(document: dict, game) -> None:
    """Raise ValueError when `game` would read the model's outputs as other actions."""
    actions = document["actions"]
    if "parts" in actions:
        recorded, expected = actions["parts"], game.actions.describe()
    else:
        # A version 1 file names its outputs and nothing more.
        recorded, expected = actions["names"], list(game.actions.names)
    if recorded != expected:
        raise ValueError(f"the model acts by {recorded}, but the game takes {expected}")


def _run_dir(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def run_context(path: Path, meta: dict | None = None) -> EvalContext:
    """The rules a checkpoint's run was trained under, ready for scoring.

    Replaying under the run's own rules matters: a rule whose default has since
    changed would otherwise score a different game.
    """
    meta = meta if meta is not None else checkpoint.read_meta(path)
    manifest = _read_manifest(_run_dir(path) / "run.json")
    recorded = (manifest.get("config") if manifest else None) or {}
    game, _ = checkpoint.saved_game(meta)
    settings = checkpoint.saved_settings(meta, recorded)
    return EvalContext(
        game=game,
        # Settings added since the run was recorded take their defaults.
        settings=load_settings(games.get(game).Settings, settings, strict=False),
        architecture=meta["architecture"],
        exploration_rate=0.0,
        replay_steps=CONFIG.history.replay_steps,
    )


def _higher(context: EvalContext) -> bool:
    return games.get(context.game).direction == "higher"


def _encoding(context: EvalContext) -> str:
    return str(games.create(context.game, context.settings).model_inputs()["encoding"])


def _read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------ configurations

def search_settings(config: dict) -> dict:
    """The part of a recorded configuration that can change a run's result.

    Manifests from before games carried their own settings are read into the
    same shape, and settings the engine no longer has are left out, so an old
    run and a new one under the same rules group together.
    """
    settings = {
        "game": str(config.get("game", "snake")),
        "settings": checkpoint.saved_settings({}, config),
    }
    for section in SEARCH_SECTIONS:
        names = {f.name for f in dataclasses.fields(getattr(CONFIG, section))}
        names -= IGNORED_SETTINGS.get(section, set())
        settings[section] = {
            name: value for name, value in (config.get(section) or {}).items() if name in names
        }
    return settings


def describe(settings: dict) -> str:
    """Name a configuration by how it differs from the current defaults.

    A game's own settings are named by their path inside its settings, such as
    "features.spec"; the engine's by section, such as "evolution.elite_count".
    """
    game = settings.get("game", CONFIG.game)
    try:
        base = games.resolve(dataclasses.replace(CONFIG, game=game))
    except games.UnknownGame:
        base = CONFIG
    # Through JSON, so tuples compare equal to the lists a manifest stores.
    defaults = search_settings(json.loads(json.dumps(dataclasses.asdict(base), default=str)))
    changes = [] if game == CONFIG.game else [f"game={game}"]
    _differences(settings.get("settings", {}), defaults["settings"], "", changes)
    for section in SEARCH_SECTIONS:
        _differences(settings.get(section, {}), defaults[section], f"{section}.", changes)
    return ", ".join(changes) or "defaults"


def _differences(values: dict, base: dict, prefix: str, out: list[str]) -> None:
    for name, value in values.items():
        if name not in base:
            continue
        if isinstance(value, dict) and isinstance(base[name], dict):
            _differences(value, base[name], f"{prefix}{name}.", out)
        elif value != base[name]:
            shown = value if isinstance(value, str) else json.dumps(value)
            out.append(f"{prefix}{name}={shown}")


def group(champions: list[Champion], same_config: bool = False) -> list[Group]:
    """Collect champions into configurations, in the order they were named."""
    if same_config:
        labels = {describe(c.settings) for c in champions if c.settings is not None}
        label = labels.pop() if len(labels) == 1 else "all champions, as one configuration"
        generations = {c.generation for c in champions}
        generation = generations.pop() if len(generations) == 1 else None
        return [Group(label, generation, list(champions))]

    groups: dict[str, Group] = {}
    for c in champions:
        if c.settings is not None:
            identity = json.dumps(c.settings, sort_keys=True)
            label = describe(c.settings)
        else:
            # Known only by game, inputs and network shape. Kept apart rather
            # than assumed to match anything else. An exported model records
            # its game and inputs but not the search that produced it.
            exported = c.path.suffix == ".json"
            encoding = _encoding(c.context)
            identity = ("exported " if exported else "unrecorded ") + json.dumps(
                [c.context.game, encoding, c.context.architecture], sort_keys=True
            )
            kind = "exported model" if exported else "configuration not recorded"
            label = f"{kind}, {encoding}"
        key = f"{identity} @ {c.generation}"
        if key not in groups:
            groups[key] = Group(label, c.generation, [])
        groups[key].champions.append(c)
    return list(groups.values())


# ------------------------------------------------------------------- scoring

def score(champions: list[Champion], seeds: list[int], workers: int) -> None:
    """Play every champion once on every seed, filling in scores and endings.

    Champions trained under identical rules share a pool. One from a different
    game or different settings gets its own, since a pool is built around one
    context.
    """
    batches: dict[str, list[Champion]] = {}
    for c in champions:
        key = json.dumps(dataclasses.asdict(c.context), sort_keys=True, default=str)
        batches.setdefault(key, []).append(c)

    for batch in batches.values():
        tasks = [(c.genome, seed, 1, False) for c in batch for seed in seeds]
        results = _play(batch[0].context, tasks, workers)
        for i, c in enumerate(batch):
            chunk = results[i * len(seeds):(i + 1) * len(seeds)]
            c.scores = np.array([r.score for r in chunk], dtype=float)
            c.ends = Counter(r.end for r in chunk)


def _play(ctx: EvalContext, tasks: list[tuple], workers: int) -> list[EvalResult]:
    workers = min(resolve_workers(workers), len(tasks))
    if workers <= 1:
        runner = make_runner(ctx)
        return [runner.score(*task) for task in tasks]
    with EvaluationPool(ctx, workers) as pool:
        return pool.map(tasks)


def t_critical(dof: float) -> float:
    """Two-sided 95% t quantile, rounded toward caution.

    Welch degrees of freedom are fractional and the table is sparse, so this
    takes the entry at or below `dof`. That can only widen an interval.
    """
    value = _T95[0][1]
    for tabulated, quantile in _T95:
        if dof >= tabulated:
            value = quantile
    return value


def compare(reference: Group, other: Group) -> Comparison:
    """Welch comparison of two configurations' mean champions."""
    difference = other.mean - reference.mean
    half_width = None
    if reference.spread is not None and other.spread is not None:
        n_ref, n_other = len(reference.champions), len(other.champions)
        a = reference.spread ** 2 / n_ref
        b = other.spread ** 2 / n_other
        denominator = a * a / (n_ref - 1) + b * b / (n_other - 1)
        dof = (a + b) ** 2 / denominator if denominator > 0 else n_ref + n_other - 2
        half_width = t_critical(dof) * math.sqrt(a + b)
    return Comparison(other.title, reference.title, difference, half_width)


def evaluate_champions(
    champions: list[Champion], held_out: HeldOutConfig, workers: int, same_config: bool = False
) -> Report:
    """Score, group and compare. The first configuration named is the reference."""
    score(champions, held_out.seeds, workers)
    groups = group(champions, same_config)
    comparisons = [compare(groups[0], g) for g in groups[1:]]
    return Report(held_out, groups, comparisons)


# ------------------------------------------------------------------ choosing

# Games used to choose a genome. They sit apart from the held-out block, so a
# pick is never judged on the games that chose it.
SELECTION_FIRST_SEED = 800_000
SCREEN_GAMES = 8
FINALISTS = 5
FINAL_GAMES = 40


@dataclass
class Choice:
    """The genome picked from a run, and how the run's own champion compared."""

    genome: np.ndarray
    source: str
    candidates: int
    games: int
    mean: float
    champion_mean: float | None

    def to_dict(self) -> dict:
        return {
            "rule": "best average",
            "source": self.source,
            "candidates": self.candidates,
            "games": self.games,
            "selection_mean": round(self.mean, 2),
            "stored_champion_mean": (
                None if self.champion_mean is None else round(self.champion_mean, 2)
            ),
            "note": "means over the games that made the choice flatter the winner; "
                    "score it on held-out games for an honest figure",
        }


def candidates_from(path: Path) -> list[tuple[str, np.ndarray]]:
    """Every genome a run can offer: its champion, its population, its past champions.

    The stored champion is listed first so that, when a population member or an
    archived champion is the same genome, it keeps the name people will recognise.
    """
    found: list[tuple[str, np.ndarray]] = []
    with np.load(path, allow_pickle=False) as archive:
        if "best_genome" in archive.files:
            found.append(("stored champion", np.array(archive["best_genome"], dtype=np.float32)))
        for index, genome in enumerate(np.array(archive["genomes"], dtype=np.float32)):
            found.append((f"population member {index}", genome))
    for replay in sorted((_run_dir(path) / "replays").glob("gen_*.json.gz")):
        try:
            payload = json.loads(gzip.open(replay, "rt", encoding="utf-8").read())
        except (OSError, ValueError):
            continue
        if payload.get("best_genome"):
            found.append((f"champion held at generation {payload.get('generation')}",
                          np.array(payload["best_genome"], dtype=np.float32)))

    unique, seen = [], set()
    for source, genome in found:
        key = genome.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((source, genome))
    return unique


def choose_by_average(
    candidates: list[tuple[str, np.ndarray]],
    context: EvalContext,
    workers: int,
    *,
    score=None,
    screen_games: int = SCREEN_GAMES,
    finalists: int = FINALISTS,
    final_games: int = FINAL_GAMES,
) -> Choice:
    """Pick the genome with the best average: screen widely, then decide narrowly.

    A run's stored champion holds its title for one best game, which is the wrong
    thing to choose on. Under a 10,000-step cap it crowned a genome scoring 44.1
    held-out while the same population held one scoring 81.5.

    Every candidate plays `screen_games` games. The leaders, and the stored
    champion so there is always a like-for-like comparison, then play
    `final_games` more, and the pick is the best average over all of them. A
    lucky screen can put a genome in the final; it cannot win the final on its own.

    `score(genomes, seeds)` returns a candidates-by-seeds matrix of scores, and
    defaults to playing the games under `context`. Without a context the scores
    are taken as higher-is-better.
    """
    # Scores are flipped for a game that wants them low, so "best" below is
    # always the largest.
    sign = 1.0 if context is None or _higher(context) else -1.0
    if score is None:
        def score(genomes, seeds):
            return _score_matrix(context, genomes, seeds, workers)

    screen_seeds = list(range(SELECTION_FIRST_SEED, SELECTION_FIRST_SEED + screen_games))
    final_start = SELECTION_FIRST_SEED + screen_games
    final_seeds = list(range(final_start, final_start + final_games))

    genomes = [genome for _, genome in candidates]
    screen = sign * np.asarray(score(genomes, screen_seeds), dtype=float)
    order = np.argsort(-screen.mean(axis=1), kind="stable")
    chosen = [int(i) for i in order[:finalists]]
    stored = next((i for i, (source, _) in enumerate(candidates)
                   if source == "stored champion"), None)
    if stored is not None and stored not in chosen:
        chosen.append(stored)

    final = sign * np.asarray(score([genomes[i] for i in chosen], final_seeds), dtype=float)
    totals = np.concatenate([screen[chosen], final], axis=1).mean(axis=1)
    best = int(np.argmax(totals))
    winner = chosen[best]
    return Choice(
        genome=genomes[winner],
        source=candidates[winner][0],
        candidates=len(candidates),
        games=screen_games + final_games,
        mean=float(sign * totals[best]),
        champion_mean=(
            float(sign * totals[chosen.index(stored)]) if stored is not None else None
        ),
    )


def _score_matrix(context: EvalContext, genomes, seeds, workers: int) -> np.ndarray:
    results = _play(context, [(g, s, 1, False) for g in genomes for s in seeds], workers)
    return np.array([r.score for r in results], dtype=float).reshape(len(genomes), len(seeds))
