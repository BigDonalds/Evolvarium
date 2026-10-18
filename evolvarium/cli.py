"""Command line interface."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
import webbrowser
from pathlib import Path

from . import games, settings
from .config import CONFIG, Config
from .contract import settings_dict
from .events import EventBus, Note
from .evolution.pool import resolve_workers
from .held_out import (
    TargetError,
    candidates_from,
    choose_by_average,
    evaluate_champions,
    load_champion,
    resolve_targets,
    run_context,
)
from .persistence import checkpoint, export_model
from .settings import SettingsError
from .telemetry.history import lineage_summaries
from .telemetry.reporter import ConsoleReporter
from .trainer import Trainer
from .web.runs import Lab
from .web.server import DashboardServer

# Flags for the settings a sweep changes most. Each is the same as --set on the
# setting it names.
SHORTHANDS = {
    "--population": ("evolution.population_size", int, "genomes in each generation"),
    "--episodes": ("evolution.evaluation_episodes", int, "evaluation episodes per genome"),
    "--tournament": ("evolution.tournament_size", int,
                     "tournament size; higher is stronger selection pressure"),
    "--elites": ("evolution.elite_count", int,
                 "genomes carried forward unchanged each generation"),
    "--immigrants": ("evolution.immigrant_count", int,
                     "fresh random genomes injected each generation"),
    "--mutation-rate": ("evolution.mutation_rate", float,
                        "share of genes perturbed in an offspring"),
    "--mutation-strength": ("evolution.mutation_strength", float, "size of each perturbation"),
    "--max-mutation-rate": ("evolution.max_mutation_rate", float,
                            "ceiling the mutation rate widens to while the best score stalls"),
    "--crossover-rate": ("evolution.crossover_rate", float,
                         "share of offspring built from two parents"),
    "--workers": ("evolution.workers", int,
                  "evaluation worker processes; 0 uses every core but one, 1 is serial"),
    "--port": ("server.port", int, "dashboard port"),
}


def _prepare_console() -> None:
    """Ask the streams for UTF-8 where the terminal can take it.

    Console output is decoration, never a reason for a run to fail, so a stream
    that refuses is left exactly as it was.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _assignment(text: str) -> tuple[str, str]:
    name, equals, value = text.partition("=")
    if not equals or not name.strip():
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=VALUE")
    return name.strip(), value.strip()


def _game_config(game: str | None) -> Config:
    if game is None:
        return CONFIG
    return dataclasses.replace(CONFIG, game=game, settings=None)


def _configure(args: argparse.Namespace) -> Config:
    """The configuration the train command asks for, refused if it cannot train as asked."""
    assignments = []
    for flag, (name, _, _) in SHORTHANDS.items():
        value = getattr(args, flag[2:].replace("-", "_"))
        if value is not None:
            assignments.append((name, value))
    if args.no_hold:
        assignments.append(("server.hold_when_viewing", False))
    assignments.extend(args.set or [])

    config = settings.apply(_game_config(args.game), assignments)
    problems = []
    try:
        settings.check(config, named=[name for name, _ in assignments],
                       dashboard=not args.headless)
    except SettingsError as error:
        problems = error.problems
    if args.open and args.headless:
        problems.append("--open has no effect with --headless, which starts no dashboard")
    if problems:
        raise SettingsError(problems)
    return config


def _refused(verb: str, error: Exception) -> int:
    print(f"  cannot {verb}:")
    for problem in getattr(error, "problems", [str(error)]):
        print(f"    {problem}")
    return 1


def _banner(reporter: ConsoleReporter, trainer: Trainer, url: str) -> None:
    game = trainer.population.game
    reporter.banner(
        run_id=trainer.run_id,
        generation=trainer.population.generation,
        population=len(trainer.population.members),
        game=trainer.config.game,
        encoding=str(game.model_inputs()["encoding"]),
        inputs=game.observation.size,
        genome=trainer.population.architecture.genome_size,
        url=url,
        workers=trainer.workers,
    )


def _wind_down(trainer: Trainer, drop) -> None:
    """Finish the generation in progress after a Ctrl+C, or `drop()` it after a second."""
    print()
    trainer.events.publish(Note(
        "interrupted, finishing the current generation (Ctrl+C again to drop it)"
    ))
    trainer.request_stop()
    try:
        trainer.join()
    except KeyboardInterrupt:
        drop()
        # The trainer ends once the workers finish the genomes they are
        # scoring. A third Ctrl+C stops waiting for it here, though the
        # process still lets the workers finish as it exits.
        with contextlib.suppress(KeyboardInterrupt):
            trainer.join()


def _latest_checkpoint(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.glob("*/checkpoints/latest.npz"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def command_train(args: argparse.Namespace) -> int:
    try:
        config = _configure(args)
    except (SettingsError, games.UnknownGame) as error:
        return _refused("train", error)

    resume_from: Path | None = None
    if args.resume:
        resume_from = (
            Path(args.resume) if args.resume != "auto"
            else _latest_checkpoint(config.paths.runs_dir)
        )
        if resume_from is None or not resume_from.exists():
            print(f"  no checkpoint to resume from ({args.resume}); starting a new run")
            resume_from = None

    events = EventBus()
    reporter = ConsoleReporter(every=args.report_every)
    reporter.attach(events)
    try:
        trainer = Trainer(
            config,
            seed=args.seed,
            resume_from=resume_from,
            events=events,
            max_generations=args.generations,
        )
    except checkpoint.CheckpointMismatch as error:
        print(f"  cannot resume: {error}")
        return 1

    # This dashboard shows the one run, alongside the runs already on disk.
    lab = Lab(config, can_start=False)
    lab.adopt(trainer)
    server = DashboardServer(lab, config.server.host, config.server.port)
    served = server.start() if not args.headless else False

    _banner(reporter, trainer, server.url if served else "disabled")
    if served and args.open:
        webbrowser.open(server.url)

    # The run's final checkpoint and the closing summary both come from the
    # trainer thread as it finishes, so this side only decides when to stop.
    trainer.start()
    try:
        trainer.join()
    except KeyboardInterrupt:
        _wind_down(trainer, lambda: trainer.stop(timeout=0))

    trainer.stop()
    lab.close()
    server.stop()
    return 0


def command_serve(args: argparse.Namespace) -> int:
    assignments = []
    if args.port is not None:
        assignments.append(("server.port", args.port))
    if args.workers is not None:
        assignments.append(("evolution.workers", args.workers))
    if args.no_hold:
        assignments.append(("server.hold_when_viewing", False))
    assignments.extend(args.set or [])
    try:
        config = settings.apply(_game_config(args.game), assignments)
        settings.check(config, named=[name for name, _ in assignments], dashboard=True)
    except (SettingsError, games.UnknownGame) as error:
        return _refused("serve", error)

    server: DashboardServer | None = None

    def announce(trainer: Trainer) -> None:
        reporter = ConsoleReporter(every=args.report_every)
        reporter.attach(trainer.events)
        _banner(reporter, trainer, server.url)

    lab = Lab(config, can_start=True, announce=announce)
    server = DashboardServer(lab, config.server.host, config.server.port)
    if not server.start():
        return 1
    print(f"  dashboard     {server.url}")
    print(f"  runs          {config.paths.runs_dir}")
    print("  runs started on the page train here; Ctrl+C ends the one training and quits")
    print()
    if args.open:
        webbrowser.open(server.url)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        live = lab.live
        if live is not None:
            _wind_down(live.trainer, lambda: lab.close(timeout=0))

    lab.close()
    server.stop()
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else _latest_checkpoint(CONFIG.paths.runs_dir)
    if path is None or not path.exists():
        print("  no checkpoint found")
        return 1

    info = checkpoint.inspect(path)
    meta = checkpoint.read_meta(path)
    print(f"  file          {info.path}")
    print(f"  saved         {info.saved_at}")
    print(f"  size          {info.size_bytes / 1024:.1f} KiB")
    print(f"  generation    {info.generation}")
    print(f"  population    {info.population_size}")
    print(f"  best score    {'-' if info.best_score is None else f'{info.best_score:g}'}")
    print(f"  best fitness  {info.best_fitness:.2f}")
    print(f"  game          {info.game}")
    print(f"  encoding      {info.encoding}")
    print(f"  architecture  {meta.get('architecture')}")
    print(f"  episodes      {meta.get('total_episodes', 0):,}")

    recorded = None
    manifest_path = path.parent.parent / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.get("config") or {}
        print(f"  seed          {manifest.get('seed')}")
        # Settings the engine no longer has are left out.
        current = settings_dict(CONFIG.evolution)
        tuned = {k: v for k, v in recorded.get("evolution", {}).items()
                 if k in current and v != current[k]}
        print(f"  evolution     {tuned or 'defaults'}")

    saved = checkpoint.saved_settings(meta, recorded)
    try:
        defaults = settings_dict(games.get(info.game).Settings())
    except games.UnknownGame:
        defaults = {}
    # Through JSON, so tuples compare equal to the lists a checkpoint stores.
    tuned = _changed(saved, json.loads(json.dumps(defaults, default=str)))
    if saved:
        print(f"  settings      {tuned or 'defaults'}")
    else:
        print("  settings      not recorded (checkpoint predates config capture)")
    return 0


def _changed(values: dict, defaults: dict, prefix: str = "") -> dict:
    """Settings that differ from their defaults, by dotted path."""
    changed = {}
    for name, value in values.items():
        base = defaults.get(name, object())
        if isinstance(value, dict) and isinstance(base, dict):
            changed.update(_changed(value, base, f"{prefix}{name}."))
        elif value != base:
            changed[f"{prefix}{name}"] = value
    return changed


def command_export(args: argparse.Namespace) -> int:
    path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(CONFIG.paths.runs_dir)
    if path is None or not path.exists():
        print("  no checkpoint found")
        return 1

    import numpy as np

    # The rules the run was trained under, which the model file records.
    try:
        context = run_context(path)
    except ValueError as error:
        print(f"  cannot export {path}: {error}")
        return 1
    selection = None
    if args.pick == "best-average":
        requested = args.workers if args.workers is not None else CONFIG.evolution.workers
        candidates = candidates_from(path)
        print(f"  choosing among {len(candidates)} genomes by their average; "
              "this plays a few thousand games")
        choice = choose_by_average(candidates, context, resolve_workers(requested))
        genome = choice.genome
        selection = choice.to_dict()
        against = ("" if choice.champion_mean is None
                   else f", against {choice.champion_mean:.1f} for the stored champion")
        print(f"  picked {choice.source}: {choice.mean:.1f} over {choice.games} "
              f"selection games{against}")
    else:
        with np.load(path, allow_pickle=False) as archive:
            if "best_genome" not in archive:
                print("  checkpoint holds no champion genome")
                return 1
            genome = np.array(archive["best_genome"], dtype=np.float32)

    config = CONFIG
    population_meta = checkpoint.read_meta(path)
    from .nn.network import Architecture

    architecture = Architecture.from_dict(population_meta["architecture"])
    destination = config.paths.exports_dir / f"{args.name}.json"
    run_dir = path.parent.parent

    report = export_model.build_report(
        generation=int(population_meta.get("generation", 0)),
        genome=genome,
        summaries=lineage_summaries(run_dir),
        game=games.create(context.game, context.settings),
        architecture=architecture,
        run_id=run_dir.name,
        destination=destination,
    )
    if selection:
        report["selection"] = selection
    result = export_model.write_model(
        report=report,
        genome=genome,
        architecture=architecture,
        destination=destination,
        archive_dir=config.paths.exports_dir / "archive",
    )
    print(f"  wrote {result['path']} ({result['size_bytes'] / 1024:.1f} KiB)")
    if result["pt"]:
        print(f"  wrote {result['pt']}")
    else:
        print("  torch is not installed, so no .pt copy was written")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    held_out = CONFIG.held_out
    if args.episodes is not None or args.first_seed is not None:
        held_out = dataclasses.replace(
            held_out,
            episodes=args.episodes if args.episodes is not None else held_out.episodes,
            first_seed=args.first_seed if args.first_seed is not None else held_out.first_seed,
        )
    if held_out.episodes < 1:
        print("  --episodes must be at least 1")
        return 1

    targets = args.targets
    if not targets:
        newest = _latest_checkpoint(CONFIG.paths.runs_dir)
        if newest is None:
            print("  no checkpoint found")
            return 1
        targets = [str(newest)]
    try:
        champions = [
            load_champion(path) for path in resolve_targets(targets, CONFIG.paths.runs_dir)
        ]
    except TargetError as error:
        print(f"  {error}")
        return 1

    requested = args.workers if args.workers is not None else CONFIG.evolution.workers
    workers = resolve_workers(requested)
    count = len(champions)
    print(f"  scoring {count} champion{'s' if count != 1 else ''}, "
          f"{held_out.episodes} episodes each, on {workers} worker{'s' if workers != 1 else ''}")
    report = evaluate_champions(champions, held_out, workers, same_config=args.same_config)

    print()
    for line in report.lines():
        print(line)
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\n  wrote {destination}")
    return 0


def command_profile(args: argparse.Namespace) -> int:
    import zipfile

    import numpy as np

    from .profiling import profile

    path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(CONFIG.paths.runs_dir)
    if path is None or not path.exists():
        print("  no checkpoint found")
        return 1
    try:
        meta = checkpoint.read_meta(path)
        context = run_context(path, meta)
        with np.load(path, allow_pickle=False) as archive:
            population = np.array(archive["genomes"], dtype=np.float32)
            champion = (np.array(archive["best_genome"], dtype=np.float32)
                        if "best_genome" in archive.files else None)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        print(f"  cannot read {path}: {error}")
        return 1

    if args.member is None:
        if champion is None:
            print("  checkpoint holds no champion yet; name a population member with --member")
            return 1
        genome, chosen = champion, "stored champion"
    elif 0 <= args.member < len(population):
        genome, chosen = population[args.member], f"population member {args.member}"
    else:
        print(f"  --member must be between 0 and {len(population) - 1}")
        return 1

    games = args.games if args.games is not None else CONFIG.held_out.episodes
    if games < 1:
        print("  --games must be at least 1")
        return 1
    seeds = dataclasses.replace(CONFIG.held_out, episodes=games).seeds
    requested = args.workers if args.workers is not None else CONFIG.evolution.workers
    trials = int((meta.get("evolution") or {}).get("evaluation_episodes")
                 or CONFIG.evolution.evaluation_episodes)
    run = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name

    print(f"  profiling the {chosen} of {run}, generation {meta.get('generation', 0)}, "
          f"on {games} held-out game{'s' if games != 1 else ''}")
    result = profile(
        context, genome, population, seeds,
        source=f"{chosen}, {run} generation {meta.get('generation', 0)}",
        workers=resolve_workers(requested), rounds=args.rounds, trials=trials,
        note=lambda text: print(f"  {text}", flush=True),
    )

    print()
    for line in result.lines():
        print(line)
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_dict(), indent=2, default=str),
                               encoding="utf-8")
        print(f"\n  wrote {destination}")
    return 0


def command_settings(args: argparse.Namespace) -> int:
    try:
        config = games.resolve(_game_config(args.game))
    except games.UnknownGame as error:
        print(f"  {error}")
        return 1
    current = settings.values(config)
    allowed = settings.choices(config.game)
    width = max(map(len, current))
    group = None
    for name, value in current.items():
        if group is not None and name.partition(".")[0] != group:
            print()
        group = name.partition(".")[0]
        line = f"  {name:<{width}}  {settings.show(value)}"
        if name in allowed:
            line += f"  ({', '.join(map(str, allowed[name]))})"
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolvarium", description="Neuroevolution of game-playing agents"
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="run evolution with the live dashboard")
    train.add_argument("--game", default=None,
                       help=f"a built-in game or package.module:Class, default {CONFIG.game}")
    train.add_argument(
        "--set", action="extend", nargs="+", type=_assignment, metavar="NAME=VALUE",
        help="set any setting of the game or the engine by name, such as board.width=12 "
             "or network.hidden_layers=32,16; the settings command lists them",
    )
    train.add_argument("--generations", type=int, default=None, help="stop after this many")
    for flag, (name, kind, about) in SHORTHANDS.items():
        train.add_argument(flag, type=kind, default=None, help=f"{about} ({name})")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--resume", nargs="?", const="auto", default=None,
                       help="resume from a checkpoint path, or 'auto' for the newest")
    train.add_argument("--report-every", type=int, default=1)
    train.add_argument("--headless", action="store_true", help="do not start the dashboard")
    train.add_argument("--no-hold", action="store_true",
                       help="never pause for the dashboard between generations "
                            "(server.hold_when_viewing=false)")
    train.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    train.set_defaults(func=command_train)

    serve = sub.add_parser(
        "serve", help="open the dashboard on every run, to start, watch and compare runs"
    )
    serve.add_argument("--game", default=None,
                       help=f"the game the page offers first, default {CONFIG.game}")
    serve.add_argument(
        "--set", action="extend", nargs="+", type=_assignment, metavar="NAME=VALUE",
        help="settings every run started on the page begins from; the page can change "
             "all but the server, paths and held_out ones",
    )
    serve.add_argument("--port", type=int, default=None, help="dashboard port (server.port)")
    serve.add_argument("--workers", type=int, default=None,
                       help="evaluation worker processes for the runs started here "
                            "(evolution.workers)")
    serve.add_argument("--report-every", type=int, default=1)
    serve.add_argument("--no-hold", action="store_true",
                       help="never pause a run for the dashboard between generations "
                            "(server.hold_when_viewing=false)")
    serve.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    serve.set_defaults(func=command_serve)

    inspect = sub.add_parser("inspect", help="describe a checkpoint")
    inspect.add_argument("path", nargs="?", default=None)
    inspect.set_defaults(func=command_inspect)

    export = sub.add_parser(
        "export", help="write a standalone JSON model, and a .pt copy if torch is installed"
    )
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--name", default="evolvarium",
                        help="file stem for the export, so several models stay apart; "
                             "e.g. v3-seed99-gen140-73")
    export.add_argument(
        "--pick", choices=("champion", "best-average"), default="champion",
        help="champion exports the genome with the run's best single game; "
             "best-average plays every genome the run holds and exports the steadiest, "
             "which is slower and far more reliable",
    )
    export.add_argument("--workers", type=int, default=None,
                        help="for --pick best-average; 0 uses every core but one")
    export.set_defaults(func=command_export)

    evaluate = sub.add_parser(
        "evaluate", help="score champions on held-out episodes and compare configurations"
    )
    evaluate.add_argument(
        "targets", nargs="*",
        help="run directories, run ids, checkpoint files, exported .json models or glob "
             "patterns; defaults to the newest run",
    )
    evaluate.add_argument(
        "--episodes", type=int, default=None,
        help=f"held-out episodes per champion, default {CONFIG.held_out.episodes}; "
             "other values are not comparable with HISTORY.md",
    )
    evaluate.add_argument(
        "--first-seed", type=int, default=None,
        help=f"first held-out seed, default {CONFIG.held_out.first_seed}",
    )
    evaluate.add_argument("--workers", type=int, default=None,
                          help="0 uses every core but one, 1 is serial")
    evaluate.add_argument(
        "--same-config", action="store_true",
        help="treat every champion as one configuration, for runs that predate run.json",
    )
    evaluate.add_argument("--json", default=None, metavar="PATH",
                          help="also write every episode's score to this file")
    evaluate.set_defaults(func=command_evaluate)

    profile = sub.add_parser(
        "profile", help="measure where the time in a move goes, and the worker pool's overhead"
    )
    profile.add_argument("checkpoint", nargs="?", default=None,
                         help="checkpoint file; defaults to the newest run")
    profile.add_argument("--member", type=int, default=None,
                         help="profile this population member's games instead of the champion's")
    profile.add_argument("--games", type=int, default=None,
                         help=f"held-out games to play, default {CONFIG.held_out.episodes}")
    profile.add_argument("--rounds", type=int, default=2,
                         help="times to score the whole population on the pool; 0 skips it")
    profile.add_argument("--workers", type=int, default=None,
                         help="pool size; 0 uses every core but one")
    profile.add_argument("--json", default=None, metavar="PATH",
                         help="also write the measurements to this file")
    profile.set_defaults(func=command_profile)

    listing = sub.add_parser(
        "settings", help="list every setting a game trains under, its value and its choices"
    )
    listing.add_argument("--game", default=None,
                         help=f"a built-in game or package.module:Class, default {CONFIG.game}")
    listing.set_defaults(func=command_settings)

    return parser


def main(argv: list[str] | None = None) -> int:
    _prepare_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["train", *(argv or [])])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
