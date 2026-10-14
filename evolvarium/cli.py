"""Command line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import webbrowser
from pathlib import Path

from .config import CONFIG, Config
from .evolution.pool import resolve_workers
from .features import available_specs
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
from .telemetry.history import lineage_summaries
from .telemetry.reporter import ConsoleReporter
from .trainer import Trainer
from .web.server import DashboardServer


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


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    sections: dict = {}
    # Every knob that shapes the search, so a sweep needs no config edits.
    evolution_flags = {
        "population_size": args.population,
        "evaluation_episodes": args.episodes,
        "tournament_size": args.tournament,
        "elite_count": args.elites,
        "immigrant_count": args.immigrants,
        "mutation_rate": args.mutation_rate,
        "mutation_strength": args.mutation_strength,
        "min_mutation_rate": args.min_mutation_rate,
        "max_mutation_rate": args.max_mutation_rate,
        "crossover_rate": args.crossover_rate,
        "workers": args.workers,
    }
    changes = {k: v for k, v in evolution_flags.items() if v is not None}
    if args.no_speciation:
        changes["use_speciation"] = False
    if changes:
        sections["evolution"] = dataclasses.replace(config.evolution, **changes)

    if args.board is not None:
        sections["board"] = dataclasses.replace(
            config.board, width=args.board, height=args.board
        )

    if args.spec is not None:
        sections["features"] = dataclasses.replace(config.features, spec=args.spec)

    if args.avoidable_penalty is not None or args.death_penalty is not None:
        changes = {}
        if args.avoidable_penalty is not None:
            changes["avoidable_death_fraction"] = args.avoidable_penalty
        if args.death_penalty is not None:
            changes["death_fraction"] = args.death_penalty
        sections["fitness"] = dataclasses.replace(config.fitness, **changes)

    if args.port is not None or args.no_hold:
        changes = {}
        if args.port is not None:
            changes["port"] = args.port
        if args.no_hold:
            changes["hold_when_viewing"] = False
        sections["server"] = dataclasses.replace(config.server, **changes)

    return config.with_overrides(**sections) if sections else config


def _latest_checkpoint(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.glob("*/checkpoints/latest.npz"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def command_train(args: argparse.Namespace) -> int:
    config = _apply_overrides(CONFIG, args)

    resume_from: Path | None = None
    if args.resume:
        resume_from = (
            Path(args.resume) if args.resume != "auto"
            else _latest_checkpoint(config.paths.runs_dir)
        )
        if resume_from is None or not resume_from.exists():
            print(f"  no checkpoint to resume from ({args.resume}); starting a new run")
            resume_from = None

    reporter = ConsoleReporter(every=args.report_every)
    try:
        trainer = Trainer(
            config,
            seed=args.seed,
            resume_from=resume_from,
            reporter=reporter,
            max_generations=args.generations,
        )
    except checkpoint.CheckpointMismatch as error:
        print(f"  cannot resume: {error}")
        return 1

    server = DashboardServer(trainer, config.server.host, config.server.port)
    served = server.start() if not args.headless else False

    reporter.banner(
        run_id=trainer.run_id,
        generation=trainer.population.generation,
        population=len(trainer.population.members),
        spec=trainer.population.feature_spec.name,
        inputs=trainer.population.feature_spec.size,
        genome=trainer.population.architecture.genome_size,
        url=server.url if served else "disabled",
        workers=trainer.workers,
    )
    if served and args.open:
        webbrowser.open(server.url)

    trainer.start()
    try:
        trainer.join()
    except KeyboardInterrupt:
        print()
        reporter.note("interrupted, finishing the current generation")
        trainer.stop()

    info = trainer.save_latest()
    server.stop()
    reporter.farewell(
        generation=trainer.population.generation,
        episodes=trainer.population.total_episodes,
        best=trainer.population.best_ever_score,
        checkpoint=str(info.path),
    )
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
    print(f"  best score    {info.best_score}")
    print(f"  best fitness  {info.best_fitness:.2f}")
    print(f"  encoding      {info.feature_spec}")
    print(f"  architecture  {meta.get('architecture')}")
    print(f"  episodes      {meta.get('total_episodes', 0):,}")

    manifest_path = path.parent.parent / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"  seed          {manifest.get('seed')}")
        evolution = manifest.get("config", {}).get("evolution", {})
        tuned = {k: v for k, v in evolution.items()
                 if v != getattr(CONFIG.evolution, k, object())}
        print(f"  evolution     {tuned or 'defaults'}")

    fitness = meta.get("fitness") or {}
    tuned = {k: v for k, v in fitness.items()
             if v != getattr(CONFIG.fitness, k, object())}
    if fitness:
        print(f"  fitness       {tuned or 'defaults'}")
    else:
        print("  fitness       not recorded (checkpoint predates config capture)")
    return 0


def command_export(args: argparse.Namespace) -> int:
    path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(CONFIG.paths.runs_dir)
    if path is None or not path.exists():
        print("  no checkpoint found")
        return 1

    import numpy as np

    # The rules the run was trained under, which the model file records.
    context = run_context(path)
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
    from .features import FeatureSpec
    from .nn.network import Architecture

    architecture = Architecture.from_dict(population_meta["architecture"])
    spec = FeatureSpec(context.spec_name, context.features)
    destination = config.paths.exports_dir / f"{args.name}.json"
    run_dir = path.parent.parent

    report = export_model.build_report(
        generation=int(population_meta.get("generation", 0)),
        genome=genome,
        summaries=lineage_summaries(run_dir),
        board=context.board,
        episode=context.episode,
        features=context.features,
        architecture=architecture,
        feature_spec=spec,
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


def command_specs(_: argparse.Namespace) -> int:
    for name, size in available_specs().items():
        print(f"  {name:<20} {size} inputs")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolvarium", description="Neuroevolution of Snake agents"
    )
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="run evolution with the live dashboard")
    train.add_argument("--generations", type=int, default=None, help="stop after this many")
    train.add_argument("--population", type=int, default=None)
    train.add_argument("--episodes", type=int, default=None, help="evaluation episodes per genome")
    train.add_argument("--board", type=int, default=None, help="square board size in cells")
    train.add_argument("--spec", default=None, choices=sorted(available_specs()),
                       help="state encoding to train under")
    train.add_argument("--tournament", type=int, default=None,
                       help="tournament size; higher is stronger selection pressure")
    train.add_argument("--elites", type=int, default=None,
                       help="genomes carried forward unchanged each generation")
    train.add_argument("--immigrants", type=int, default=None,
                       help="fresh random genomes injected each generation")
    train.add_argument("--mutation-rate", type=float, default=None,
                       help="share of genes perturbed in an offspring")
    train.add_argument("--mutation-strength", type=float, default=None,
                       help="size of each perturbation")
    train.add_argument("--min-mutation-rate", type=float, default=None,
                       help="floor the adaptive mutation schedule can fall to")
    train.add_argument("--max-mutation-rate", type=float, default=None,
                       help="ceiling the schedule escalates to while stagnating")
    train.add_argument("--crossover-rate", type=float, default=None,
                       help="share of offspring built from two parents")
    train.add_argument("--avoidable-penalty", type=float, default=None,
                       help="cost of dying when a safe move existed, as a fraction "
                            "of food earned")
    train.add_argument("--death-penalty", type=float, default=None,
                       help="cost of dying at all, as a fraction of food earned; "
                            "running out the clock is exempt")
    train.add_argument("--workers", type=int, default=None,
                       help="evaluation worker processes; 0 uses every core but one, 1 is serial")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--port", type=int, default=None)
    train.add_argument("--resume", nargs="?", const="auto", default=None,
                       help="resume from a checkpoint path, or 'auto' for the newest")
    train.add_argument("--report-every", type=int, default=1)
    train.add_argument("--headless", action="store_true", help="do not start the dashboard")
    train.add_argument("--no-hold", action="store_true",
                       help="never pause for the dashboard between generations")
    train.add_argument("--no-speciation", action="store_true")
    train.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    train.set_defaults(func=command_train)

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

    specs = sub.add_parser("specs", help="list feature encodings")
    specs.set_defaults(func=command_specs)

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
