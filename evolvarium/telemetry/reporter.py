"""Console output.

Every figure printed here comes from the generation that just finished being
evaluated, never from the freshly bred population that has not played yet.
Reporting after the population is replaced is what produces the familiar
"scoring individuals: 5/100" line that only ever counts the elites.

When several runs train in one process, each run's reporter is given a tag that
starts every line it prints, so the runs can be told apart as their lines mix.
"""

from __future__ import annotations

import sys
import time

from ..events import (
    CheckpointSaved,
    Event,
    EventBus,
    GenerationEvaluated,
    Held,
    Note,
    Resumed,
    RunFinished,
)
from ..evolution.population import GenerationReport

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"


def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _rule_character() -> str:
    """Pick a divider the active console encoding can actually render.

    Windows consoles still default to cp1252, which has no box-drawing glyphs
    and raises rather than substituting.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "─".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "-"
    return "─"


class ConsoleReporter:
    def __init__(self, *, colour: bool | None = None, every: int = 1, tag: str | None = None):
        self.colour = _supports_colour() if colour is None else colour
        self.every = max(1, every)
        self.tag = tag
        self.lead = "  " if tag is None else f"  [{tag}] "
        self.started = time.time()
        self.rule = _rule_character()
        self._last_checkpoint: CheckpointSaved | None = None

    def attach(self, events: EventBus) -> None:
        events.subscribe(
            self.handle, GenerationEvaluated, Held, Resumed, Note, CheckpointSaved, RunFinished
        )

    def handle(self, event: Event) -> None:
        match event:
            case GenerationEvaluated(report=report):
                self.generation(report)
            case Held(generation=generation):
                self.held(generation)
            case Resumed(generation=generation, reason=reason):
                self.resumed(generation, reason)
            case Note(message=message):
                self.note(message)
            case CheckpointSaved():
                self._last_checkpoint = event
            case RunFinished():
                self.finished(event)

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.colour else text

    def banner(self, *, run_id: str, generation: int, population: int, game: str,
               encoding: str, inputs: int, genome: int, url: str, workers: int = 1) -> None:
        line = self.rule * 62
        print(self._paint(line, _DIM))
        print(self._paint("  evolvarium  |  neuroevolution", _BOLD))
        print(self._paint(line, _DIM))
        print(f"  run           {run_id}" + ("" if self.tag is None else f"  [{self.tag}]"))
        print(f"  generation    {generation}")
        print(f"  population    {population}")
        print(f"  game          {game}")
        print(f"  encoding      {encoding} ({inputs} inputs)")
        print(f"  genome        {genome} parameters")
        print(f"  workers       {workers}")
        print(f"  dashboard     {self._paint(url, _CYAN)}")
        print(self._paint(line, _DIM))
        print()

    def generation(self, report: GenerationReport) -> None:
        if report.generation % self.every and not report.record:
            return

        ends = report.ends
        top_end = max(ends, key=ends.get) if ends else "-"
        marker = self._paint(" record", _GREEN) if report.record else ""
        stagnation = (
            self._paint(f"stag {report.stagnation:>3}", _YELLOW)
            if report.stagnation > 10
            else f"stag {report.stagnation:>3}"
        )

        print(
            f"{self.lead}gen {report.generation:>5}"
            f"  best {report.max_episode_score:>3g}"
            f"  mean {report.mean_score:>6.3f}"
            f"  scored% {report.scoring_fraction * 100:>5.1f}"
            f"  fit {report.best_fitness:>9.1f}"
            f"  steps {report.mean_steps:>6.1f}"
            f"  div {report.diversity:>5.3f}"
            f"  mut {report.mutation_rate:>5.3f}"
            f"  {stagnation}"
            f"  {top_end:<11}"
            f"{marker}"
        )

    def held(self, generation: int) -> None:
        print(self._paint(
            f"{self.lead}held at generation {generation} — dashboard is viewing", _YELLOW
        ))

    def resumed(self, generation: int, reason: str) -> None:
        print(self._paint(f"{self.lead}resumed from generation {generation} ({reason})", _DIM))

    def note(self, message: str) -> None:
        print(f"{self.lead}{message}")

    def finished(self, event: RunFinished) -> None:
        if event.error:
            self.note(f"training stopped: {event.error}")
        elif event.reason == "abandoned":
            self.note("quit part way through a generation; that generation was not saved")
        saved = self._last_checkpoint
        if saved is None:
            where = "none written"
        elif saved.generation != event.generation:
            # After an error or a forced quit the newest checkpoint is older
            # than the generation the run reached.
            where = f"{saved.path} (generation {saved.generation})"
        else:
            where = saved.path
        self.farewell(generation=event.generation, episodes=event.episodes,
                      best=event.best_score, checkpoint=where)

    def farewell(self, *, generation: int, episodes: int, best: float | None,
                 checkpoint: str) -> None:
        elapsed = time.time() - self.started
        hours, rest = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rest, 60)
        line = self.rule * 62
        print()
        print(self._paint(line, _DIM))
        lead = self.lead
        print(f"{lead}generations   {generation}")
        print(f"{lead}episodes      {episodes:,}")
        print(f"{lead}best score    {'-' if best is None else best}")
        print(f"{lead}runtime       {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"{lead}checkpoint    {checkpoint}")
        print(self._paint(line, _DIM))
