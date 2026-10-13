"""Console output.

Every figure printed here comes from the generation that just finished being
evaluated, never from the freshly bred population that has not played yet.
Reporting after the population is replaced is what produces the familiar
"scoring individuals: 5/100" line that only ever counts the elites.
"""

from __future__ import annotations

import sys
import time

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
    def __init__(self, *, colour: bool | None = None, every: int = 1):
        self.colour = _supports_colour() if colour is None else colour
        self.every = max(1, every)
        self.started = time.time()
        self.rule = _rule_character()

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.colour else text

    def banner(self, *, run_id: str, generation: int, population: int, spec: str,
               inputs: int, genome: int, url: str, workers: int = 1) -> None:
        line = self.rule * 62
        print(self._paint(line, _DIM))
        print(self._paint("  evolvarium  |  neuroevolution", _BOLD))
        print(self._paint(line, _DIM))
        print(f"  run           {run_id}")
        print(f"  generation    {generation}")
        print(f"  population    {population}")
        print(f"  encoding      {spec} ({inputs} inputs)")
        print(f"  genome        {genome} parameters")
        print(f"  workers       {workers}")
        print(f"  dashboard     {self._paint(url, _CYAN)}")
        print(self._paint(line, _DIM))
        print()

    def generation(self, report: GenerationReport) -> None:
        if report.generation % self.every and not report.record:
            return

        deaths = report.deaths
        top_death = max(deaths, key=deaths.get) if deaths else "-"
        marker = self._paint(" record", _GREEN) if report.record else ""
        stagnation = (
            self._paint(f"stag {report.stagnation:>3}", _YELLOW)
            if report.stagnation > 10
            else f"stag {report.stagnation:>3}"
        )

        print(
            f"  gen {report.generation:>5}"
            f"  best {report.max_episode_score:>3}"
            f"  mean {report.mean_score:>6.3f}"
            f"  eat% {report.scoring_fraction * 100:>5.1f}"
            f"  fit {report.best_fitness:>9.1f}"
            f"  steps {report.mean_steps:>6.1f}"
            f"  div {report.diversity:>5.3f}"
            f"  mut {report.mutation_rate:>5.3f}"
            f"  {stagnation}"
            f"  {top_death:<11}"
            f"{marker}"
        )

    def held(self, generation: int) -> None:
        print(self._paint(f"  held at generation {generation} — dashboard is viewing", _YELLOW))

    def resumed(self, generation: int, reason: str) -> None:
        print(self._paint(f"  resumed from generation {generation} ({reason})", _DIM))

    def note(self, message: str) -> None:
        print(f"  {message}")

    def farewell(self, *, generation: int, episodes: int, best: int, checkpoint: str) -> None:
        elapsed = time.time() - self.started
        hours, rest = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rest, 60)
        line = self.rule * 62
        print()
        print(self._paint(line, _DIM))
        print(f"  generations   {generation}")
        print(f"  episodes      {episodes:,}")
        print(f"  best score    {best}")
        print(f"  runtime       {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  checkpoint    {checkpoint}")
        print(self._paint(line, _DIM))
