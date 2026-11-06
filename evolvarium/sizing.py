"""Fitting a run to the time someone has for it.

`sizing.minutes` names the time. Before the run starts, untrained networks play a
sample of games on the workers the run will have, which gives the seconds one
game takes there. The run then gets the largest population and number of games
per genome that leave time for `sizing.generations` generations. A setting
someone asked for by name stays as they set it, and the size is fitted around it.

A saved run shows why the sample alone is not enough. Episodes get longer as
players get better, so a generation late in a Snake run takes 6 to 35 times what
an early one did. The sample cannot see how far a game will go, so `_GROWTH`
stands in for it, and the trainer ends the run before a generation that would
not fit. The time is kept whichever way the guess goes wrong: a game that slows
more than expected reaches fewer generations.

Sizing only decides how big the run is. The run's seed still decides everything
it evolves, so a sized run and one given the same size by hand are the same run.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .config import Config, EvolutionConfig
from .evolution.evaluation import GameFault
from .evolution.pool import EvaluationPool, WorkerLost
from .evolution.population import Population
from .fixes import Fixes

# The first episode seed of the sample, in a block of its own. Nothing the sample
# plays reaches the run, so these seeds only ever affect the timing.
SEED = 650_000

# The sizes a run can take, smallest first, each a population and its games per
# genome. At a fixed time, 30 with 2 games did worse than 50 with 2 on both the
# arena and Snake, and the defaults did worse than every size between.
SIZES = ((50, 2), (100, 2), (100, 3), (150, 4))

# Generations cost this many times what the sample's games suggest, on average
# over a run: 1.2 to 1.8 times on the arena, 2.2 to 3.3 on cart-pole and 2.5 to 8
# on Snake, more the smaller the population, since a small one plays more
# generations and gets further.
_GROWTH = 3.0

# Of the time, the share the sample may use, within these limits in seconds.
_SAMPLE_SHARE = 0.03
_SAMPLE_SECONDS = (1.0, 15.0)
# Enough games for the time per game to settle on a fast game.
_SAMPLE_GAMES = 450

# Of the time, the share a probe may use.
_PROBE_SHARE = 0.2


@dataclass(frozen=True)
class Plan:
    """What a run's size came to, and the measurements it came from."""

    minutes: float
    workers: int
    # Seconds of the workers' time a game takes, measured on a sample.
    seconds_per_game: float
    games_sampled: int
    mean_steps: float
    population: int
    episodes: int
    elites: int
    immigrants: int
    # Generations the time leaves room for, allowing for episodes getting longer.
    generations: int
    wanted: int
    # Spent before the first generation, on any probe and on the sample.
    setup_seconds: float
    # Settings left as someone asked for them.
    kept: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        return self.minutes * 60.0

    def apply(self, config: Config) -> Config:
        """`config` at this size."""
        evolution = dataclasses.replace(
            config.evolution,
            population_size=self.population,
            evaluation_episodes=self.episodes,
            elite_count=self.elites,
            immigrant_count=self.immigrants,
        )
        return dataclasses.replace(config, evolution=evolution)

    def to_dict(self) -> dict:
        return asdict(self)

    def notes(self) -> list[str]:
        """The plan in a few lines, for a run's log."""
        plural = "" if self.workers == 1 else "s"
        lines = [
            f"sizing: {self.minutes:g} minutes; {self.games_sampled} untrained games took "
            f"{self.seconds_per_game * 1000:.3g} ms each on {self.workers} worker{plural}",
        ]
        size = f"population {self.population} with {self.episodes} games each"
        if self.generations >= self.wanted:
            lines.append(f"sizing: {size}, the largest that leaves time for "
                         f"{self.wanted} generations")
        else:
            lines.append(f"sizing: {size} leaves time for about {self.generations} of the "
                         f"{self.wanted} generations wanted; more time would allow the rest")
        if self.kept:
            lines.append(f"sizing: left as asked: {', '.join(self.kept)}")
        return lines


def with_probe_limit(config: Config) -> Config:
    """`config` with a probe held to its share of the run's time."""
    if config.sizing.minutes == 0:
        return config
    share = config.sizing.minutes * 60.0 * _PROBE_SHARE
    return dataclasses.replace(
        config, probe=dataclasses.replace(config.probe, seconds=min(config.probe.seconds, share))
    )


def _sample(config: Config, fixes: Fixes | None, workers: int) -> tuple[float, int, float]:
    """Seconds per game, games played and their mean steps, on `workers` workers.

    The workers start in the first batch, which is left out of the time unless a
    game is so slow that it uses all of it.
    """
    allowed = min(max(config.sizing.minutes * 60.0 * _SAMPLE_SHARE, _SAMPLE_SECONDS[0]),
                  _SAMPLE_SECONDS[1])
    population = Population(config, seed=SEED, fixes=fixes)
    genomes = itertools.cycle(member.genome for member in population.members)
    tasks = [(genome, SEED + index, 1, True)
             for index, genome in zip(range(_SAMPLE_GAMES), genomes, strict=False)]
    clock = time.perf_counter
    batch = 4 * workers
    try:
        with EvaluationPool(population.context, workers, own_process=True) as pool:
            began = clock()
            first = pool.map(tasks[:workers])
            took = clock() - began
            if took >= allowed:
                return took / workers, workers, _mean_steps(first)
            played = []
            began = clock()
            start = workers
            while start < len(tasks) and clock() - began < allowed:
                played += pool.map(tasks[start:start + batch])
                start += batch
            took = clock() - began
    except WorkerLost as error:
        raise GameFault(f"sizing the run lost its workers: {error}") from error
    if not played:
        return took / workers, workers, _mean_steps(first)
    return took / len(played), len(played), _mean_steps(played)


def _mean_steps(results: list) -> float:
    return sum(result.steps for result in results) / len(results)


def _sizes(evolution: EvolutionConfig, named: set[str]) -> list[tuple[int, int]]:
    """The sizes open to a run, smallest first, around whatever was asked for by name."""
    keep_population = "evolution.population_size" in named
    keep_episodes = "evolution.evaluation_episodes" in named
    sizes = []
    for population, episodes in SIZES:
        size = (evolution.population_size if keep_population else population,
                evolution.evaluation_episodes if keep_episodes else episodes)
        if size in sizes:
            continue
        elites, immigrants = _carried(evolution, size[0], named)
        if elites + immigrants < size[0] and evolution.tournament_size <= size[0]:
            sizes.append(size)
    # Asked for by name, a size can sit below some that are not. Dropping those keeps
    # the list in order of cost, and the settings it starts from always train.
    return sorted(sizes, key=lambda pair: pair[0] * pair[1]) or [
        (evolution.population_size, evolution.evaluation_episodes)
    ]


def _carried(evolution: EvolutionConfig, population: int, named: set[str]) -> tuple[int, int]:
    """Elites and immigrants at `population`, in the same share of it as they had."""
    def scaled(name: str, count: int) -> int:
        if name in named or count == 0:
            return count
        return max(1, round(count * population / evolution.population_size))

    return (scaled("evolution.elite_count", evolution.elite_count),
            scaled("evolution.immigrant_count", evolution.immigrant_count))


def _generations(left: float, seconds_per_game: float, population: int, episodes: int) -> int:
    return int(left / (population * episodes * seconds_per_game * _GROWTH))


def _choose(sizes: list[tuple[int, int]], left: float, seconds_per_game: float,
            wanted: int) -> tuple[int, int]:
    """The largest of `sizes` that leaves time for `wanted` generations, or else the smallest."""
    fits = [size for size in sizes if _generations(left, seconds_per_game, *size) >= wanted]
    return fits[-1] if fits else sizes[0]


def plan(config: Config, *, fixes: Fixes | None = None, named: Iterable[str] = (),
         workers: int = 1, probe_seconds: float = 0.0) -> Plan:
    """Measure the game `config` names and work out how big a run fits its time.

    `named` holds the settings someone asked for by name. Raises `GameFault` if
    the game's code fails while the sample plays it.
    """
    began = time.perf_counter()
    seconds_per_game, games_sampled, mean_steps = _sample(config, fixes, workers)
    setup = probe_seconds + time.perf_counter() - began

    names = set(named)
    evolution = config.evolution
    wanted = config.sizing.generations
    left = max(config.sizing.minutes * 60.0 - setup, 0.0)
    population, episodes = _choose(_sizes(evolution, names), left, seconds_per_game, wanted)
    elites, immigrants = _carried(evolution, population, names)
    kept = tuple(name for name in ("evolution.population_size", "evolution.evaluation_episodes",
                                   "evolution.elite_count", "evolution.immigrant_count")
                 if name in names)
    return Plan(
        minutes=config.sizing.minutes,
        workers=workers,
        seconds_per_game=seconds_per_game,
        games_sampled=games_sampled,
        mean_steps=mean_steps,
        population=population,
        episodes=episodes,
        elites=elites,
        immigrants=immigrants,
        generations=_generations(left, seconds_per_game, population, episodes),
        wanted=wanted,
        setup_seconds=setup,
        kept=kept,
    )
