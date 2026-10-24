"""Worker pool for genome evaluation.

The pool is created once and reused for the whole run. On Windows each worker is
a fresh interpreter that re-imports numpy and this package, which costs a second
or two; paying that per generation would cancel most of the gain.

`resolve_workers` deliberately leaves a core free. The dashboard's HTTP thread
and the trainer itself still need to run, and measured speedup from six workers
to eight was 4.89x against 5.33x, so the last core buys very little and costs
responsiveness.

`WorkerBudget` divides those cores between the runs one dashboard trains at
once. Results never depend on how many workers play them, so a run can be given
more or fewer between generations without changing what it evolves.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from typing import Any

from .evaluation import EvalContext, init_worker, run_task


def resolve_workers(requested: int) -> int:
    """Turn a configured worker count into an actual one.

    0 means "decide for me": every core but one.
    """
    if requested > 0:
        return requested
    return max(1, (os.cpu_count() or 2) - 1)


class WorkerBudget:
    """A fixed number of workers shared fairly between the jobs using them.

    A job that asks for fewer than an even split gets what it asks for, and the
    rest is split evenly between the others, the earliest to join taking any
    worker left over. Every job gets at least one, so a budget with more jobs
    than workers runs some of them on the same core.
    """

    def __init__(self, total: int):
        self.total = max(1, total)
        self._wanted: dict[str, int] = {}
        self._lock = threading.Lock()

    def join(self, key: str, wanted: int = 0) -> int:
        """Add a job wanting `wanted` workers, 0 for as many as it can get, and return its share."""
        with self._lock:
            self._wanted.setdefault(key, max(0, wanted))
            return self._shares()[key]

    def leave(self, key: str) -> None:
        with self._lock:
            self._wanted.pop(key, None)

    def share(self, key: str) -> int:
        """The workers `key` may use now, which changes as other jobs join and leave."""
        with self._lock:
            return self._shares().get(key, 1)

    @property
    def jobs(self) -> int:
        with self._lock:
            return len(self._wanted)

    def _shares(self) -> dict[str, int]:
        shares: dict[str, int] = {}
        left = self.total
        open_ = list(self._wanted)
        while open_:
            fair = left // len(open_)
            capped = [key for key in open_ if 0 < self._wanted[key] <= fair]
            if not capped:
                break
            for key in capped:
                shares[key] = self._wanted[key]
                left -= self._wanted[key]
            open_ = [key for key in open_ if key not in shares]
        if open_:
            base, extra = divmod(left, len(open_))
            for place, key in enumerate(open_):
                shares[key] = max(1, base + (1 if place < extra else 0))
        return shares


class EvaluationPool:
    """Process pool that scores genomes, or a no-op wrapper when serial.

    `own_process` puts even a single worker in a process of its own. Several
    trainers in one interpreter would otherwise queue on its lock, however many
    cores are free.
    """

    def __init__(self, ctx: EvalContext, workers: int, *, own_process: bool = False):
        self.workers = workers
        self._ctx = ctx
        self._executor = None
        if workers > 1 or own_process:
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor(
                max_workers=workers, initializer=init_worker, initargs=(ctx,)
            )

    @property
    def active(self) -> bool:
        return self._executor is not None

    def map(
        self, tasks: Iterable[tuple], run: Callable[[tuple], Any] = run_task,
        played: Callable[[], None] | None = None,
    ) -> list[Any]:
        """Score every task, returning results in the order given.

        `run` is called on each task inside a worker. It has to be a module-level
        function so the worker can import it, and anything other than `run_task`
        is there to wrap it: the profiler times tasks this way. `played`, if
        given, is called as each result arrives, in order.

        Order matters: the parent applies results back onto its own list of
        members positionally, so the mapping must not depend on which worker
        finished first. `map` guarantees that regardless of chunk size.

        One task per dispatch is deliberate. Episode length spans two orders of
        magnitude within a single generation -- measured at 7 steps for the
        weakest genome against 1,276 for the strongest -- so fixed blocks leave a
        worker grinding through several long episodes while the rest idle. A
        genome is a couple of kilobytes, so the extra dispatches cost far less
        than the idle time they avoid.

        Measured against chunks of five, interleaved over five rounds to cancel
        drift: faster in four rounds, mean difference 1.5s per generation. The
        machine drifted 4.7x slower across that benchmark on identical work, so
        treat the size of the win as unresolved; the direction matches theory and
        the downside is bounded either way.
        """
        tasks = list(tasks)
        if self._executor is None:
            raise RuntimeError("pool is not active; the caller should run serially")
        results = self._executor.map(run, tasks, chunksize=1)
        if played is None:
            return list(results)
        collected = []
        for result in results:
            collected.append(result)
            played()
        return collected

    def close(self, wait: bool = False) -> None:
        """Shut the workers down, waiting for them to exit when `wait` is set."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> EvaluationPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
