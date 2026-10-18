"""Worker pool for genome evaluation.

The pool is created once and reused for the whole run. On Windows each worker is
a fresh interpreter that re-imports numpy and this package, which costs a second
or two; paying that per generation would cancel most of the gain.

`resolve_workers` deliberately leaves a core free. The dashboard's HTTP thread
and the trainer itself still need to run, and measured speedup from six workers
to eight was 4.89x against 5.33x, so the last core buys very little and costs
responsiveness.
"""

from __future__ import annotations

import os
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


class EvaluationPool:
    """Process pool that scores genomes, or a no-op wrapper when serial."""

    def __init__(self, ctx: EvalContext, workers: int):
        self.workers = workers
        self._ctx = ctx
        self._executor = None
        if workers > 1:
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

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> EvaluationPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
