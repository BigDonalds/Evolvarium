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

A worker that dies or goes quiet is replaced, and the genome it was scoring is
played again on another. A task is a pure function of what it is given, so the
second go returns exactly what the first would have. Losing a second worker on
the same genome ends the job, as does a game raising or running over its time:
neither takes the process running the pool down with it.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable
from multiprocessing import connection
from typing import Any

from .evaluation import EvalContext, GameFault, beat_interval, init_worker, run_task

# How long a busy worker may stay quiet beyond a step's limit and the interval
# between its reports, for the time a loaded machine takes to schedule it.
SLACK = 2.0


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


class WorkerLost(RuntimeError):
    """Workers kept dying or going quiet, where one more would not help."""


class WorkerError(RuntimeError):
    """The engine's own code failed in a worker. The cause holds its traceback."""


class RemoteTraceback(Exception):
    def __str__(self) -> str:
        return str(self.args[0])


def _serve(conn, ctx: EvalContext) -> None:
    """A worker's whole life: start, then score tasks until told to stop."""
    try:
        init_worker(ctx, beat=lambda seed, step: conn.send(("beat", seed, step)))
    except Exception as error:
        conn.send(("failed", f"{type(error).__name__}: {error}"))
        return
    conn.send(("ready",))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        index, run, task = message
        try:
            reply = ("done", index, run(task))
        except GameFault as fault:
            reply = ("fault", index, fault)
        except Exception as error:
            reply = ("error", index, f"{type(error).__name__}: {error}", traceback.format_exc())
        try:
            conn.send(reply)
        except OSError:
            return
        except Exception as error:
            # Pickling comes before anything is written, so the pipe is still clean.
            conn.send(("error", index, f"its result could not be sent back: "
                       f"{type(error).__name__}: {error}", traceback.format_exc()))


class _Worker:
    """One worker process, and what the pool last heard from it."""

    def __init__(self, process, conn, starts: int):
        self.process = process
        self.conn = conn
        # Times in a row this place in the pool has been filled without the
        # worker in it finishing a task.
        self.starts = starts
        self.ready = False
        self.gone = False
        self.task: int | None = None
        self.heard = time.perf_counter()
        # The episode and step of its last report on the task it has.
        self.seed: int | None = None
        self.step = 0

    def whereabouts(self) -> str:
        if self.seed is None:
            return "before it reported a step"
        return f"last heard from at step {self.step} of the episode with seed {self.seed}"


class EvaluationPool:
    """Worker processes that score genomes, or a stand-in when serial.

    `own_process` puts even a single worker in a process of its own. Several
    trainers in one interpreter would otherwise queue on its lock, however many
    cores are free.
    """

    def __init__(self, ctx: EvalContext, workers: int, *, own_process: bool = False):
        self.workers = workers
        self._ctx = ctx
        self._processes = workers > 1 or own_process
        self._spawn = multiprocessing.get_context()
        limits = ctx.limits
        self._step_seconds = limits.step_seconds
        self._stalled = limits.step_seconds + beat_interval(limits) + SLACK
        self._lock = threading.Lock()
        self._closed = False
        self._mapping = False
        self._notes: list[str] = []
        self._crew: list[_Worker] = []
        if self._processes:
            self._crew = [self._start() for _ in range(workers)]

    @property
    def active(self) -> bool:
        return self._processes and not self._closed

    def take_notes(self) -> list[str]:
        """The problems the pool got past since the last call, such as a lost worker."""
        notes, self._notes = self._notes, []
        return notes

    def map(
        self, tasks: Iterable[tuple], run: Callable[[tuple], Any] = run_task,
        played: Callable[[], None] | None = None,
    ) -> list[Any]:
        """Score every task, returning results in the order given.

        `run` is called on each task inside a worker. It has to be a module-level
        function so the worker can import it, and anything other than `run_task`
        is there to wrap it: the profiler times tasks this way. `played`, if
        given, is called once for each result as it arrives.

        Order matters: the parent applies results back onto its own list of
        members positionally, so the mapping must not depend on which worker
        finished first. `map` guarantees that whatever order they finish in.

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

        A game that raises or runs over its time raises the same `GameFault` it
        would in this process. Anything that ends `map` early closes the pool.
        """
        tasks = list(tasks)
        if not self._processes:
            raise RuntimeError("pool is not active; the caller should run serially")
        with self._lock:
            if self._closed:
                raise RuntimeError("the pool is closed")
            self._mapping = True
        try:
            return self._map(tasks, run, played)
        except BaseException:
            with self._lock:
                self._closed = True
            raise
        finally:
            with self._lock:
                self._mapping = False
                closed = self._closed
            if closed:
                self._kill()

    def _map(self, tasks: list[tuple], run: Callable, played: Callable | None) -> list[Any]:
        clock = time.perf_counter
        results: list[Any] = [None] * len(tasks)
        waiting = deque(range(len(tasks)))
        # How each task lost so far was lost.
        lost: dict[int, str] = {}
        left = len(tasks)
        while left:
            for worker in self._crew:
                if worker.ready and not worker.gone and worker.task is None and waiting:
                    index = waiting.popleft()
                    worker.task, worker.seed, worker.step = index, None, 0
                    worker.heard = clock()
                    try:
                        worker.conn.send((index, run, tasks[index]))
                    except OSError:
                        worker.gone = True

            before = clock()
            quiet = [worker.heard + self._stalled - before
                     for worker in self._crew if worker.task is not None]
            timeout = min(max(min(quiet, default=1.0), 0.0), 1.0)
            connection.wait([w.conn for w in self._crew] + [w.process.sentinel for w in self._crew],
                 timeout)
            now = clock()
            if self._closed:
                raise RuntimeError("the pool was closed while it was scoring")
            if now - before - timeout > SLACK:
                # This process was held up well past the wait it asked for, as
                # when the machine sleeps, so the silence since is not theirs.
                for worker in self._crew:
                    worker.heard = now

            for worker in list(self._crew):
                left -= self._read(worker, results, played, now)
            for worker in list(self._crew):
                if worker.gone or not worker.process.is_alive():
                    self._replace(worker, waiting, lost, None)
                elif worker.task is not None and now - worker.heard > self._stalled:
                    self._replace(worker, waiting, lost, now - worker.heard)
        return results

    def _read(self, worker: _Worker, results: list, played: Callable | None,
              now: float) -> int:
        """Take every message `worker` has sent, returning how many tasks it finished."""
        finished = 0
        while not worker.gone:
            try:
                if not worker.conn.poll():
                    break
                message = worker.conn.recv()
            except (EOFError, OSError):
                worker.gone = True
                break
            kind = message[0]
            worker.heard = now
            if kind == "beat":
                worker.seed, worker.step = message[1], message[2]
            elif kind == "done":
                results[message[1]] = message[2]
                worker.task = None
                worker.starts = 0
                finished += 1
                if played is not None:
                    played()
            elif kind == "ready":
                worker.ready = True
            elif kind == "fault":
                raise message[2]
            elif kind == "error":
                raise WorkerError(f"scoring failed in a worker: {message[2]}") from (
                    RemoteTraceback(message[3]))
            else:
                raise WorkerError(f"a worker could not start: {message[1]}")
        return finished

    def _replace(self, worker: _Worker, waiting: deque, lost: dict[int, str],
                 silent: float | None) -> None:
        """Start a new worker in place of one that died, or went quiet and is ended here."""
        self._crew.remove(worker)
        process = worker.process
        if silent is not None:
            process.kill()
        else:
            # A worker whose pipe has closed is on its way out.
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
        process.join(timeout=5.0)
        worker.conn.close()

        if silent is None:
            what = f"a worker stopped with exit code {process.exitcode}"
        else:
            what = (f"a worker went {silent:.1f} s without a word, longer than "
                    f"limits.step_seconds allows ({self._step_seconds:g} s)")
        index = worker.task
        if index is None:
            if worker.starts >= 2:
                when = "between genomes" if worker.ready else "while starting"
                raise WorkerLost(f"{what} {when}, for the second time in a row")
            starts = worker.starts + 1
        else:
            description = f"{what} while scoring a genome, {worker.whereabouts()}"
            if index in lost:
                raise WorkerLost(f"{description}. It was the second worker lost on that "
                                 f"genome; the first time, {lost[index]}")
            lost[index] = description
            waiting.appendleft(index)
            self._notes.append(f"{description}, so the genome was played again")
            starts = 1
        self._crew.append(self._start(starts))

    def _start(self, starts: int = 1) -> _Worker:
        ours, theirs = self._spawn.Pipe()
        process = self._spawn.Process(target=_serve, args=(theirs, self._ctx), daemon=True)
        process.start()
        # The worker has its own copy now. Closing this one lets the pool see
        # the pipe end when the worker does.
        theirs.close()
        return _Worker(process, ours, starts)

    def _kill(self) -> None:
        crew, self._crew = self._crew, []
        for worker in crew:
            if worker.process.is_alive():
                worker.process.kill()
        for worker in crew:
            worker.process.join(timeout=5.0)
            worker.conn.close()

    def close(self, wait: bool = False) -> None:
        """Shut the workers down, waiting for them to exit when `wait` is set.

        Closed while another thread is in `map`, the workers are ended at once
        and that `map` raises.
        """
        with self._lock:
            self._closed = True
            mapping = self._mapping
        if mapping:
            for worker in list(self._crew):
                worker.process.kill()
            return
        crew, self._crew = self._crew, []
        for worker in crew:
            if not worker.ready:
                worker.process.kill()
                continue
            try:
                worker.conn.send(None)
            except OSError:
                pass
        for worker in crew:
            if wait:
                worker.process.join(timeout=self._stalled)
                if worker.process.is_alive():
                    worker.process.kill()
            worker.conn.close()

    def __enter__(self) -> EvaluationPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
