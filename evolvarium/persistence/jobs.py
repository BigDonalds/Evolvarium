"""The training a dashboard has taken on, written beside each run.

`serve` writes `job.json` into a run's directory when it starts the run and
again when the run ends. A run whose job still says "training" when nothing is
training it was cut off by a restart or a killed process, and the next `serve`
on the same runs directory carries it on.

    generations  the generation the run stops at, or null to train until stopped
    state        "training", "stopped", "finished" or "failed"
    restarts     when the run was carried on, and from which generation
    error        why a run failed, when it did

`job.lock` is held with the operating system's own file locking for as long as
the run trains in a process. The lock goes with the process however it ends, so
a run whose lock can be taken is one nothing is training, and two dashboards on
one runs directory never carry the same run on.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .checkpoint import replace_with_retry

JOB_FILE = "job.json"
LOCK_FILE = "job.lock"
STATES = ("training", "stopped", "finished", "failed")


def read(run_dir: Path) -> dict | None:
    """The run's job, or None for a run no dashboard started or one unreadable."""
    try:
        data = json.loads((run_dir / JOB_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("state") not in STATES:
        return None
    return data


def write(run_dir: Path, job: dict) -> None:
    path = run_dir / JOB_FILE
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(job, indent=2), encoding="utf-8")
    # The page reads the job as it lists the runs.
    replace_with_retry(staging, path)


def begin(run_dir: Path, generations: int | None) -> dict:
    job = {"generations": generations, "state": "training", "restarts": []}
    write(run_dir, job)
    return job


def update(run_dir: Path, **fields) -> dict | None:
    """Change some of the job's fields; a run with no job is left without one."""
    job = read(run_dir)
    if job is None:
        return None
    job.update(fields)
    write(run_dir, job)
    return job


def restarted(run_dir: Path, generation: int) -> dict | None:
    job = read(run_dir)
    if job is None:
        return None
    job["state"] = "training"
    job.pop("error", None)
    job.setdefault("restarts", []).append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "generation": generation,
    })
    write(run_dir, job)
    return job


def unfinished(runs_dir: Path) -> list[Path]:
    """Every run whose job is still training, oldest first."""
    found = []
    for path in runs_dir.glob(f"*/{JOB_FILE}"):
        job = read(path.parent)
        if job is None or job["state"] != "training":
            continue
        try:
            started = (path.parent / "run.json").stat().st_mtime
        except OSError:
            continue
        found.append((started, path.parent.name, path.parent))
    return [run_dir for _, _, run_dir in sorted(found)]


class JobLock:
    """The lock a process holds on a run for as long as it trains it."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / LOCK_FILE
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock without waiting, and say whether it was free."""
        if self._handle is not None:
            return True
        # Kept open for as long as the lock is held.
        handle = open(self.path, "a+b")
        try:
            _lock(handle)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            _unlock(handle)
        except OSError:
            pass
        finally:
            handle.close()


if os.name == "nt":
    import msvcrt

    # Windows locks a range of bytes through one handle. The range is the first
    # byte, which may lie past the end of the empty file.
    def _lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
