"""Coordination between the training loop and the dashboard.

The trainer runs far faster than anyone can read. Without backpressure the
dashboard would replace the generation being examined with a newer one before
the viewer had finished with it, and generations would scroll past unseen.

`TrainingGate` lets the trainer pause at a generation boundary, and only at a
boundary: an episode is never interrupted part-way, so no partial evaluation is
ever recorded.

A hold that depends on a browser has an obvious failure mode. If the page were
the only thing able to release it, closing the tab or losing the network would
strand the run forever. Viewer presence is therefore a lease that expires:
heartbeats keep it alive, and once one lapses the trainer resumes on its own.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class GateState:
    holding: bool
    hold_enabled: bool
    viewer_present: bool
    credits: int
    generation: int
    waiting_since: float | None

    def to_dict(self) -> dict:
        return {
            "holding": self.holding,
            "hold_enabled": self.hold_enabled,
            "viewer_present": self.viewer_present,
            "credits": self.credits,
            "generation": self.generation,
            "held_for": round(time.time() - self.waiting_since, 1) if self.waiting_since else 0.0,
        }


class TrainingGate:
    def __init__(self, *, viewer_timeout: float, hold_enabled: bool = True):
        self.viewer_timeout = viewer_timeout
        self._hold_enabled = hold_enabled
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)

        self._last_heartbeat = 0.0
        self._credits = 0
        self._holding = False
        self._waiting_since: float | None = None
        self._generation = 0
        self._stopped = False

    # ------------------------------------------------------------------ viewer

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.time()

    def _viewer_present_locked(self) -> bool:
        return (time.time() - self._last_heartbeat) < self.viewer_timeout

    def viewer_present(self) -> bool:
        with self._lock:
            return self._viewer_present_locked()

    # ------------------------------------------------------------------ controls

    def release(self, generations: int = 1) -> None:
        """Allow `generations` more generations to run before holding again."""
        with self._lock:
            self._credits += max(1, generations)
            self._wake.notify_all()

    def set_hold(self, enabled: bool) -> None:
        with self._lock:
            self._hold_enabled = enabled
            if not enabled:
                self._credits = 0
            self._wake.notify_all()

    def stop(self) -> None:
        """Release any waiting trainer so shutdown cannot block."""
        with self._lock:
            self._stopped = True
            self._wake.notify_all()

    def state(self) -> GateState:
        with self._lock:
            return GateState(
                holding=self._holding,
                hold_enabled=self._hold_enabled,
                viewer_present=self._viewer_present_locked(),
                credits=self._credits,
                generation=self._generation,
                waiting_since=self._waiting_since,
            )

    # ------------------------------------------------------------------ trainer

    def wait_at_boundary(self, generation: int, on_hold=None, on_resume=None) -> None:
        """Block between generations while a viewer is present and hold is on."""
        with self._lock:
            self._generation = generation
            if self._stopped or not self._hold_enabled:
                return
            if not self._viewer_present_locked():
                return
            if self._credits > 0:
                self._credits -= 1
                return

            self._holding = True
            self._waiting_since = time.time()

        if on_hold is not None:
            on_hold(generation)

        reason = "released"
        with self._lock:
            while not self._stopped:
                if not self._hold_enabled:
                    reason = "hold disabled"
                    break
                if self._credits > 0:
                    self._credits -= 1
                    reason = "continued"
                    break
                if not self._viewer_present_locked():
                    reason = "viewer left"
                    break
                # Re-check periodically so an expiring viewer lease is noticed
                # even when no control call arrives to wake this thread.
                self._wake.wait(timeout=0.5)

            self._holding = False
            self._waiting_since = None

        if on_resume is not None:
            on_resume(generation, reason)
