"""Training events.

The trainer announces what happens as a run proceeds and does not know who is
listening. History, checkpoints, console output and the dashboard each subscribe
to the events they need, so another view is another subscriber rather than one
more call wired into the training loop.

Subscribers come in two kinds. Record keepers (history and checkpoints) are what
a run leaves behind. They hear every event first, so a view is only ever told
about something that has already been written down, and if one of them fails
the run stops rather than carrying on without a record. Views hear it after
them. A view that raises is dropped and the failure announced as a note, and
training carries on: a broken chart is not a reason to lose a run.

Handlers run on the thread that publishes, which for everything but a couple of
notes is the trainer thread between generations. They must not draw from the
population's random generator. A run with nobody listening evolves exactly as
one with every subscriber attached.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

    from .evolution.population import GenerationReport


class Event:
    kind: ClassVar[str] = "event"

    def to_dict(self) -> dict:
        """JSON-ready form, for the dashboard's event stream."""
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return {"kind": self.kind, **values}


@dataclass(frozen=True)
class RunStarted(Event):
    kind: ClassVar[str] = "run_started"
    run_id: str
    generation: int
    resumed_from: str | None


@dataclass(frozen=True)
class GenerationStarted(Event):
    kind: ClassVar[str] = "generation_started"
    generation: int


@dataclass(frozen=True)
class GenerationEvaluated(Event):
    """Every candidate has played and the next generation has been bred.

    `report` still holds the members that played, replays included. The best
    genome is the run's best so far, which is not always this generation's.
    """

    kind: ClassVar[str] = "generation"
    report: GenerationReport
    best_genome: np.ndarray | None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "generation": self.report.generation,
            "summary": self.report.to_dict(),
        }


@dataclass(frozen=True)
class RecordSet(Event):
    kind: ClassVar[str] = "record"
    generation: int
    score: float
    fitness: float


@dataclass(frozen=True)
class Held(Event):
    """The trainer is waiting at a boundary for the dashboard to let it go on."""

    kind: ClassVar[str] = "held"
    generation: int


@dataclass(frozen=True)
class Resumed(Event):
    kind: ClassVar[str] = "resumed"
    generation: int
    reason: str


@dataclass(frozen=True)
class CheckpointSaved(Event):
    kind: ClassVar[str] = "checkpoint"
    path: str
    generation: int
    trigger: str  # "periodic", "record" or "final"


@dataclass(frozen=True)
class Note(Event):
    kind: ClassVar[str] = "note"
    message: str


@dataclass(frozen=True)
class RunFinished(Event):
    """The loop has ended.

    `reason` is "limit", "stopped", "abandoned" or "error". Abandoned means the
    workers were shut down under a generation that had been told to stop. After
    that or an error the population may be part way through a generation, so
    nothing should be saved from it.
    """

    kind: ClassVar[str] = "run_finished"
    generation: int
    episodes: int
    # None when the run ended before its first generation was scored.
    best_score: float | None
    reason: str
    error: str | None = None


Handler = Callable[[Event], None]


@dataclass(eq=False)
class _Subscription:
    handler: Handler
    kinds: tuple[type[Event], ...]

    def wants(self, event: Event) -> bool:
        return not self.kinds or isinstance(event, self.kinds)


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keepers: list[_Subscription] = []
        self._views: list[_Subscription] = []

    def subscribe(self, handler: Handler, *kinds: type[Event],
                  keeps_record: bool = False) -> Callable[[], None]:
        """Call `handler` with each event of the given kinds, or with every event.

        Returns a function that ends the subscription.
        """
        subscription = _Subscription(handler, kinds)
        with self._lock:
            (self._keepers if keeps_record else self._views).append(subscription)
        return lambda: self._remove(subscription)

    def _remove(self, subscription: _Subscription) -> None:
        with self._lock:
            for group in (self._keepers, self._views):
                if subscription in group:
                    group.remove(subscription)

    def publish(self, event: Event) -> None:
        with self._lock:
            keepers = list(self._keepers)
            views = list(self._views)

        for subscription in keepers:
            if subscription.wants(event):
                subscription.handler(event)

        dropped = []
        for subscription in views:
            if not subscription.wants(event):
                continue
            try:
                subscription.handler(event)
            except Exception as error:
                self._remove(subscription)
                dropped.append((subscription.handler, error))

        # Announced after the loop, and by then the failed view is gone, so a
        # view that also fails on notes cannot feed itself.
        for handler, error in dropped:
            name = getattr(handler, "__qualname__", repr(handler))
            self.publish(Note(
                f"{name} stopped receiving events after {type(error).__name__}: {error}"
            ))
