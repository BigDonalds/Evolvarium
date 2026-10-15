"""Training events for the dashboard, as a server-sent event stream.

The trainer publishes on its own thread and must never wait for a browser, so
each event is formatted once into a bounded log and every open page reads the
log at its own pace. A page that reconnects asks to continue after the last id
it saw. When the log no longer reaches back that far, or the id came from an
earlier server process, the page is told to reload its state from the regular
endpoints instead of carrying on with a gap it cannot see.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from ..events import Event


class EventLog:
    def __init__(self, capacity: int = 1024):
        # Ids carry the server's start time, so a page left open across a
        # restart cannot take the new process's event 40 for the one it saw.
        self.epoch = format(time.time_ns(), "x")
        self._messages: deque[tuple[int, str]] = deque(maxlen=capacity)
        self._issued = 0
        self._changed = threading.Condition()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._changed:
            return self._closed

    def add(self, event: Event) -> None:
        data = json.dumps(event.to_dict(), separators=(",", ":"), default=float)
        with self._changed:
            self._issued += 1
            message = f"id: {self.epoch}-{self._issued}\nevent: {event.kind}\ndata: {data}\n\n"
            self._messages.append((self._issued, message))
            self._changed.notify_all()

    def close(self) -> None:
        """Wake every waiting reader so open streams end."""
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def cursor_for(self, last_event_id: str | None) -> tuple[int, bool]:
        """Where a page continues from, and whether it has missed events.

        A page without an id is starting fresh. It loads everything else from
        the regular endpoints, so it starts at the present.
        """
        with self._changed:
            if not last_event_id:
                return self._issued, False
            epoch, _, number = last_event_id.partition("-")
            try:
                seen = int(number)
            except ValueError:
                return self._issued, True
            if epoch != self.epoch or seen > self._issued or seen < self._oldest() - 1:
                return self._issued, True
            return seen, False

    def read(self, cursor: int, timeout: float) -> tuple[int, list[str], bool]:
        """Messages after `cursor`, waiting up to `timeout` for the first.

        Returns the new cursor, the messages and whether some were lost to the
        log's capacity, in which case the cursor jumps to the present.
        """
        with self._changed:
            if not self._closed and self._issued <= cursor:
                self._changed.wait(timeout)
            if cursor < self._oldest() - 1:
                return self._issued, [], True
            messages = [text for number, text in self._messages if number > cursor]
            return self._issued, messages, False

    def _oldest(self) -> int:
        return self._messages[0][0] if self._messages else self._issued + 1
