"""Dashboard HTTP server.

Standard library only, so the dashboard adds no dependencies to the project.

Every GET about a run takes `run=ID` and otherwise answers for the run training
here, or the newest run when none is.

Endpoints
    GET  /                            dashboard page
    GET  /api/runs                    every run in the runs directory
    GET  /api/games                   the games a run can be started on
    GET  /api/settings?game=NAME      the settings a new run of that game starts from
    POST /api/runs/start              start a run: game, settings, generations, seed
    POST /api/runs/stop               end the live run after its generation
    GET  /api/compare?runs=A,B        several runs' curves and totals side by side
    GET  /api/status                  a run's status, and the gate's for the live run
    GET  /api/events                  live training events (server-sent events)
    GET  /api/curve                   per-generation series for the charts
    GET  /api/generation?n=N          one generation's agents and replays
    GET  /api/generations             which generations can still be replayed
    POST /api/heartbeat               viewer presence, keeps the hold lease alive
    POST /api/control                 hold / continue / run free
    GET  /api/checkpoints             checkpoints on disk
    GET  /api/export/preview?n=N      export report for the champion, writes nothing
    POST /api/export/choose           start choosing a genome by its average
    GET  /api/export/job?id=ID        how far a choice has got, and its report
    POST /api/export/cancel           stop a choice
    POST /api/export                  write the model file
"""

from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import games
from ..persistence import export_model
from .runs import Lab, RunView
from .stream import EventLog

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
# Responses at least this long are compressed for a browser that accepts it. A
# late generation's replays are about 2 MB as JSON and under 200 KB compressed,
# at level 5, in about 25 ms.
_COMPRESS_FROM = 1024
_COMPRESS_LEVEL = 5
# Runs set side by side at once.
_COMPARE_AT_MOST = 8


def accepts_gzip(header: str | None) -> bool:
    """Whether an Accept-Encoding header allows a gzip response."""
    for part in (header or "").split(","):
        coding, _, params = part.partition(";")
        if coding.strip().lower() not in ("gzip", "*"):
            continue
        quality = params.strip().lower()
        if quality.startswith("q="):
            try:
                return float(quality[2:]) > 0
            except ValueError:
                return False
        return True
    return False


def _whole(value):
    """A whole number sent as text read as one; anything else is left for the caller."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    return value


class NoSuchRun(LookupError):
    pass


class DashboardServer:
    def __init__(self, lab: Lab, host: str, port: int):
        self.lab = lab
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.events = EventLog()
        self._unsubscribe = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> bool:
        handler = _make_handler(self)
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as error:
            print(f"  dashboard unavailable on {self.host}:{self.port}: {error}")
            return False
        self._httpd.daemon_threads = True
        self._unsubscribe = self.lab.listen(lambda run, event: self.events.add(event, run))
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="dashboard", daemon=True
        )
        self._thread.start()
        # Warm torch, for the export's .pt copy, off the request path.
        export_model.preload()
        return True

    def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self.events.close()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except OSError:
                pass

    # ------------------------------------------------------------------ handlers

    def view(self, name: str | None) -> RunView:
        view = self.lab.run(name)
        if view is None:
            raise NoSuchRun(name)
        return view

    def runs(self) -> dict:
        live = self.lab.live
        return {
            "runs": self.lab.runs(),
            "live": None if live is None else live.run_id,
            "can_start": self.lab.can_start,
            "game": self.lab.config.game,
        }

    def games(self) -> dict:
        return {"games": self.lab.games(), "default": self.lab.config.game}

    def settings(self, game: str | None) -> dict:
        game = game or self.lab.config.game
        try:
            return {"game": game, "settings": self.lab.settings_rows(game)}
        except games.UnknownGame as error:
            return {"game": game, "error": str(error)}

    def start_run(self, payload: dict) -> dict:
        chosen = payload.get("settings") or {}
        if not isinstance(chosen, dict):
            return {"problems": ["settings are sent as names and values"]}
        return self.lab.start(
            str(payload.get("game") or self.lab.config.game),
            chosen,
            generations=_whole(payload.get("generations")),
            seed=_whole(payload.get("seed")),
        )

    def status(self, name: str | None) -> dict:
        view = self.view(name)
        status = view.status()
        status["can_start"] = self.lab.can_start
        status["ended_here"] = self.lab.ended_here(view)
        return status

    def control(self, payload: dict) -> dict:
        live = self.lab.live
        if live is None or payload.get("run") not in (None, live.run_id):
            return {"error": "that run is not training here"}
        action = str(payload.get("action", ""))
        gate = live.trainer.gate
        if action == "continue":
            count = _whole(payload.get("count", 1))
            gate.release(count if isinstance(count, int) and count > 0 else 1)
        elif action == "hold":
            gate.set_hold(True)
        elif action == "run_free":
            gate.set_hold(False)
        else:
            return {"error": f"unknown action {action!r}"}
        return {"ok": True, "gate": gate.state().to_dict()}

    def heartbeat(self) -> dict:
        """Tell the live run someone is watching; past runs have no gate."""
        live = self.lab.live
        if live is None:
            return {"ok": True, "gate": None}
        gate = live.trainer.gate
        gate.heartbeat()
        return {"ok": True, "gate": gate.state().to_dict()}

    def compare(self, names: list[str]) -> dict:
        names = [name for name in names if name][:_COMPARE_AT_MOST]
        return self.lab.compare(names)

    def export_write(self, payload: dict) -> dict:
        return self.lab.export(self.view(payload.get("run")), payload)


def _make_handler(server: DashboardServer):
    lab = server.lab

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Reap idle keep-alive connections rather than tying up a thread each.
        timeout = 30

        # ---------------------------------------------------------- transport

        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            compressed = len(payload) >= _COMPRESS_FROM and accepts_gzip(
                self.headers.get("Accept-Encoding")
            )
            if compressed:
                payload = gzip.compress(payload, _COMPRESS_LEVEL)
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                if compressed:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # A dashboard tab navigating away mid-request is routine and
                # says nothing about the health of the run.
                self.close_connection = True

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                self.close_connection = True

        def _json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data, separators=(",", ":"), default=float).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def _static(self, name: str) -> None:
            path = (STATIC_DIR / name).resolve()
            if not path.is_file() or STATIC_DIR.resolve() not in path.parents:
                self._send(b"not found", "text/plain; charset=utf-8", 404)
                return
            content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            self._send(path.read_bytes(), content_type)

        def _stream(self) -> None:
            """Write training events as they happen, for as long as the page stays."""
            log = server.events
            cursor, lost = log.cursor_for(self.headers.get("Last-Event-ID"))
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                # No length is known in advance, so the end of the connection
                # marks the end of the response.
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"retry: 2000\n\n")
                while not log.closed:
                    if lost:
                        self.wfile.write(b"event: resync\ndata: {}\n\n")
                    cursor, messages, lost = log.read(cursor, timeout=15.0)
                    # A comment on a quiet stream shows a closed tab as a
                    # failed write here rather than a thread waiting forever.
                    text = "".join(messages) if messages else ": idle\n\n"
                    self.wfile.write(text.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                self.close_connection = True

        def _body(self) -> dict:
            """Read and parse the request body.

            The body must be consumed on every POST even when its content is
            unused. Closing a connection with unread bytes still in the receive
            buffer makes the peer see an aborted connection rather than the
            response that was already written.
            """
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        # ---------------------------------------------------------- routing

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)

            def text(key: str) -> str | None:
                return query.get(key, [None])[0] or None

            def number(key: str) -> int | None:
                raw = text(key)
                if raw in (None, "latest"):
                    return None
                try:
                    return int(raw)
                except ValueError:
                    return None

            run = text("run")
            try:
                if route in ("/", "/index.html"):
                    self._static("index.html")
                elif route in ("/style.css", "/app.js"):
                    self._static(route.lstrip("/"))
                elif route == "/api/runs":
                    self._json(server.runs())
                elif route == "/api/games":
                    self._json(server.games())
                elif route == "/api/settings":
                    self._json(server.settings(text("game")))
                elif route == "/api/compare":
                    self._json(server.compare((text("runs") or "").split(",")))
                elif route == "/api/status":
                    self._json(server.status(run))
                elif route == "/api/events":
                    self._stream()
                elif route == "/api/curve":
                    self._json(server.view(run).curve())
                elif route == "/api/generations":
                    self._json(server.view(run).generations())
                elif route == "/api/generation":
                    self._json(server.view(run).generation(number("n")))
                elif route == "/api/checkpoints":
                    self._json(server.view(run).checkpoints())
                elif route == "/api/export/preview":
                    self._json(lab.preview(server.view(run), number("n")))
                elif route == "/api/export/job":
                    self._json(lab.job(text("id")))
                else:
                    self._json({"error": "not found"}, 404)
            except NoSuchRun:
                self._json({"error": "no such run", "run": run}, 404)

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            payload = self._body()

            try:
                if route == "/api/heartbeat":
                    self._json(server.heartbeat())
                elif route == "/api/control":
                    self._json(server.control(payload))
                elif route == "/api/runs/start":
                    self._json(server.start_run(payload))
                elif route == "/api/runs/stop":
                    self._json(lab.stop(payload.get("run")))
                elif route == "/api/export/choose":
                    self._json(lab.choose(server.view(payload.get("run"))))
                elif route == "/api/export/cancel":
                    self._json(lab.cancel(payload.get("job")))
                elif route == "/api/export":
                    self._json(server.export_write(payload))
                else:
                    self._json({"error": "not found"}, 404)
            except NoSuchRun:
                self._json({"error": "no such run", "run": payload.get("run")}, 404)

        def log_message(self, fmt: str, *args) -> None:
            """Request logging is suppressed; the trainer owns the console."""

    return Handler
