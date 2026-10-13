"""Dashboard HTTP server.

Standard library only, so the dashboard adds no dependencies to the project.

Endpoints
    GET  /                          dashboard page
    GET  /api/status                live training status and gate state
    GET  /api/curve                 per-generation series for the charts
    GET  /api/generation?n=N        one generation's agents and replays
    GET  /api/generations           which generations can still be replayed
    POST /api/heartbeat             viewer presence, keeps the hold lease alive
    POST /api/control               hold / continue / run free
    GET  /api/checkpoints           checkpoints on disk
    GET  /api/export/preview?n=N    export report, writes nothing
    POST /api/export                write the .pt model
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..persistence import checkpoint, export_torch
from ..telemetry.history import lineage_summaries

if TYPE_CHECKING:  # the trainer imports this module, so keep the edge one-way
    from ..trainer import Trainer

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class DashboardServer:
    def __init__(self, trainer: Trainer, host: str, port: int):
        self.trainer = trainer
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

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
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="dashboard", daemon=True
        )
        self._thread.start()
        # Warm the export dependency off the request path.
        export_torch.preload()
        return True

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except OSError:
                pass

    # ------------------------------------------------------------------ handlers

    def status(self) -> dict:
        return self.trainer.status()

    def curve(self) -> dict:
        return self.trainer.history.curve()

    def generations(self) -> dict:
        available = self.trainer.history.available()
        first, last = self.trainer.history.bounds()
        return {
            "available": available,
            "first": first,
            "last": last,
            "current": self.trainer.population.generation,
            "retained": len(available),
        }

    def generation(self, number: int | None) -> dict:
        record = (
            self.trainer.history.latest() if number is None else self.trainer.history.get(number)
        )
        if record is None:
            return {"error": "generation not available", "generation": number}
        return {
            "generation": record.generation,
            "summary": record.summary,
            "agents": record.agents,
            "board": {
                "width": self.trainer.config.board.width,
                "height": self.trainer.config.board.height,
            },
        }

    def control(self, payload: dict) -> dict:
        action = str(payload.get("action", ""))
        gate = self.trainer.gate
        if action == "continue":
            gate.release(int(payload.get("count", 1)))
        elif action == "hold":
            gate.set_hold(True)
        elif action == "run_free":
            gate.set_hold(False)
        else:
            return {"error": f"unknown action {action!r}"}
        return {"ok": True, "gate": gate.state().to_dict()}

    def checkpoints(self) -> dict:
        found = checkpoint.list_checkpoints(self.trainer.checkpoint_dir)
        return {"checkpoints": [c.to_dict() for c in found]}

    # ------------------------------------------------------------------ export

    def _genome_for(self, generation: int | None) -> tuple[int, np.ndarray | None]:
        """Champion genome as of `generation`, from the archived history."""
        population = self.trainer.population
        if generation is None:
            genome = population.best_ever_genome
            return population.generation, None if genome is None else np.asarray(genome)

        record = self.trainer.history.get(generation)
        if record is not None and record.best_genome:
            return generation, np.asarray(record.best_genome, dtype=np.float32)

        # Fall back to the most recent archived generation at or before the
        # request, so a request for an evicted generation still resolves.
        for candidate in reversed(self.trainer.history.available()):
            if candidate > generation:
                continue
            earlier = self.trainer.history.get(candidate)
            if earlier is not None and earlier.best_genome:
                return candidate, np.asarray(earlier.best_genome, dtype=np.float32)
        return generation, None

    def _export_destination(self) -> Path:
        return self.trainer.config.paths.exports_dir / "evolvarium.pt"

    def export_preview(self, generation: int | None) -> dict:
        resolved, genome = self._genome_for(generation)
        report = export_torch.build_report(
            generation=resolved,
            genome=genome,
            summaries=lineage_summaries(self.trainer.run_dir),
            config=self.trainer.config,
            architecture=self.trainer.population.architecture,
            feature_spec=self.trainer.population.feature_spec,
            run_id=self.trainer.run_id,
            destination=self._export_destination(),
        )
        report["requested_generation"] = generation
        report["resolved_generation"] = resolved
        return report

    def export_write(self, payload: dict) -> dict:
        requested = payload.get("generation")
        generation = int(requested) if requested is not None else None
        resolved, genome = self._genome_for(generation)
        if genome is None:
            return {
                "error": "no champion genome archived for that generation",
                "generation": resolved,
            }

        report = self.export_preview(generation)
        try:
            return export_torch.write_model(
                report=report,
                genome=genome,
                architecture=self.trainer.population.architecture,
                destination=self._export_destination(),
                archive_dir=self.trainer.config.paths.exports_dir / "archive",
            )
        except export_torch.TorchUnavailable as error:
            return {"error": str(error)}


def _make_handler(server: DashboardServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Reap idle keep-alive connections rather than tying up a thread each.
        timeout = 30

        # ---------------------------------------------------------- transport

        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
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
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}

        # ---------------------------------------------------------- routing

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)

            def number(key: str) -> int | None:
                raw = query.get(key, [None])[0]
                if raw in (None, "", "latest"):
                    return None
                try:
                    return int(raw)
                except ValueError:
                    return None

            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route in ("/style.css", "/app.js"):
                self._static(route.lstrip("/"))
            elif route == "/api/status":
                self._json(server.status())
            elif route == "/api/curve":
                self._json(server.curve())
            elif route == "/api/generations":
                self._json(server.generations())
            elif route == "/api/generation":
                self._json(server.generation(number("n")))
            elif route == "/api/checkpoints":
                self._json(server.checkpoints())
            elif route == "/api/export/preview":
                self._json(server.export_preview(number("n")))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            payload = self._body()

            if route == "/api/heartbeat":
                server.trainer.gate.heartbeat()
                self._json({"ok": True, "gate": server.trainer.gate.state().to_dict()})
            elif route == "/api/control":
                self._json(server.control(payload))
            elif route == "/api/export":
                self._json(server.export_write(payload))
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, fmt: str, *args) -> None:
            """Request logging is suppressed; the trainer owns the console."""

    return Handler
