"""Loopback HTTP service for the training monitor and its versioned read API."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from alohamini.paths import WorkspacePaths

from .evaluation import TRAJECTORY_MODES
from .history import AXES, number
from .workspace import Monitor, console_tail

ASSETS = Path(__file__).parent / "static"


def integer(value, low, high):
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"Expected an integer between {low} and {high}")
    return value


def bound(value):
    if value is None:
        return None
    result = float(value)
    if not number(result):
        raise ValueError("Expected a finite axis bound")
    return result


def smoothing(value):
    if value is None or value == "":
        return 0.0
    result = float(value)
    if not number(result) or not 0.0 <= result < 0.999:
        raise ValueError("Expected smooth in [0, 0.999)")
    return result


def dispatch(monitor, parts, query):
    def arg(key, default=None):
        values = query.get(key)
        if values is None:
            return default
        if len(values) != 1:
            raise ValueError(f"Expected a single {key}")
        return values[0]

    def budget():
        return integer(arg("points", "800"), 16, 4096)

    if parts == ["runs"]:
        return {"runs": monitor.catalog.discover()}
    if parts == ["compare"]:
        return monitor.compare()
    if parts == ["datasets"]:
        return {"datasets": monitor.datasets.listing()}
    if len(parts) in (2, 3) and parts[0] == "datasets":
        if len(parts) == 2:
            run = monitor.catalog.get(arg("run")) if arg("run") else None
            return monitor.datasets.details(parts[1], run)
        if parts[2] == "trajectory":
            mode = arg("mode", "position")
            if mode not in TRAJECTORY_MODES:
                raise ValueError("Invalid trajectory mode")
            return monitor.datasets.trajectory(
                parts[1],
                integer(arg("episode", "0"), 0, 10**9),
                arg("field"),
                budget(),
                mode=mode,
            )
    if len(parts) in (2, 3) and parts[0] == "runs":
        run = monitor.catalog.get(parts[1])
        stitch = arg("stitch") == "1"
        if len(parts) == 2:
            return monitor.summary(run, brief=arg("brief") == "1", stitch=stitch)
        if parts[2] == "events":
            return monitor.events(run, stitch=stitch)
        if parts[2] == "diagnostics":
            axis = arg("axis", "step")
            low, high = bound(arg("min")), bound(arg("max"))
            if axis not in AXES or (low is not None and high is not None and low > high):
                raise ValueError("Invalid diagnostics axis or range")
            return monitor.diagnostics(
                run,
                integer(arg("window", "100"), 2, 4096),
                arg("metric", "loss"),
                bound(arg("target")),
                low=low,
                high=high,
                axis=axis,
                stitch=stitch,
            )
        if parts[2] == "series":
            axis = arg("axis", "step")
            keys = list(dict.fromkeys(query.get("metric", ["loss"])))
            low, high = bound(arg("min")), bound(arg("max"))
            if (
                axis not in AXES
                or len(keys) > 48
                or (low is not None and high is not None and low > high)
            ):
                raise ValueError("Invalid axis, range or metric count")
            cursors = {p: arg("after_" + p) for p in ("train", "validation")}
            return monitor.chart(
                run,
                keys,
                axis,
                low,
                high,
                budget(),
                cursors,
                integer(arg("mean_window", "0"), 0, 4096),
                smooth=smoothing(arg("smooth")),
                stitch=stitch,
            )
        if parts[2] == "records":
            phase = arg("phase", "train")
            if phase not in ("train", "validation"):
                raise ValueError("Unknown metric phase")
            return monitor.records(
                run, phase, arg("cursor"), integer(arg("limit", "100"), 1, 1000), arg("tail") == "1"
            )
        if parts[2] == "console":
            return console_tail(run, arg("name"), arg("cursor"))
    raise KeyError("Unknown route")


class Handler(BaseHTTPRequestHandler):
    server_version = "AlohaMiniMonitor/2"

    def log_message(self, fmt, *args):
        logging.getLogger(__name__).debug(fmt, *args)

    def _cors_headers(self):
        origin = self.headers.get("Origin")
        if origin and origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Expose-Headers", "ETag")
            self.send_header("Vary", "Origin")

    def send_bytes(self, status, body, content_type, *, etag=None, disposition=None):
        if etag and status == 200 and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if etag:
            self.send_header("ETag", etag)
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def result(self, status, value):
        body = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        etag = f'"{hashlib.sha256(body).hexdigest()[:20]}"' if status == 200 else None
        self.send_bytes(status, body, "application/json; charset=utf-8", etag=etag)

    def do_GET(self):
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname
            if host not in ("localhost", "127.0.0.1", "::1"):
                self.result(403, {"error": "Expected a loopback Host"})
                return
            url = urlsplit(self.path)
            if url.path.startswith("/api/v1/"):
                parts = url.path[8:].split("/")
                query = parse_qs(url.query, max_num_fields=64)
                if len(parts) == 3 and parts[0] == "runs" and parts[2] == "export.csv":
                    run = self.server.monitor.catalog.get(parts[1])
                    axis = (query.get("axis") or ["step"])[0]
                    if axis not in AXES:
                        raise ValueError("Invalid axis")
                    keys = list(dict.fromkeys(query.get("metric", ["loss"])))
                    low = bound((query.get("min") or [None])[0])
                    high = bound((query.get("max") or [None])[0])
                    stitch = (query.get("stitch") or ["0"])[0] == "1"
                    data = self.server.monitor.export_csv(run, keys, axis, low, high, stitch=stitch)
                    safe_name = "".join(
                        c if c.isalnum() or c in "._-" else "_" for c in run.root.name
                    )
                    self.send_bytes(
                        200,
                        data,
                        "text/csv; charset=utf-8",
                        disposition=f'attachment; filename="{safe_name}-metrics.csv"',
                    )
                    return
                self.result(200, dispatch(self.server.monitor, parts, query))
            elif url.path in self.server.assets:
                data, mime = self.server.assets[url.path]
                self.send_bytes(200, data, mime)
            else:
                self.result(404, {"error": "Not found"})
        except (ConnectionError, TimeoutError):
            pass
        except (FileNotFoundError, KeyError) as exc:
            self.result(404, {"error": str(exc.args[0])})
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            self.result(400, {"error": str(exc)})
        except Exception:
            logging.getLogger(__name__).exception("Monitor request failed")
            self.result(500, {"error": "Monitor request failed"})


class MonitorServer(ThreadingHTTPServer):
    def __init__(self, port, monitor, origins=()):
        self.monitor = monitor
        self.origins = set(origins)
        self.assets = {}
        for path in ASSETS.iterdir():
            if path.suffix in {".html", ".css", ".js", ".svg"}:
                mime = mimetypes.guess_type(path.name)[0] or "text/plain"
                self.assets["/" + path.name] = path.read_bytes(), mime + "; charset=utf-8"
        self.assets["/"] = self.assets["/index.html"]
        super().__init__(("127.0.0.1", port), Handler)


def main(argv=None):
    paths = WorkspacePaths()
    parser = argparse.ArgumentParser(
        prog="alohamini train monitor",
        description="View local training metrics, logs and evaluations.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=paths.runs,
        help="Run collection or a single output directory (default: workspace/runs)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=paths.logs / "training",
        help="Detached launcher config and console directory",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        action="append",
        help="Evaluation dataset or collection; repeat to add more roots",
    )
    parser.add_argument("--port", type=int, default=7860, help="Local port (0 chooses a free port)")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the browser")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Allow browser API reads from this exact Studio origin",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be in 0..65535")
    for origin in args.allow_origin:
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path
            or parsed.query
            or parsed.fragment
            or "*" in origin
        ):
            parser.error("allow-origin must be an exact HTTP(S) origin")
    monitor = Monitor(args.runs_dir, args.logs_dir, args.datasets_dir or [paths.datasets])
    with MonitorServer(args.port, monitor, args.allow_origin) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"AlohaMini Training Monitor  {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
