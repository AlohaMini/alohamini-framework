"""Discover native training attempts and their detached launch logs."""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .diagnostics import analyze
from .evaluation import EvaluationLibrary
from .files import directories, document, identifier, stamp
from .history import History

RESUME_CONFIG = re.compile(r"resume-(\d+)-(\d+)\.json\Z")
RESUME_METRICS = re.compile(r"metrics-from-(\d+)-(\d+)\.jsonl\Z")


@dataclass
class Attempt:
    root: Path
    suffix: str = ""
    logs: dict = field(default_factory=dict)
    launch_config: Path | None = None
    launch_report: Path | None = None
    ordinal: int = 0

    @property
    def id(self):
        return identifier(self.root, self.suffix)

    @property
    def config_path(self):
        if self.suffix:
            return self.root / f"resume-{self.suffix}.json"
        return self.root / "train.json"

    def metric_path(self, phase):
        name = "metrics" if phase == "train" else "validation-metrics"
        suffix = f"-from-{self.suffix}" if self.suffix else ""
        return self.root / f"{name}{suffix}.jsonl"

    @property
    def schema_path(self):
        return self.root / (self.metric_path("train").stem + "-schema.json")

    @property
    def resources_path(self):
        recorded = self.root / (self.metric_path("train").stem + "-resources.json")
        return recorded if self.suffix or stamp(recorded) else self.root / "resources.json"

    def config(self):
        return document(self.config_path) or (
            document(self.launch_config) if self.launch_config else {}
        )

    def item(self, previous=None):
        configuration = [
            stamp(self.config_path),
            stamp(self.schema_path),
            stamp(self.resources_path),
        ]
        if self.launch_config:
            configuration.append(stamp(self.launch_config))
        if self.launch_report:
            configuration.append(stamp(self.launch_report))
        phases = [stamp(self.metric_path(p)) for p in ("train", "validation")]
        # The final offline report is shared by every attempt in the directory.
        # Use attempt-specific files to order runs and display their update time.
        own_files = [*configuration, *phases]
        metadata_version = identifier(
            *configuration, stamp(self.root / "offline-evaluation.json"), tuple(self.logs)
        )
        cfg = (
            previous
            if previous and previous["metadata_version"] == metadata_version
            else self.config()
        )
        return {
            "id": self.id,
            "name": self.root.name,
            "attempt": self.suffix or None,
            "ordinal": self.ordinal,
            "policy": cfg.get("policy"),
            "steps": cfg.get("steps"),
            "version": identifier(metadata_version, self.ordinal, *phases),
            "metadata_version": metadata_version,
            "updated_ns": str(max((s[3] for s in own_files if s), default=0)),
        }


@lru_cache(maxsize=512)
def attempt_suffixes(root, directory_signature):
    """Appending metric records leaves the directory listing unchanged."""
    suffixes = set()
    if directory_signature is None:
        return suffixes
    try:
        for path in root.iterdir():
            if path.name in ("train.json", "metrics.jsonl", "training.lock"):
                suffix = ""
            else:
                match = RESUME_CONFIG.fullmatch(path.name) or RESUME_METRICS.fullmatch(path.name)
                if not match:
                    continue
                suffix = "-".join(match.groups())
            if not path.is_symlink() and path.is_file():
                suffixes.add(suffix)
    except (FileNotFoundError, NotADirectoryError):
        pass
    return suffixes


@lru_cache(maxsize=4096)
def launch_target(path, signature, workspace_runs):
    cfg = document(path)
    location, name = cfg.get("output_dir"), cfg.get("run_name")
    if isinstance(location, str) and Path(location).is_absolute():
        root = Path(location).resolve()
    elif not location and isinstance(name, str) and Path(name).name == name:
        root = (workspace_runs / name).resolve()
    else:
        return None, False
    return root, bool(cfg.get("resume"))


class Catalog:
    def __init__(self, runs, logs):
        self.runs = runs.expanduser().resolve()
        self.logs = logs.expanduser().resolve()
        self.lock = Lock()
        self.until = 0
        self.attempts = {}
        self.items = []

    def discover(self):
        with self.lock:
            if time.monotonic() < self.until:
                return self.items
            attempts = {}

            def attempt(root, suffix=""):
                key = root, suffix
                if key not in attempts:
                    attempts[key] = Attempt(root, suffix)
                return attempts[key]

            def inspect(root):
                suffixes = attempt_suffixes(root, stamp(root))
                if "" in suffixes:
                    attempt(root)
                resumes = sorted(filter(None, suffixes), key=lambda s: int(s.split("-")[1]))
                for ordinal, suffix in enumerate(resumes, 1):
                    attempt(root, suffix).ordinal = ordinal
                return bool(suffixes)

            single = inspect(self.runs)
            if not single:
                for root in directories(self.runs):
                    inspect(root)

            launches = {}
            for path in sorted(self.logs.glob("*.json")):
                root, resumed = launch_target(path, stamp(path), self.logs.parent.parent / "runs")
                if root is None:
                    continue
                if root != self.runs and (single or root.parent != self.runs):
                    continue
                log = path.with_suffix(".log")
                if not resumed:
                    attempt(root).launch_config = path
                if stamp(log):
                    launches.setdefault(root, {})[log.name] = log
            for run in attempts.values():
                run.logs = launches.get(run.root, {}).copy()
                if run.suffix:
                    continue
                # Historical launch bundles use an exact run name under a sibling logs/.
                folders = [self.logs]
                if run.root.parent.name == "runs":
                    folders.append(run.root.parent.parent / "logs")
                for folder in dict.fromkeys(folders):
                    path = folder / f"{run.root.name}.preflight.json"
                    report = document(path)
                    if type(report.get("passed")) is not bool:
                        continue
                    cfg = run.config()
                    if (
                        report.get("dataset")
                        and cfg.get("dataset")
                        and report["dataset"] != cfg["dataset"]
                    ):
                        continue
                    run.launch_report = path
                    for suffix in (".log", ".preflight.log"):
                        log = folder / (run.root.name + suffix)
                        if stamp(log):
                            run.logs.setdefault(log.name, log)
                    break
            self.attempts = {a.id: a for a in attempts.values()}
            previous = {item["id"]: item for item in self.items}
            self.items = sorted(
                (a.item(previous.get(a.id)) for a in attempts.values()),
                key=lambda a: int(a["updated_ns"]),
                reverse=True,
            )
            self.until = time.monotonic() + 0.75
            return self.items

    def get(self, key):
        self.discover()
        try:
            return self.attempts[key]
        except KeyError:
            raise KeyError("Unknown training attempt") from None


class IndexedAttempt:
    def __init__(self, run):
        self.lock = Lock()
        self.phases = {p: History(run.metric_path(p)) for p in ("train", "validation")}


class Monitor:
    """One shared, lazily refreshed index per recently viewed attempt."""

    def __init__(self, runs, logs, datasets):
        self.catalog = Catalog(runs, logs)
        self.datasets = EvaluationLibrary(datasets)
        self.cache = OrderedDict()
        self.lock = Lock()

    def index(self, run):
        with self.lock:
            key = run.id
            if key not in self.cache:
                if len(self.cache) >= 6:
                    self.cache.popitem(last=False)
                self.cache[key] = IndexedAttempt(run)
            self.cache.move_to_end(key)
            return self.cache[key]

    def summary(self, run, *, brief=False):
        result = run.item()
        index = self.index(run)
        with index.lock:
            phases = {}
            for name, history in index.phases.items():
                history.refresh()
                phases[name] = history.summary()
        result["phases"] = phases
        if not brief:
            result.update(
                config=run.config(),
                schema=document(run.schema_path),
                preflight=document(run.resources_path) or None,
                launch_preflight=document(run.launch_report) if run.launch_report else None,
                console_parameters=console_parameters(run),
                logs=list(run.logs),
                offline=document(run.root / "offline-evaluation.json") or None,
            )
        return result

    def chart(self, run, keys, axis, low, high, budget, cursors, mean_window=0):
        index = self.index(run)
        with index.lock:
            result = {}
            for phase, history in index.phases.items():
                history.refresh()
                result[phase] = history.chart(
                    keys,
                    axis,
                    low,
                    high,
                    budget,
                    cursors[phase],
                    mean_window if phase == "train" else 0,
                )
            return result

    def diagnostics(self, run, window, metric, target):
        index = self.index(run)
        with index.lock:
            for history in index.phases.values():
                history.refresh()
            return analyze(
                index.phases["train"],
                index.phases["validation"],
                run.config(),
                window,
                metric,
                target,
            )

    def records(self, run, phase, cursor, limit, tail):
        index = self.index(run)
        with index.lock:
            history = index.phases[phase]
            return history.records(cursor, limit, tail)


@lru_cache(maxsize=128)
def _parameter_header(path, identity, length, modified):
    with path.open("rb") as stream:
        header = stream.read(length)
    counts = {}
    for key in ("num_learnable_params", "num_total_params"):
        match = re.search(rb"\b" + key.encode() + rb"=(\d+)\b", header)
        if not match:
            return None
        counts[key] = int(match[1])
    return counts if counts["num_total_params"] >= counts["num_learnable_params"] else None


def console_parameters(run):
    if document(run.schema_path).get("runtime"):
        return None
    name = document(run.schema_path).get("console_log") or (
        run.root.name + ".log" if not run.suffix else ""
    )
    path = run.logs.get(name)
    sig = stamp(path) if path else None
    if not sig:
        return None
    limit = 256 * 1024
    return _parameter_header(path, sig[:2], min(sig[2], limit), sig[3] if sig[2] <= limit else 0)


def console_tail(run, name, cursor):
    """A bounded byte tail; do not read or transfer the console history on refresh."""
    if name not in run.logs:
        raise KeyError("Unknown launch log")
    path = run.logs[name]
    sig = stamp(path)
    if not sig:
        raise KeyError("Missing launch log")
    prefix = f"{sig[0]}-{sig[1]}."
    offset = None
    if cursor and cursor.startswith(prefix):
        value = cursor[len(prefix) :]
        if value.isascii() and value.isdigit() and int(value) <= sig[2]:
            offset = int(value)
    start = max(offset or 0, sig[2] - 48 * 1024)
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(sig[2] - start)
        # Leave an incomplete UTF-8 suffix for the next read.
        from codecs import getincrementaldecoder

        decoder = getincrementaldecoder("utf-8")("replace")
        text = decoder.decode(data)
        end = stream.tell() - len(decoder.getstate()[0])
    return {"text": text, "cursor": prefix + str(end), "reset": start != offset}
