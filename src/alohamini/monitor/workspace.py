"""Discover native training attempts, stitched resume lineages, events and compare tables."""

from __future__ import annotations

import csv
import io
import json
import math
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
from .history import (
    History,
    effective_batch_from_config,
    generalization_gap_series,
    number,
    stitched_chart,
    stitched_pairs,
)

RESUME_CONFIG = re.compile(r"resume-(\d+)-(\d+)\.json\Z")
RESUME_METRICS = re.compile(r"metrics-from-(\d+)-(\d+)\.jsonl\Z")
ACTIVE_MTIME_WINDOW_S = 180.0


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
    def group_id(self):
        return identifier(self.root)

    @property
    def resume_start_step(self):
        if not self.suffix:
            return None
        try:
            return int(self.suffix.split("-")[0])
        except ValueError:
            return None

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
    def events_path(self):
        suffix = f"-from-{self.suffix}" if self.suffix else ""
        return self.root / f"events{suffix}.jsonl"

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
        events_sig = stamp(self.events_path)
        lock_sig = stamp(self.root / "training.lock")
        # The final offline report is shared by every attempt in the directory.
        # Use attempt-specific files to order runs and display their update time.
        own_files = [*configuration, *phases, events_sig]
        metadata_version = identifier(
            *configuration,
            stamp(self.root / "offline-evaluation.json"),
            tuple(self.logs),
            None if self.suffix else stamp(self.root / "statistics.json"),
        )
        cfg = (
            previous
            if previous and previous["metadata_version"] == metadata_version
            else self.config()
        )
        latest_ns = max((s[3] for s in own_files if s), default=0)
        now_s = time.time()
        recently_written = bool(latest_ns and (now_s - latest_ns / 1e9) <= ACTIVE_MTIME_WINDOW_S)
        status = "running" if (lock_sig and recently_written) else "idle"
        return {
            "id": self.id,
            "group_id": self.group_id,
            "name": self.root.name,
            "attempt": self.suffix or None,
            "resume_step": self.resume_start_step,
            "ordinal": self.ordinal,
            "policy": cfg.get("policy"),
            "steps": cfg.get("steps"),
            "status": status,
            "version": identifier(metadata_version, self.ordinal, *phases, events_sig, status),
            "metadata_version": metadata_version,
            "updated_ns": str(latest_ns),
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

    def siblings(self, run):
        self.discover()
        return sorted(
            (a for a in self.attempts.values() if a.root == run.root),
            key=lambda a: (0 if not a.suffix else 1, a.ordinal),
        )


class IndexedAttempt:
    def __init__(self, run):
        self.lock = Lock()
        self.phases = {p: History(run.metric_path(p)) for p in ("train", "validation")}


@lru_cache(maxsize=128)
def recorded_asset_paths(path, signature, policy):
    """Keep only resolved model locators from a legacy initial run's data contract."""
    if not signature:
        return {}
    try:
        with path.open("rb") as stream:
            artifact = json.load(stream)
    except (OSError, ValueError, RecursionError):
        return {}
    contract = artifact.get("contract") if isinstance(artifact, dict) else None
    if not isinstance(contract, dict) or contract.get("policy") != policy:
        return {}
    model = contract.get("model")
    if not isinstance(model, dict):
        return {}
    return {
        k: v
        for k, v in model.items()
        if isinstance(v, str)
        and (k.endswith(("_path", "_model_id", "_model_name", "_name_or_path")) or "tokenizer" in k)
    }


def _read_jsonl(path):
    if not stamp(path):
        return []
    rows = []
    try:
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def _checkpoint_steps(root):
    ckpt_root = root / "checkpoints"
    if not stamp(ckpt_root):
        return []
    steps = []
    for path in directories(ckpt_root):
        if path.name.isdigit():
            steps.append(int(path.name))
    return sorted(steps)


def _offline_mae_mean(offline):
    if not isinstance(offline, dict):
        return None
    mae = offline.get("mae_by_action")
    if not isinstance(mae, dict) or not mae:
        return None
    vals = [v for v in mae.values() if number(v)]
    return sum(vals) / len(vals) if vals else None


def _offline_checkpoint_step(offline):
    if not isinstance(offline, dict):
        return None
    path = offline.get("checkpoint")
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return None


def _best_validation(history):
    loss = history.series.get("loss")
    if not loss:
        return None
    rows = loss.row_indices()
    low_idx, _ = loss._extrema(0, len(rows))
    if low_idx is None:
        return None
    row = rows[low_idx]
    return {
        "loss": loss.values[low_idx],
        "step": int(history.axes["step"][row]),
    }


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
                if len(self.cache) >= 24:
                    self.cache.popitem(last=False)
                self.cache[key] = IndexedAttempt(run)
            self.cache.move_to_end(key)
            return self.cache[key]

    def lineage(self, run):
        """Return ordered [(Attempt, max_step_inclusive_or_None)] leading up to `run`."""
        siblings = self.catalog.siblings(run)
        if not run.suffix or len(siblings) <= 1:
            return [(run, None)]
        # Walk backwards from `run` to find earlier attempts covering `resume_start_step`.
        chain = []
        current = run
        visited = {run.id}
        while current.suffix:
            target = current.resume_start_step
            if target is None:
                break
            candidates = [
                s
                for s in siblings
                if s.id not in visited
                and (s.ordinal < current.ordinal)
                and (
                    not s.suffix
                    or (s.resume_start_step is not None and s.resume_start_step < target)
                )
            ]
            if not candidates:
                break
            # Prefer the closest earlier attempt whose recorded steps reach >= target,
            # falling back to the initial run if none fully cover it.
            chosen = None
            for cand in reversed(candidates):
                idx = self.index(cand)
                with idx.lock:
                    idx.phases["train"].refresh()
                    steps = idx.phases["train"].axes.get("step")
                    if steps and len(steps) and steps[0] <= target <= steps[-1]:
                        chosen = cand
                        break
            if chosen is None:
                chosen = candidates[0]
            chain.append((chosen, target))
            visited.add(chosen.id)
            current = chosen
        chain.reverse()
        chain.append((run, None))
        return chain

    def _lineage_histories(self, run, phase, stitch):
        if not stitch:
            idx = self.index(run)
            with idx.lock:
                h = idx.phases[phase]
                h.refresh()
            return [(h, None)], []
        chain = self.lineage(run)
        segments = []
        resume_steps = []
        for attempt, max_step in chain:
            idx = self.index(attempt)
            with idx.lock:
                h = idx.phases[phase]
                h.refresh()
            segments.append((h, max_step))
            if attempt.resume_start_step is not None and len(segments) > 1:
                resume_steps.append(attempt.resume_start_step)
        return segments, resume_steps

    def summary(self, run, *, brief=False, stitch=False):
        result = run.item()
        siblings = self.catalog.siblings(run)
        result["has_resumes"] = len(siblings) > 1
        cfg = run.config()
        schema = document(run.schema_path)
        chain = self.lineage(run) if (stitch and len(siblings) > 1) else [(run, None)]
        phases = {}
        for name in ("train", "validation"):
            if len(chain) == 1:
                index = self.index(run)
                with index.lock:
                    history = index.phases[name]
                    history.refresh()
                    phases[name] = history.summary()
                    if name == "train":
                        phases[name]["tail_mean_loss"] = history.tail_mean("loss", 100)
                        phases[name]["tail_component_means"] = {
                            k: m
                            for k in phases[name]["policy_metrics"]
                            if (m := history.tail_mean(k, 100)) is not None
                        }
                        phases[name]["derived_metrics"] = history.available_derived(cfg, schema)
                    else:
                        phases[name]["best"] = _best_validation(history)
            else:
                records = 0
                invalid = 0
                nonfinite = {}
                axes = None
                metrics = []
                policy_metrics = []
                latest = {}
                best_val = None
                derived = []
                tail_mean = None
                for attempt, max_step in chain:
                    index = self.index(attempt)
                    with index.lock:
                        h = index.phases[name]
                        h.refresh()
                        s = h.summary()
                        steps = h.axes.get("step")
                        if max_step is not None and steps and len(steps):
                            from bisect import bisect_right

                            records += bisect_right(steps, max_step)
                        else:
                            records += s["records"]
                        invalid += s["invalid_lines"]
                        for k, v in s["nonfinite"].items():
                            nonfinite[k] = nonfinite.get(k, 0) + v
                        if s["records"]:
                            axes = (
                                s["axes"] if axes is None else [a for a in axes if a in s["axes"]]
                            )
                            latest = s["latest"]
                        for k in s["metrics"]:
                            if k not in metrics:
                                metrics.append(k)
                        for k in s["policy_metrics"]:
                            if k not in policy_metrics:
                                policy_metrics.append(k)
                        if name == "train":
                            tm = h.tail_mean("loss", 100)
                            if tm is not None:
                                tail_mean = tm
                            for k in h.available_derived(cfg, schema):
                                if k not in derived:
                                    derived.append(k)
                        else:
                            bv = _best_validation(h)
                            if bv and (best_val is None or bv["loss"] < best_val["loss"]):
                                best_val = bv
                phases[name] = {
                    "records": records,
                    "invalid_lines": invalid,
                    "nonfinite": nonfinite,
                    "axes": axes or [],
                    "metrics": metrics,
                    "policy_metrics": policy_metrics,
                    "latest": latest,
                }
                if name == "train":
                    phases[name]["tail_mean_loss"] = tail_mean
                    phases[name]["tail_component_means"] = {
                        k: m
                        for k in policy_metrics
                        if (m := self.index(chain[-1][0]).phases["train"].tail_mean(k, 100))
                        is not None
                    }
                    phases[name]["derived_metrics"] = derived
                else:
                    phases[name]["best"] = best_val
        if phases["validation"]["records"] and phases["train"]["records"]:
            if "derived/generalization_gap" not in phases["train"].get("derived_metrics", []):
                phases["train"].setdefault("derived_metrics", []).append(
                    "derived/generalization_gap"
                )
        # Update completed status if target steps reached and not currently running
        latest_step = phases["train"]["latest"].get("step")
        if (
            result["status"] != "running"
            and number(latest_step)
            and number(result.get("steps"))
            and latest_step >= result["steps"]
        ):
            result["status"] = "completed"
        result["phases"] = phases
        result["stitched"] = len(chain) > 1
        result["resume_steps"] = [
            a.resume_start_step for a, _ in chain[1:] if a.resume_start_step is not None
        ]
        if not brief:
            report = document(run.resources_path) or None
            stats = run.root / "statistics.json"
            result.update(
                config=cfg,
                schema=schema,
                preflight=report,
                recorded_asset_paths=(
                    recorded_asset_paths(stats, stamp(stats), result["policy"])
                    if not run.suffix and not report
                    else {}
                ),
                launch_preflight=document(run.launch_report) if run.launch_report else None,
                console_parameters=console_parameters(run),
                logs=list(run.logs),
                offline=document(run.root / "offline-evaluation.json") or None,
            )
        return result

    def chart(
        self,
        run,
        keys,
        axis,
        low,
        high,
        budget,
        cursors,
        mean_window=0,
        *,
        smooth=0.0,
        stitch=False,
    ):
        cfg = run.config()
        schema = document(run.schema_path)
        want_gap = "derived/generalization_gap" in keys
        query_keys = [k for k in keys if k != "derived/generalization_gap"]
        chain = (
            self.lineage(run) if (stitch and len(self.catalog.siblings(run)) > 1) else [(run, None)]
        )

        if len(chain) == 1:
            index = self.index(run)
            with index.lock:
                result = {}
                for phase, history in index.phases.items():
                    history.refresh()
                    result[phase] = history.chart(
                        query_keys,
                        axis,
                        low,
                        high,
                        budget,
                        cursors[phase],
                        mean_window if phase == "train" else 0,
                        smooth=smooth if phase == "train" else 0.0,
                        config=cfg,
                        schema=schema,
                    )
                if want_gap:
                    gap, rb, source = generalization_gap_series(
                        [(index.phases["train"], None)],
                        [(index.phases["validation"], None)],
                        axis,
                        low,
                        high,
                        budget,
                    )
                    if gap is not None:
                        result["validation"]["series"]["derived/generalization_gap"] = gap
                        result["validation"]["gap_source"] = source
                        if rb is not None:
                            result["validation"].setdefault("bounds", {})[
                                "derived/generalization_gap"
                            ] = rb
        else:
            train_segs, resume_steps = self._lineage_histories(run, "train", True)
            val_segs, _ = self._lineage_histories(run, "validation", True)
            result = {
                "train": stitched_chart(
                    train_segs,
                    query_keys,
                    axis,
                    low,
                    high,
                    budget,
                    mean_window,
                    smooth=smooth,
                    config=cfg,
                    schema=schema,
                ),
                "validation": stitched_chart(
                    val_segs,
                    query_keys,
                    axis,
                    low,
                    high,
                    budget,
                    0,
                    smooth=0.0,
                    config=cfg,
                    schema=schema,
                ),
                "resume_steps": resume_steps,
            }
            if want_gap:
                gap, rb, source = generalization_gap_series(
                    train_segs, val_segs, axis, low, high, budget
                )
                if gap is not None:
                    result["validation"]["series"]["derived/generalization_gap"] = gap
                    result["validation"]["gap_source"] = source
                    if rb is not None:
                        result["validation"].setdefault("bounds", {})[
                            "derived/generalization_gap"
                        ] = rb
        clip = (cfg.get("optimizer") or {}).get("grad_clip_norm")
        if number(clip) and clip > 0:
            result["clip_threshold"] = clip
        return result

    def events(self, run, *, stitch=False):
        chain = (
            self.lineage(run) if (stitch and len(self.catalog.siblings(run)) > 1) else [(run, None)]
        )
        events = []
        seen = set()

        def push(kind, step, **extra):
            if not number(step):
                return
            key = (kind, int(step))
            if key in seen:
                return
            seen.add(key)
            events.append({"type": kind, "step": int(step), **extra})

        # 1. Explicit events.jsonl records
        for attempt, max_step in chain:
            for item in _read_jsonl(attempt.events_path):
                step = item.get("step")
                kind = item.get("type")
                if not isinstance(kind, str) or not number(step):
                    continue
                if max_step is not None and step > max_step:
                    continue
                push(kind, step, **{k: v for k, v in item.items() if k not in ("type", "step")})
            if attempt.resume_start_step is not None and len(chain) > 1:
                push("resume", attempt.resume_start_step, attempt=attempt.suffix)

        # 2. Config warmup end
        cfg = run.config()
        warmup = (cfg.get("scheduler") or {}).get("warmup_steps")
        if number(warmup) and warmup > 0:
            push("warmup_end", int(warmup))

        # 3. Synthesized events from indexed metrics (best val, AMP overflow, non-finite)
        best_val = None
        for attempt, max_step in chain:
            idx = self.index(attempt)
            with idx.lock:
                for ph in ("train", "validation"):
                    idx.phases[ph].refresh()
                th, vh = idx.phases["train"], idx.phases["validation"]
                bv = _best_validation(vh)
                if bv and (max_step is None or bv["step"] <= max_step):
                    if best_val is None or bv["loss"] < best_val["loss"]:
                        best_val = bv
                # Skipped update jumps
                sk = th.series.get("skipped_updates")
                if sk and len(sk.values) > 1:
                    rows = sk.row_indices()
                    steps = th.axes["step"]
                    prev = sk.values[0]
                    for i in range(1, len(sk.values)):
                        cur = sk.values[i]
                        if math.isfinite(cur) and math.isfinite(prev) and cur > prev:
                            st = int(steps[rows[i]])
                            if max_step is None or st <= max_step:
                                push("amp_overflow", st, skipped=int(cur - prev))
                        prev = cur
                # First few non-finite markers
                for metric_name, ser in th.series.items():
                    if ser.gaps:
                        rows = ser.row_indices()
                        steps = th.axes["step"]
                        for g_idx in ser.gaps[:8]:
                            st = int(steps[rows[g_idx]])
                            if max_step is None or st <= max_step:
                                push("nonfinite", st, metric=metric_name)
        if best_val:
            push("validation_best", best_val["step"], loss=best_val["loss"])

        # 4. Checkpoints on disk & offline evaluation report
        for ckpt_step in _checkpoint_steps(run.root):
            push("checkpoint", ckpt_step)
        offline = document(run.root / "offline-evaluation.json")
        mae = _offline_mae_mean(offline)
        ckpt_step = _offline_checkpoint_step(offline)
        if mae is not None and ckpt_step is not None:
            push("offline_eval", ckpt_step, mae_mean=mae, weights=offline.get("weights"))

        events.sort(key=lambda e: (e["step"], e["type"]))
        return {"events": events}

    def diagnostics(
        self, run, window, metric, target, *, low=None, high=None, axis="step", stitch=False
    ):
        chain = (
            self.lineage(run) if (stitch and len(self.catalog.siblings(run)) > 1) else [(run, None)]
        )
        if len(chain) == 1:
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
                    low=low,
                    high=high,
                    axis=axis,
                )
        # Build lightweight stitched in-memory History views for multi-attempt diagnostics.
        th = History(run.metric_path("train"))
        vh = History(run.metric_path("validation"))
        for target_hist, phase in ((th, "train"), (vh, "validation")):
            segs, _ = self._lineage_histories(run, phase, True)
            # Re-index only accepted rows within each segment's step cutoff.
            all_keys = []
            for seg_h, _ in segs:
                for k in seg_h.series:
                    if k not in all_keys:
                        all_keys.append(k)
            elapsed_offset = 0.0
            last_step = -1
            for seg_h, max_step in segs:
                steps = seg_h.axes.get("step")
                if not steps or not len(steps):
                    continue
                from bisect import bisect_right

                limit = bisect_right(steps, max_step) if max_step is not None else len(steps)
                # Extract row-level records from series efficiently
                samples = seg_h.axes.get("samples")
                elapsed = seg_h.axes.get("elapsed_s")
                row_maps = {}
                for k, s in seg_h.series.items():
                    row_maps[k] = (
                        s.values
                        if s.rows is None
                        and s.first_row == 0
                        and len(s.values) == len(seg_h.offsets)
                        else dict(zip(s.row_indices(), s.values, strict=True))
                    )
                for row in range(limit):
                    st = int(steps[row])
                    if st <= last_step:
                        continue
                    last_step = st
                    rec = {"step": st}
                    if samples is not None:
                        rec["samples"] = samples[row]
                    if elapsed is not None:
                        rec["elapsed_s"] = elapsed[row] + elapsed_offset
                    for k, m in row_maps.items():
                        v = m[row] if not isinstance(m, dict) else m.get(row)
                        if v is not None:
                            rec[k] = v if math.isfinite(v) else None
                    target_hist._append(rec)
                if elapsed is not None and limit > 0:
                    elapsed_offset += elapsed[limit - 1]
            if segs and segs[-1][0].latest:
                target_hist.latest = segs[-1][0].latest.copy()
        return analyze(th, vh, run.config(), window, metric, target, low=low, high=high, axis=axis)

    def compare(self):
        """Compact summary of every run (hyperparams and outcomes) for Compare and Studio."""
        entries = self.catalog.discover()
        rows = []
        # Identify latest attempt per group so the UI can toggle grouped vs all attempts.
        latest_by_group = {}
        for item in entries:
            gid = item["group_id"]
            if gid not in latest_by_group:
                latest_by_group[gid] = item["id"]
        for item in entries:
            run = self.catalog.attempts.get(item["id"])
            if run is None:
                continue
            cfg = run.config()
            schema = document(run.schema_path)
            runtime = schema.get("runtime") or {}
            offline = document(run.root / "offline-evaluation.json") or None
            idx = self.index(run)
            with idx.lock:
                th, vh = idx.phases["train"], idx.phases["validation"]
                th.refresh()
                vh.refresh()
                latest_step = th.latest.get("step")
                train_mean = th.tail_mean("loss", 100)
                latest_val = vh.latest.get("loss")
                best_val = _best_validation(vh)
                elapsed_s = (
                    th.axes["elapsed_s"][-1]
                    if "elapsed_s" in th.axes and len(th.axes["elapsed_s"])
                    else None
                )
                samples = (
                    th.axes["samples"][-1]
                    if "samples" in th.axes and len(th.axes["samples"])
                    else None
                )
                # Quick throughput from last 100 steps
                sps = None
                if "samples_per_s" in th.series and len(th.series["samples_per_s"].values):
                    sps = th.tail_mean("samples_per_s", 100)
                elif elapsed_s and samples and elapsed_s > 0:
                    sps = samples / elapsed_s
                elif "update_s" in th.series and "dataloading_s" in th.series:
                    u = th.tail_mean("update_s", 100)
                    d = th.tail_mean("dataloading_s", 100)
                    eb = effective_batch_from_config(cfg, schema)
                    if u is not None and d is not None and (u + d) > 0 and eb:
                        sps = eb / (u + d)
                # Quick clipping & grad norm
                clip_thr = (cfg.get("optimizer") or {}).get("grad_clip_norm")
                grad_mean = th.tail_mean("grad_norm", 100)
                clip_frac = None
                if "grad_clipped" in th.series:
                    clip_frac = th.tail_mean("grad_clipped", 100)
                elif "grad_norm" in th.series and number(clip_thr) and clip_thr > 0:
                    gn = th.series["grad_norm"].values[-100:]
                    fin = [v for v in gn if math.isfinite(v)]
                    if fin:
                        clip_frac = sum(1 for v in fin if v > clip_thr) / len(fin)
                gen_gap = vh.latest.get("generalization_gap")
            # Inherit stitched lineage validation for resumed attempts with no new eval yet.
            if run.suffix and (best_val is None or not number(latest_val)):
                for anc, max_step in self.lineage(run)[:-1]:
                    a_idx = self.index(anc)
                    with a_idx.lock:
                        avh = a_idx.phases["validation"]
                        avh.refresh()
                        abv = _best_validation(avh)
                        if abv and (max_step is None or abv["step"] <= max_step):
                            if best_val is None or abv["loss"] < best_val["loss"]:
                                best_val = abv
                        if not number(latest_val) and number(avh.latest.get("loss")):
                            latest_val = avh.latest.get("loss")
            if not number(gen_gap) and number(latest_val) and number(train_mean):
                gen_gap = latest_val - train_mean
            trainability = runtime.get("trainability")
            console_p = console_parameters(run) if not trainability else None
            frozen_pct = None
            if (
                trainability
                and (trainability.get("trainable", 0) + trainability.get("frozen", 0)) > 0
            ):
                tot = trainability["trainable"] + trainability["frozen"]
                frozen_pct = (100.0 * trainability["frozen"]) / tot
            elif console_p and console_p.get("num_total_params"):
                frozen_pct = (
                    100.0
                    * (console_p["num_total_params"] - console_p["num_learnable_params"])
                    / console_p["num_total_params"]
                )
            opt = cfg.get("optimizer") or {}
            sched = cfg.get("scheduler") or {}
            status = item["status"]
            if (
                status != "running"
                and number(latest_step)
                and number(item.get("steps"))
                and latest_step >= item["steps"]
            ):
                status = "completed"
            rows.append(
                {
                    "id": item["id"],
                    "group_id": item["group_id"],
                    "is_latest_in_group": latest_by_group.get(item["group_id"]) == item["id"],
                    "name": item["name"],
                    "attempt": item["attempt"],
                    "ordinal": item["ordinal"],
                    "policy": item["policy"],
                    "status": status,
                    "steps": item["steps"],
                    "latest_step": latest_step,
                    "progress": (latest_step / item["steps"])
                    if number(latest_step) and number(item.get("steps")) and item["steps"] > 0
                    else None,
                    "train_loss_mean": train_mean,
                    "val_loss_latest": latest_val if number(latest_val) else None,
                    "val_loss_best": best_val["loss"] if best_val else None,
                    "val_best_step": best_val["step"] if best_val else None,
                    "generalization_gap": gen_gap if number(gen_gap) else None,
                    "grad_norm_mean": grad_mean,
                    "clip_fraction": clip_frac,
                    "samples": samples,
                    "elapsed_s": elapsed_s,
                    "samples_per_s": sps,
                    "offline_mae": _offline_mae_mean(offline),
                    "frozen_pct": frozen_pct,
                    "hparams": {
                        "lr": opt.get("lr"),
                        "weight_decay": opt.get("weight_decay"),
                        "grad_clip_norm": opt.get("grad_clip_norm"),
                        "optimizer": opt.get("type"),
                        "scheduler": sched.get("type"),
                        "warmup_steps": sched.get("warmup_steps"),
                        "batch_size": cfg.get("batch_size"),
                        "effective_batch_size": effective_batch_from_config(cfg, schema),
                        "mixed_precision": cfg.get("mixed_precision"),
                        "seed": cfg.get("seed"),
                        "git_commit": runtime.get("git_commit"),
                    },
                }
            )
        return {"runs": rows}

    def export_csv(self, run, keys, axis, low, high, *, stitch=False):
        cfg = run.config()
        schema = document(run.schema_path)
        train_segs, _ = self._lineage_histories(run, "train", stitch)
        val_segs, _ = self._lineage_histories(run, "validation", stitch)
        rows = OrderedDict()
        for phase, segs in (("train", train_segs), ("validation", val_segs)):
            for key in keys:
                if key == "derived/generalization_gap":
                    if phase != "validation":
                        continue
                    plotted, _, _ = generalization_gap_series(
                        train_segs, val_segs, axis, low, high, 10**7
                    )
                    pairs = plotted["points"] if plotted else []
                else:
                    pairs = stitched_pairs(segs, key, axis, cfg, schema)
                    if low is not None or high is not None:
                        pairs = [
                            (x, y)
                            for x, y in pairs
                            if (low is None or x >= low) and (high is None or x <= high)
                        ]
                col = f"{phase}:{key}"
                for x, y in pairs:
                    rows.setdefault(x, {})[col] = "" if y is None else y
        cols = []
        for entry in rows.values():
            for col in entry:
                if col not in cols:
                    cols.append(col)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([axis, *cols])
        for x in sorted(rows):
            entry = rows[x]
            xv = int(x) if axis != "elapsed_s" and float(x).is_integer() else x
            writer.writerow([xv, *(entry.get(c, "") for c in cols)])
        return buf.getvalue().encode("utf-8")

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
