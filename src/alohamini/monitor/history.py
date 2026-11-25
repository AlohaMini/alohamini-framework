"""Incremental JSONL indexing, derived curves, smoothing and extrema-preserving plots."""

from __future__ import annotations

import json
import math
import os
import uuid
from array import array
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path


def number(value):
    if type(value) is int:
        return value.bit_length() < 1024
    return type(value) is float and math.isfinite(value)


def json_safe(value):
    """JSON responses represent a non-finite input as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def quantile(ordered, p):
    if not ordered:
        return None
    index = (len(ordered) - 1) * p
    left = int(index)
    return ordered[left] + (index - left) * (ordered[math.ceil(index)] - ordered[left])


class Series:
    """Sparse record indices, packed doubles and extrema for complete 64-point blocks."""

    block = 64

    def __init__(self):
        self.rows = None
        self.first_row = 0
        self.values = array("d")
        self.gaps = array("Q")
        self.blocks = []

    def append(self, row, value):
        index = len(self.values)
        if not index:
            self.first_row = row
        # Dense metrics share the stream's row numbering. Allocate an index only
        # when this metric skips a row.
        if self.rows is None and self.first_row + index != row:
            self.rows = array("Q", range(self.first_row, self.first_row + index))
        if self.rows is not None:
            self.rows.append(row)
        if not number(value):
            value = math.nan
            self.gaps.append(index)
        self.values.append(value)
        if (index + 1) % self.block == 0:
            self.blocks.append(self._scan(index + 1 - self.block, index + 1))

    def _scan(self, start, stop):
        low = high = None
        values = self.values
        for index in range(start, stop):
            v = values[index]
            if math.isfinite(v):
                if low is None or v < values[low]:
                    low = index
                if high is None or v > values[high]:
                    high = index
        return low, high

    def _extrema(self, start, stop):
        block = self.block
        first, last = (start + block - 1) // block, stop // block
        if first >= last:
            return self._scan(start, stop)
        edges = (self._scan(start, first * block), self._scan(last * block, stop))
        pairs = [*edges, *self.blocks[first:last]]
        lows = (i for i, _ in pairs if i is not None)
        highs = (i for _, i in pairs if i is not None)
        return (
            min(lows, key=self.values.__getitem__, default=None),
            max(highs, key=self.values.__getitem__, default=None),
        )

    def row_indices(self):
        return (
            self.rows
            if self.rows is not None
            else range(self.first_row, self.first_row + len(self.values))
        )

    def robust_bounds(self, first_row, stop_row, low_p=0.01, high_p=0.98):
        """Exact raw finite percentiles inside the selected row interval."""
        rows = self.row_indices()
        start, stop = bisect_left(rows, first_row), bisect_left(rows, stop_row)
        vals = [self.values[i] for i in range(start, stop) if math.isfinite(self.values[i])]
        if len(vals) < 6:
            return None
        vals.sort()
        return [quantile(vals, low_p), quantile(vals, high_p)]

    def plot(self, axis, first_row, stop_row, budget):
        rows = self.row_indices()
        start, stop = bisect_left(rows, first_row), bisect_left(rows, stop_row)
        count = stop - start
        if count <= budget:
            indices = range(start, stop)
        else:
            # Up to two extrema per bucket and one gap between adjacent selected points.
            buckets = max(1, (budget - 3) // 4) if self.gaps else (budget - 2) // 2
            kept = {start, stop - 1}
            for i in range(buckets):
                left = start + i * count // buckets
                right = start + (i + 1) * count // buckets
                kept.update(k for k in self._extrema(left, right) if k is not None)
            indices = sorted(kept)
            if self.gaps:
                for a, b in pairwise(indices):
                    gap = bisect_right(self.gaps, a)
                    if gap < len(self.gaps) and self.gaps[gap] < b:
                        kept.add(self.gaps[gap])
                indices = sorted(kept)
        points = [[axis[rows[i]], json_safe(self.values[i])] for i in indices]
        return {"count": count, "points": points}


AXES = ("step", "samples", "elapsed_s")

DERIVED_METRICS = (
    "derived/step_s",
    "derived/data_wait_ratio",
    "derived/samples_per_s",
    "derived/clip_fraction",
)


def loss_metric(key):
    return (
        key in {"loss", "policy/loss"}
        or key.startswith(("loss_", "losses_after_"))
        or key.endswith("_loss")
    )


def effective_batch_from_config(config, schema=None):
    runtime = (schema or {}).get("runtime") or {}
    if number(runtime.get("effective_batch_size")) and runtime["effective_batch_size"] > 0:
        return float(runtime["effective_batch_size"])
    if not isinstance(config, dict):
        return None
    batch = config.get("batch_size")
    if not number(batch) or batch <= 0:
        return None
    accum = config.get("gradient_accumulation_steps") or 1
    world = config.get("world_size") or config.get("num_processes") or 1
    if not number(accum) or accum <= 0:
        accum = 1
    if not number(world) or world <= 0:
        world = 1
    return float(batch * accum * world)


def _compute_ema_points(pairs, alpha):
    """Debiased exponential moving average (TensorBoard/W&B style) over (x, y) pairs."""
    out = []
    acc = 0.0
    weight = 0.0
    for x, y in pairs:
        if y is None or not math.isfinite(y):
            out.append([x, None])
            continue
        acc = alpha * acc + (1.0 - alpha) * y
        weight = alpha * weight + (1.0 - alpha)
        out.append([x, acc / weight if weight else y])
    return out


def _plot_pairs(pairs, low, high, budget):
    """Range-filter and extrema-preserving downsample arbitrary (x, y) point lists."""
    if low is not None or high is not None:
        xs = [p[0] for p in pairs]
        start = bisect_left(xs, low) if low is not None else 0
        stop = bisect_right(xs, high) if high is not None else len(pairs)
        pairs = pairs[start:stop]
    if not pairs:
        return {"count": 0, "points": []}, None
    series = Series()
    axis = array("d")
    for idx, (x, y) in enumerate(pairs):
        axis.append(float(x))
        series.append(idx, y)
    return series.plot(axis, 0, len(pairs), budget), series.robust_bounds(0, len(pairs))


class History:
    """Index a single attempt and phase. The caller serializes access.

    Cursors count accepted records, including repeated optimizer steps. Replacement,
    truncation and cache eviction start a new generation. Partial lines wait for LF.
    """

    def __init__(self, path: Path):
        self.path = path
        self._clear()

    def _clear(self):
        self.axes = {key: array("d") for key in AXES}
        self.series = {}
        self.policy_metrics = {}
        self.means = OrderedDict()
        self.emas = OrderedDict()
        self.derived = OrderedDict()
        self.offsets = array("Q")
        self.latest = {}
        self.invalid_lines = 0
        self.offset = 0
        self.anchor = b""
        self.signature = None
        self.generation = uuid.uuid4().hex

    def _open(self):
        if self.path.is_symlink():
            raise ValueError("Invalid metric file")
        try:
            return self.path.open("rb")
        except FileNotFoundError:
            return nullcontext()

    def refresh(self):
        with self._open() as stream:
            self._index(stream)

    def _index(self, stream):
        if stream is None:
            if self.signature is not None:
                self._clear()
            return
        stat = os.fstat(stream.fileno())
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature == self.signature:
            return
        if self.signature:
            stream.seek(self.offset - len(self.anchor))
            if (
                signature[:2] != self.signature[:2]
                or stat.st_size <= self.signature[2]
                or stream.read(len(self.anchor)) != self.anchor
            ):
                self._clear()
        stream.seek(self.offset)
        appended = False
        while self.offset < stat.st_size:
            line = stream.readline(stat.st_size - self.offset)
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
                self._append(record)
                appended = True
            except (TypeError, ValueError, OverflowError, RecursionError):
                self.invalid_lines += 1
            self.offset += len(line)
        if appended:
            self.derived.clear()
        stream.seek(max(0, self.offset - 64))
        self.anchor = stream.read(min(self.offset, 64))
        self.signature = signature

    def _append(self, record):
        if not isinstance(record, dict):
            raise ValueError("Expected an object")
        step = record.get("step")
        if type(step) is not int or not 0 <= step <= 2**53 - 1:
            raise ValueError("Invalid optimizer step")
        if self.offsets and step < self.axes["step"][-1]:
            raise ValueError("Steps decreased within one attempt")
        nested = record.get("metrics")
        metrics = dict(nested) if isinstance(nested, dict) else {}
        legacy = record.get("policy_metrics")
        # Older trainers stored the step summary and the policy output in two
        # dictionaries. Preserve both when a name (especially "loss") collides.
        policy_keys = []
        if isinstance(legacy, dict):
            for key, value in legacy.items():
                name = "policy/" + key if key in metrics or key in record else key
                metrics[name] = value
                policy_keys.append(name)
        else:
            policy_keys = list(metrics)
        group_rates = record.get("lr_groups")
        if isinstance(group_rates, dict):
            metrics.update({"lr/" + key: value for key, value in group_rates.items()})
        for key, value in record.items():
            if key not in (*AXES, "event_time_s", "optimizer_step") and not isinstance(
                value, (list, dict)
            ):
                metrics[key] = value
        row = len(self.offsets)
        for key in list(self.axes):
            value = record.get(key)
            axis = self.axes[key]
            if not number(value) or (row and value < axis[-1]):
                del self.axes[key]
            else:
                axis.append(value)
        for key, value in metrics.items():
            if isinstance(value, str) and value.lower() in {
                "nan",
                "inf",
                "-inf",
                "infinity",
                "-infinity",
            }:
                value = None
            if value is None or type(value) in (int, float):
                if key not in self.series:
                    self.series[key] = Series()
                self.series[key].append(row, value)
        for key in policy_keys:
            if key in self.series:
                self.policy_metrics[key] = None
        self.offsets.append(self.offset)
        self.latest = record.copy()
        if isinstance(legacy, dict) and isinstance(nested, dict):
            for key in ("loss", "grad_norm", "lr", "gpu_mem_gb"):
                if key in nested:
                    self.latest.setdefault(key, nested[key])

    @property
    def cursor(self):
        return f"{self.generation}.{len(self.offsets)}"

    def position(self, cursor):
        prefix, dot, index = (cursor or "").partition(".")
        if prefix == self.generation and dot and index.isascii() and index.isdigit():
            value = int(index)
            if value <= len(self.offsets):
                return value
        return None

    def available_derived(self, config=None, schema=None):
        names = []
        has_timing = "update_s" in self.series and "dataloading_s" in self.series
        if "step_s" in self.series or has_timing:
            names.append("derived/step_s")
        if has_timing:
            names.append("derived/data_wait_ratio")
        if "samples_per_s" in self.series or (
            ("samples" in self.axes or effective_batch_from_config(config, schema))
            and ("step_s" in self.series or has_timing)
        ):
            names.append("derived/samples_per_s")
        threshold = ((config or {}).get("optimizer") or {}).get("grad_clip_norm")
        if "grad_clipped" in self.series or (
            "grad_norm" in self.series and number(threshold) and threshold > 0
        ):
            names.append("derived/clip_fraction")
        return names

    def _row_map(self, key):
        s = self.series.get(key)
        if s is None:
            return None
        rows = s.row_indices()
        if s.rows is None and s.first_row == 0 and len(s.values) == len(self.offsets):
            return s.values
        return dict(zip(rows, s.values, strict=True))

    def derived_series(self, key, config=None, schema=None):
        threshold = ((config or {}).get("optimizer") or {}).get("grad_clip_norm")
        eff_batch = effective_batch_from_config(config, schema)
        cache_key = (key, threshold, eff_batch)
        if cache_key in self.derived:
            self.derived.move_to_end(cache_key)
            return self.derived[cache_key]
        result = None
        n = len(self.offsets)
        if key == "derived/step_s":
            if "step_s" in self.series:
                result = self.series["step_s"]
            elif "update_s" in self.series and "dataloading_s" in self.series:
                u, d = self._row_map("update_s"), self._row_map("dataloading_s")
                s = Series()
                for row in range(n):
                    uv = u[row] if isinstance(u, array) else u.get(row)
                    dv = d[row] if isinstance(d, array) else d.get(row)
                    if (
                        uv is not None
                        and dv is not None
                        and math.isfinite(uv)
                        and math.isfinite(dv)
                    ):
                        s.append(row, uv + dv)
                result = s
        elif key == "derived/data_wait_ratio":
            if "update_s" in self.series and "dataloading_s" in self.series:
                u, d = self._row_map("update_s"), self._row_map("dataloading_s")
                s = Series()
                for row in range(n):
                    uv = u[row] if isinstance(u, array) else u.get(row)
                    dv = d[row] if isinstance(d, array) else d.get(row)
                    if (
                        uv is not None
                        and dv is not None
                        and math.isfinite(uv)
                        and math.isfinite(dv)
                    ):
                        total = uv + dv
                        if total > 0:
                            s.append(row, dv / total)
                result = s
        elif key == "derived/samples_per_s":
            if "samples_per_s" in self.series:
                result = self.series["samples_per_s"]
            else:
                step_s = self.derived_series("derived/step_s", config, schema)
                if step_s is not None:
                    samples = self.axes.get("samples")
                    rows = step_s.row_indices()
                    s = Series()
                    for idx, row in enumerate(rows):
                        dt = step_s.values[idx]
                        if not math.isfinite(dt) or dt <= 0:
                            continue
                        if samples is not None:
                            delta = (
                                samples[row] - samples[row - 1]
                                if row > 0
                                else (eff_batch or samples[row])
                            )
                            if delta > 0:
                                s.append(row, delta / dt)
                        elif eff_batch:
                            s.append(row, eff_batch / dt)
                    result = s
        elif key == "derived/clip_fraction":
            window = 50
            if "grad_clipped" in self.series:
                src = self.series["grad_clipped"]
                rows = src.row_indices()
                flags = [
                    (rows[i], 1.0 if src.values[i] == 1 else 0.0)
                    for i in range(len(rows))
                    if src.values[i] in (0, 1)
                ]
            elif "grad_norm" in self.series and number(threshold) and threshold > 0:
                src = self.series["grad_norm"]
                rows = src.row_indices()
                flags = [
                    (rows[i], 1.0 if src.values[i] > threshold else 0.0)
                    for i in range(len(rows))
                    if math.isfinite(src.values[i])
                ]
            else:
                flags = []
            if flags:
                s = Series()
                running = 0.0
                for idx, (row, flag) in enumerate(flags):
                    running += flag
                    if idx >= window:
                        running -= flags[idx - window][1]
                    denom = min(idx + 1, window)
                    s.append(row, running / denom)
                result = s
        if result is not None:
            if len(self.derived) >= 32:
                self.derived.popitem(last=False)
            self.derived[cache_key] = result
        return result

    def resolve_series(self, key, config=None, schema=None):
        if key in self.series:
            return self.series[key]
        if key.startswith("derived/"):
            return self.derived_series(key, config, schema)
        return None

    def tail_mean(self, key, width=100):
        series = self.series.get(key)
        if series is None or not series.values:
            return None
        stop = len(series.values)
        start = max(0, stop - width)
        vals = [series.values[i] for i in range(start, stop) if math.isfinite(series.values[i])]
        return math.fsum(vals) / len(vals) if vals else None

    def summary(self):
        return {
            "records": len(self.offsets),
            "invalid_lines": self.invalid_lines,
            "nonfinite": {k: len(s.gaps) for k, s in self.series.items() if s.gaps},
            "axes": list(self.axes) if self.offsets else [],
            "metrics": list(self.series),
            "policy_metrics": list(self.policy_metrics),
            "latest": json_safe(self.latest),
        }

    def window_means(self, key, width):
        """Completed non-overlapping windows from raw records, located at the last row."""
        cache_key = (key, width)
        if cache_key not in self.means:
            if len(self.means) >= 64:
                self.means.popitem(last=False)
            self.means[cache_key] = Series()
        self.means.move_to_end(cache_key)
        result = self.means[cache_key]
        series = self.series[key]
        rows = series.row_indices()
        for start in range(len(result.values) * width, len(self.offsets) - width + 1, width):
            stop = start + width
            left, right = bisect_left(rows, start), bisect_left(rows, stop)
            value = math.fsum(series.values[left:right]) / width if right - left == width else None
            result.append(stop - 1, value)
        return result

    def ema_series(self, key, alpha, config=None, schema=None):
        """Debiased EMA computed incrementally over raw observations of any scalar series."""
        alpha = round(float(alpha), 3)
        series = self.resolve_series(key, config, schema)
        if series is None:
            return None
        cache_key = (key, alpha, id(series))
        entry = self.emas.get(cache_key)
        if entry is None:
            if len(self.emas) >= 96:
                self.emas.popitem(last=False)
            entry = [Series(), 0.0, 0.0]
            self.emas[cache_key] = entry
        else:
            self.emas.move_to_end(cache_key)
        result, acc, weight = entry
        rows = series.row_indices()
        done = len(result.values)
        for index in range(done, len(series.values)):
            v = series.values[index]
            row = rows[index]
            if not math.isfinite(v):
                result.append(row, None)
            else:
                acc = alpha * acc + (1.0 - alpha) * v
                weight = alpha * weight + (1.0 - alpha)
                result.append(row, acc / weight if weight else v)
        entry[1], entry[2] = acc, weight
        return result

    def chart(
        self,
        keys,
        axis,
        low,
        high,
        budget,
        cursor,
        mean_window=0,
        *,
        smooth=0.0,
        config=None,
        schema=None,
    ):
        values = self.axes.get(axis)
        first = self.position(cursor)
        has_derived = any(k.startswith("derived/") for k in keys)
        append = (
            first is not None
            and not has_derived
            and len(self.offsets) - first <= min(budget // 2, 256)
        )
        result = {
            "mode": "append" if append else "replace",
            "cursor": self.cursor,
            "extent": [values[0], values[-1]] if values else None,
            "series": {},
        }
        if mean_window:
            result["means"] = {}
        if smooth > 0:
            result["smoothed"] = {}
        bounds = {}
        if values:
            start = bisect_left(values, low) if low is not None else 0
            stop = bisect_right(values, high) if high is not None else len(values)
            if append:
                start = max(start, first)
            for key in keys:
                series = self.resolve_series(key, config, schema)
                if series is not None:
                    result["series"][key] = series.plot(values, min(start, stop), stop, budget)
                    if not append:
                        rb = series.robust_bounds(min(start, stop), stop)
                        if rb is not None:
                            bounds[key] = rb
                    if mean_window and key in self.series and loss_metric(key):
                        result["means"][key] = (
                            result["series"][key]
                            if mean_window == 1
                            else self.window_means(key, mean_window).plot(
                                values, min(start, stop), stop, budget
                            )
                        )
                    if smooth > 0:
                        ema = self.ema_series(key, smooth, config, schema)
                        if ema is not None:
                            result["smoothed"][key] = ema.plot(
                                values, min(start, stop), stop, budget
                            )
        if bounds:
            result["bounds"] = bounds
        return result

    def records(self, cursor, limit, tail):
        with self._open() as stream:
            # A writer may rotate the path while this page is being read.
            # Read the indexed bytes from the same open file description.
            self._index(stream)
            start = self.position(cursor) if cursor else 0
            reset = start is None
            if reset or (tail and not cursor):
                start = max(0, len(self.offsets) - limit) if tail else 0
            stop = min(len(self.offsets), start + limit)
            rows = []
            for offset in self.offsets[start:stop]:
                stream.seek(offset)
                rows.append(json_safe(json.loads(stream.readline())))
        return {
            "records": rows,
            "cursor": f"{self.generation}.{stop}",
            "more": stop < len(self.offsets),
            "reset": reset,
            "invalid_lines": self.invalid_lines,
        }


def stitched_pairs(segments, key, axis, config=None, schema=None):
    """Collect raw (axis_value, scalar) pairs across stitched (history, max_step) segments."""
    pairs = []
    elapsed_offset = 0.0
    for idx, (history, max_step) in enumerate(segments):
        axis_vals = history.axes.get(axis)
        steps = history.axes.get("step")
        if axis_vals is None or not len(axis_vals):
            continue
        series = history.resolve_series(key, config, schema)
        if series is None:
            if axis == "elapsed_s" and len(axis_vals):
                limit_row = bisect_right(steps, max_step) if max_step is not None else len(steps)
                if limit_row > 0:
                    elapsed_offset += axis_vals[limit_row - 1]
            continue
        rows = series.row_indices()
        limit_row = bisect_right(steps, max_step) if max_step is not None else len(steps)
        stop = bisect_left(rows, limit_row)
        start = 0
        if idx > 0 and pairs and axis == "step":
            prev_x = pairs[-1][0]
            while start < stop and axis_vals[rows[start]] <= prev_x:
                start += 1
        shift = elapsed_offset if axis == "elapsed_s" else 0.0
        for i in range(start, stop):
            row = rows[i]
            val = series.values[i]
            pairs.append((axis_vals[row] + shift, val if math.isfinite(val) else None))
        if axis == "elapsed_s" and limit_row > 0:
            elapsed_offset += axis_vals[limit_row - 1]
    return pairs


def stitched_extent(segments, axis):
    first = None
    last = None
    elapsed_offset = 0.0
    for history, max_step in segments:
        axis_vals = history.axes.get(axis)
        steps = history.axes.get("step")
        if axis_vals is None or not len(axis_vals):
            continue
        limit_row = bisect_right(steps, max_step) if max_step is not None else len(steps)
        if limit_row <= 0:
            continue
        shift = elapsed_offset if axis == "elapsed_s" else 0.0
        lo, hi = axis_vals[0] + shift, axis_vals[limit_row - 1] + shift
        if first is None:
            first = lo
        last = hi
        if axis == "elapsed_s":
            elapsed_offset += axis_vals[limit_row - 1]
    return [first, last] if first is not None and last is not None else None


def stitched_chart(
    segments,
    keys,
    axis,
    low,
    high,
    budget,
    mean_window=0,
    *,
    smooth=0.0,
    config=None,
    schema=None,
):
    extent = stitched_extent(segments, axis)
    result = {
        "mode": "replace",
        "cursor": "stitched:" + ":".join(h.cursor for h, _ in segments),
        "extent": extent,
        "series": {},
    }
    if mean_window:
        result["means"] = {}
    if smooth > 0:
        result["smoothed"] = {}
    bounds = {}
    if extent is not None:
        for key in keys:
            pairs = stitched_pairs(segments, key, axis, config, schema)
            if not pairs:
                continue
            plotted, rb = _plot_pairs(pairs, low, high, budget)
            result["series"][key] = plotted
            if rb is not None:
                bounds[key] = rb
            if mean_window and loss_metric(key):
                if mean_window == 1:
                    result["means"][key] = plotted
                else:
                    win_pairs = []
                    for i in range(0, len(pairs) - mean_window + 1, mean_window):
                        chunk = pairs[i : i + mean_window]
                        vals = [v for _, v in chunk if v is not None]
                        win_pairs.append(
                            (
                                chunk[-1][0],
                                math.fsum(vals) / mean_window if len(vals) == mean_window else None,
                            )
                        )
                    result["means"][key], _ = _plot_pairs(win_pairs, low, high, budget)
            if smooth > 0:
                ema_pairs = _compute_ema_points(pairs, smooth)
                result["smoothed"][key], _ = _plot_pairs(ema_pairs, low, high, budget)
    if bounds:
        result["bounds"] = bounds
    return result


def generalization_gap_series(train_segments, val_segments, axis, low, high, budget, window=50):
    """Return paired generalization_gap if recorded, else val_loss minus nearby training mean."""
    paired = stitched_pairs(val_segments, "generalization_gap", axis)
    if paired:
        plotted, rb = _plot_pairs(paired, low, high, budget)
        return plotted, rb, "paired"
    val_loss = stitched_pairs(val_segments, "loss", axis)
    train_loss = stitched_pairs(train_segments, "loss", axis)
    if not val_loss or not train_loss:
        return None, None, None
    train_xs = [p[0] for p in train_loss]
    gap_pairs = []
    for vx, vy in val_loss:
        if vy is None:
            gap_pairs.append((vx, None))
            continue
        right = bisect_right(train_xs, vx)
        left = max(0, right - window)
        vals = [train_loss[i][1] for i in range(left, right) if train_loss[i][1] is not None]
        if vals:
            gap_pairs.append((vx, vy - math.fsum(vals) / len(vals)))
    if not gap_pairs:
        return None, None, None
    plotted, rb = _plot_pairs(gap_pairs, low, high, budget)
    return plotted, rb, "approx_window"
