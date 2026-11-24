"""Incremental JSONL indexing and extrema-preserving plots of recorded scalars."""

from __future__ import annotations

import json
import math
import os
import uuid
from array import array
from bisect import bisect_left, bisect_right
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

    def plot(self, axis, first_row, stop_row, budget):
        rows = (
            self.rows
            if self.rows is not None
            else range(self.first_row, self.first_row + len(self.values))
        )
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
        while self.offset < stat.st_size:
            line = stream.readline(stat.st_size - self.offset)
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
                self._append(record)
            except (TypeError, ValueError, OverflowError, RecursionError):
                self.invalid_lines += 1
            self.offset += len(line)
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
        metrics = {}
        for key, value in record.items():
            if key in (*AXES, "event_time_s", "optimizer_step"):
                continue
            if key in ("metrics", "lr_groups") and isinstance(value, dict):
                prefix = "lr/" if key == "lr_groups" else ""
                metrics.update({prefix + k: v for k, v in value.items()})
            elif not isinstance(value, (dict, list)):
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
        self.offsets.append(self.offset)
        self.latest = record

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

    def summary(self):
        return {
            "records": len(self.offsets),
            "invalid_lines": self.invalid_lines,
            "nonfinite": {k: len(s.gaps) for k, s in self.series.items() if s.gaps},
            "axes": list(self.axes) if self.offsets else [],
            "metrics": list(self.series),
            "latest": json_safe(self.latest),
        }

    def chart(self, keys, axis, low, high, budget, cursor):
        values = self.axes.get(axis)
        first = self.position(cursor)
        append = first is not None and len(self.offsets) - first <= min(budget // 2, 256)
        result = {
            "mode": "append" if append else "replace",
            "cursor": self.cursor,
            "extent": [values[0], values[-1]] if values else None,
            "series": {},
        }
        if values:
            start = bisect_left(values, low) if low is not None else 0
            stop = bisect_right(values, high) if high is not None else len(values)
            if append:
                start = max(start, first)
            for key in keys:
                if key in self.series:
                    result["series"][key] = self.series[key].plot(
                        values, min(start, stop), stop, budget
                    )
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
