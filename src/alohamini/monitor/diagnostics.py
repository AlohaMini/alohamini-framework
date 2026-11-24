"""Diagnostics from unsampled optimizer records and observed validation points."""

from __future__ import annotations

import math
from bisect import bisect_left
from statistics import fmean, pstdev

from .history import loss_metric, number


def observations(history, key, start, stop):
    series = history.series.get(key)
    if series is None:
        return []
    rows = series.row_indices()
    return [
        (rows[i], series.values[i])
        for i in range(bisect_left(rows, start), bisect_left(rows, stop))
        if math.isfinite(series.values[i])
    ]


def quantile(ordered, p):
    index = (len(ordered) - 1) * p
    left = int(index)
    return ordered[left] + (index - left) * (ordered[math.ceil(index)] - ordered[left])


def distribution(pairs, steps):
    if not pairs:
        return None
    values = sorted(v for _, v in pairs)
    low, high = values[0], values[-1]
    bins = [0] * (12 if high > low else 1)
    for value in values:
        i = int((value - low) / (high - low) * len(bins)) if high > low else 0
        bins[min(i, len(bins) - 1)] += 1
    mean = fmean(values)
    xmean = fmean(steps[row] for row, _ in pairs)
    xx = sum((steps[row] - xmean) ** 2 for row, _ in pairs)
    xy = sum((steps[row] - xmean) * (v - mean) for row, v in pairs)
    peak = max(pairs, key=lambda pair: pair[1])[0]
    return {
        "n": len(values),
        "mean": mean,
        "sd": pstdev(values),
        "min": low,
        "max": high,
        "peak_step": int(steps[peak]),
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "slope_per_step": xy / xx if xx else None,
        "bins": bins,
    }


def at(history, row):
    return {k: axis[row] for k, axis in history.axes.items()}


def validation_results(history, target):
    loss = history.series.get("loss")
    if not loss:
        return {}
    rows = loss.row_indices()
    index, _ = loss._extrema(0, len(rows))
    if index is None:
        return {}
    best = {"loss": loss.values[index], **at(history, rows[index])}
    recent = []
    for i in range(len(rows) - 1, -1, -1):
        if math.isfinite(loss.values[i]):
            recent.append({"loss": loss.values[i], **at(history, rows[i])})
            if len(recent) == 2:
                break
    hit = None
    if target is not None and best["loss"] <= target:
        # Skip full blocks whose minimum could not reach the target.
        for start in range(0, len(rows), loss.block):
            block = start // loss.block
            if block < len(loss.blocks):
                minimum, _ = loss.blocks[block]
                if minimum is None or loss.values[minimum] > target:
                    continue
            stop = min(start + loss.block, len(rows))
            index = next((i for i in range(start, stop) if loss.values[i] <= target), None)
            if index is not None:
                hit = {"loss": loss.values[index], **at(history, rows[index])}
                break
    return {
        "best": best,
        "first_hit": hit,
        "count": len(rows) - len(loss.gaps),
        "latest": recent[0],
        "previous": recent[1] if len(recent) == 2 else None,
    }


def analyze(train, validation, config, window, metric, target):
    stop = len(train.offsets)
    start = max(0, stop - window)
    steps = train.axes["step"]
    result = {
        "metric": metric,
        "window": window,
        "records": stop - start,
        "from_step": int(steps[start]) if stop else None,
        "to_step": int(steps[-1]) if stop else None,
        "validation": validation_results(validation, target),
    }

    def recent(key):
        return observations(train, key, start, stop)

    current = recent(metric)
    result["distribution"] = distribution(current, steps)
    previous_start = max(0, start - window)
    prior = observations(train, metric, previous_start, start)
    result["previous_window"] = (
        {
            "records": start - previous_start,
            "n": len(prior),
            "from_step": int(steps[previous_start]),
            "to_step": int(steps[start - 1]),
            "mean": fmean(v for _, v in prior) if prior else None,
        }
        if start
        else None
    )
    norm = recent("grad_norm")
    result["gradient"] = distribution(norm, steps)
    threshold = (config.get("optimizer") or {}).get("grad_clip_norm")
    flags = recent("grad_clipped")
    clip = None
    if flags:
        decisions = [v for _, v in flags if v in (0, 1)]
        source = "recorded"
    elif norm and number(threshold) and threshold > 0:
        decisions = [v > threshold for _, v in norm]
        source = "from_norm"
    else:
        decisions = []
        source = None
    if decisions:
        n = int(sum(decisions))
        clip = {
            "count": n,
            "n": len(decisions),
            "fraction": n / len(decisions),
            "source": source,
            "threshold": threshold,
        }
    result["clipping"] = clip
    skipped = observations(train, "skipped_updates", max(0, start - 1), stop)
    result["skipped"] = None
    if skipped and skipped[-1][0] == stop - 1 and (not start or skipped[0][0] == start - 1):
        result["skipped"] = int(skipped[-1][1] - (skipped[0][1] if start else 0))

    elapsed = train.axes.get("elapsed_s")
    samples = train.axes.get("samples")
    result["samples_per_s"] = None
    if elapsed and samples and elapsed[-1] > elapsed[start]:
        result["samples_per_s"] = (samples[-1] - samples[start]) / (elapsed[-1] - elapsed[start])
    result["timing"] = {}
    for k in ("update_s", "dataloading_s"):
        pairs = recent(k)
        if pairs:
            result["timing"][k] = fmean(v for _, v in pairs)
    components = {}
    for key in train.series:
        if key != "loss" and loss_metric(key):
            pairs = current if key == metric else recent(key)
            if pairs:
                prior_values = (
                    prior if key == metric else observations(train, key, previous_start, start)
                )
                components[key] = {
                    "mean": fmean(v for _, v in pairs),
                    "n": len(pairs),
                    "previous_mean": fmean(v for _, v in prior_values) if prior_values else None,
                    "previous_n": len(prior_values),
                }
    result["components"] = components
    result["rates"] = {}
    for key in train.series:
        if key == "lr" or key.startswith("lr/"):
            pairs = recent(key)
            if pairs:
                result["rates"][key] = {"first": pairs[0][1], "last": pairs[-1][1]}
    latest = validation.latest
    result["paired"] = (
        {
            key: latest[key]
            for key in (
                "step",
                "weights",
                "train_eval_loss",
                "generalization_gap",
                "train_eval_samples",
                "loss",
            )
            if key in latest
        }
        if "train_eval_loss" in latest
        else None
    )
    result["modules"] = {}
    for key, series in train.series.items():
        if key.startswith("module/") and key.count("/") == 2:
            _, module, quantity = key.split("/", 2)
            row = series.row_indices()[-1]
            value = series.values[-1]
            result["modules"].setdefault(module, {"step": int(steps[row])})[quantity] = (
                value if math.isfinite(value) else None
            )
    return result
