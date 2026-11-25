"""Diagnostics from unsampled optimizer records, stage comparisons and validation signals."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from itertools import pairwise
from statistics import fmean, pstdev

from .history import loss_metric, number, quantile


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
        "p90": quantile(values, 0.90),
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
    finite_points = []
    for i in range(len(rows)):
        if math.isfinite(loss.values[i]):
            finite_points.append({"loss": loss.values[i], **at(history, rows[i])})
    recent = list(reversed(finite_points[-2:]))
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
    consecutive_rises = 0
    for a, b in pairwise(finite_points):
        if b["loss"] > a["loss"]:
            consecutive_rises += 1
        else:
            consecutive_rises = 0
    return {
        "best": best,
        "first_hit": hit,
        "count": len(finite_points),
        "latest": recent[0],
        "previous": recent[1] if len(recent) == 2 else None,
        "consecutive_rises": consecutive_rises,
    }


def _stage_summary(history, key, start, stop, label):
    if stop <= start:
        return None
    steps = history.axes["step"]
    pairs = observations(history, key, start, stop)
    if not pairs:
        return None
    vals = sorted(v for _, v in pairs)
    return {
        "stage": label,
        "from_step": int(steps[start]),
        "to_step": int(steps[stop - 1]),
        "records": stop - start,
        "n": len(vals),
        "mean": fmean(vals),
        "p50": quantile(vals, 0.5),
        "p95": quantile(vals, 0.95),
        "sd": pstdev(vals),
    }


def _derived_gap(train, validation):
    vloss = validation.series.get("loss")
    tloss = train.series.get("loss")
    if not vloss or not tloss or not len(train.offsets):
        return None
    tsteps = train.axes["step"]
    vsteps = validation.axes["step"]
    vrows = vloss.row_indices()
    trows = tloss.row_indices()
    points = []
    for i in range(len(vrows)):
        vv = vloss.values[i]
        if not math.isfinite(vv):
            continue
        vstep = int(vsteps[vrows[i]])
        right_row = bisect_right(tsteps, vstep)
        left_row = max(0, right_row - 50)
        lo_i = bisect_left(trows, left_row)
        hi_i = bisect_left(trows, right_row)
        tvals = [tloss.values[j] for j in range(lo_i, hi_i) if math.isfinite(tloss.values[j])]
        if tvals:
            tmean = fmean(tvals)
            points.append({"step": vstep, "val_loss": vv, "train_mean": tmean, "gap": vv - tmean})
    if not points:
        return None
    return {
        "source": "approx_window",
        "latest": points[-1],
        "earliest": points[0],
        "delta": points[-1]["gap"] - points[0]["gap"] if len(points) > 1 else None,
        "count": len(points),
    }


def _build_alerts(train, validation, result, stages, data_wait_ratio):
    alerts = []
    val = result.get("validation") or {}
    # 1. Overfitting signal: validation rising repeatedly & above best while train fell
    if (
        val.get("count", 0) >= 3
        and val.get("latest")
        and val.get("best")
        and val["latest"]["step"] > val["best"]["step"]
        and val["best"]["loss"] > 0
    ):
        ratio = (val["latest"]["loss"] - val["best"]["loss"]) / abs(val["best"]["loss"])
        rises = val.get("consecutive_rises", 0)
        train_dropped = (
            len(stages) == 3
            and stages[0]
            and stages[2]
            and stages[2]["mean"] < stages[0]["mean"] * 0.95
        )
        if ratio >= 0.05 and (rises >= 2 or train_dropped):
            alerts.append(
                {
                    "code": "overfitting",
                    "severity": "warn",
                    "step": int(val["best"]["step"]),
                    "data": {
                        "best_step": int(val["best"]["step"]),
                        "best_loss": val["best"]["loss"],
                        "latest_step": int(val["latest"]["step"]),
                        "latest_loss": val["latest"]["loss"],
                        "rise_pct": ratio * 100.0,
                    },
                }
            )
    # 2. Plateau signal: late third vs mid third barely improves
    if (
        len(train.offsets) >= 300
        and len(stages) == 3
        and stages[1]
        and stages[2]
        and stages[1]["mean"] > 0
    ):
        rel = (stages[1]["mean"] - stages[2]["mean"]) / abs(stages[1]["mean"])
        if -0.015 <= rel <= 0.01:
            alerts.append(
                {
                    "code": "plateau",
                    "severity": "info",
                    "step": stages[2]["from_step"],
                    "data": {
                        "mid_mean": stages[1]["mean"],
                        "late_mean": stages[2]["mean"],
                        "change_pct": -rel * 100.0,
                    },
                }
            )
    # 3. Heavy gradient clipping
    clip = result.get("clipping")
    if clip and clip["n"] >= 10 and clip["fraction"] >= 0.25:
        alerts.append(
            {
                "code": "heavy_clipping",
                "severity": "warn",
                "step": result["to_step"],
                "data": {"fraction_pct": clip["fraction"] * 100.0, "threshold": clip["threshold"]},
            }
        )
    # 4. AMP skipped updates
    if result.get("skipped") and result["skipped"] > 0:
        alerts.append(
            {
                "code": "amp_skips",
                "severity": "warn",
                "step": result["to_step"],
                "data": {"skipped": result["skipped"]},
            }
        )
    # 5. Data loading bottleneck
    if data_wait_ratio is not None and data_wait_ratio >= 0.35:
        alerts.append(
            {
                "code": "data_bottleneck",
                "severity": "info",
                "step": result["to_step"],
                "data": {"wait_pct": data_wait_ratio * 100.0},
            }
        )
    # 6. Non-finite scalars anywhere in train/validation
    nonfinite_count = sum(len(s.gaps) for s in train.series.values()) + sum(
        len(s.gaps) for s in validation.series.values()
    )
    if nonfinite_count > 0:
        alerts.append(
            {
                "code": "nonfinite",
                "severity": "error",
                "step": result["to_step"],
                "data": {"count": nonfinite_count},
            }
        )
    # 7. Module update probe warnings
    for name, m in (result.get("modules") or {}).items():
        if m.get("trainable_tensors") and m.get("grad_tensors") == 0:
            alerts.append(
                {
                    "code": "module_disconnected",
                    "severity": "warn",
                    "step": m.get("step"),
                    "data": {"module": name},
                }
            )
        elif (
            number(m.get("grad_norm"))
            and m["grad_norm"] > 0
            and number(m.get("relative_update"))
            and m["relative_update"] < 1e-6
        ):
            alerts.append(
                {
                    "code": "module_tiny_update",
                    "severity": "info",
                    "step": m.get("step"),
                    "data": {"module": name, "relative_update": m["relative_update"]},
                }
            )
    if (
        not alerts
        and len(stages) == 3
        and stages[0]
        and stages[2]
        and stages[0]["mean"] > 0
        and stages[2]["mean"] < stages[0]["mean"] * 0.95
    ):
        alerts.append(
            {
                "code": "healthy_descent",
                "severity": "ok",
                "step": stages[2]["to_step"],
                "data": {
                    "drop_pct": (stages[0]["mean"] - stages[2]["mean"])
                    / abs(stages[0]["mean"])
                    * 100.0
                },
            }
        )
    return alerts


def analyze(train, validation, config, window, metric, target, *, low=None, high=None, axis="step"):
    total = len(train.offsets)
    steps = train.axes["step"]
    axis_vals = train.axes.get(axis)
    custom_range = low is not None or high is not None
    if custom_range and axis_vals is not None and total:
        start = bisect_left(axis_vals, low) if low is not None else 0
        stop = bisect_right(axis_vals, high) if high is not None else total
        if stop < start:
            stop = start
        span = stop - start
    else:
        stop = total
        start = max(0, stop - window)
        span = window

    result = {
        "metric": metric,
        "window": window,
        "range_mode": "custom" if custom_range else "tail",
        "records": stop - start,
        "total_records": total,
        "from_step": int(steps[start]) if stop > start else None,
        "to_step": int(steps[stop - 1]) if stop > start else None,
        "validation": validation_results(validation, target),
    }

    def recent(key):
        return observations(train, key, start, stop)

    current = recent(metric)
    result["distribution"] = distribution(current, steps)
    previous_start = max(0, start - max(1, span))
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

    # Full-run macro stages (Early / Mid / Late thirds) for macroscopic trend inspection.
    stages = []
    if total >= 6:
        t1 = total // 3
        t2 = (2 * total) // 3
        for s_lo, s_hi, name in ((0, t1, "early"), (t1, t2, "mid"), (t2, total, "late")):
            st = _stage_summary(train, metric, s_lo, s_hi, name)
            if st:
                stages.append(st)
    result["stages"] = stages

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
    if (
        skipped
        and stop > start
        and skipped[-1][0] == stop - 1
        and (not start or skipped[0][0] == start - 1)
    ):
        result["skipped"] = int(skipped[-1][1] - (skipped[0][1] if start else 0))

    elapsed = train.axes.get("elapsed_s")
    samples = train.axes.get("samples")
    result["samples_per_s"] = None
    if elapsed and samples and stop > start and elapsed[stop - 1] > elapsed[start]:
        result["samples_per_s"] = (samples[stop - 1] - samples[start]) / (
            elapsed[stop - 1] - elapsed[start]
        )
    result["timing"] = {}
    for k in ("update_s", "dataloading_s"):
        pairs = recent(k)
        if pairs:
            result["timing"][k] = fmean(v for _, v in pairs)
    data_wait_ratio = None
    if "update_s" in result["timing"] and "dataloading_s" in result["timing"]:
        denom = result["timing"]["update_s"] + result["timing"]["dataloading_s"]
        if denom > 0:
            data_wait_ratio = result["timing"]["dataloading_s"] / denom
    result["data_wait_ratio"] = data_wait_ratio

    total_loss_pairs = recent("loss")
    total_loss_mean = fmean(v for _, v in total_loss_pairs) if total_loss_pairs else None
    components = {}
    for key in train.series:
        if key != "loss" and loss_metric(key):
            pairs = current if key == metric else recent(key)
            if pairs:
                prior_values = (
                    prior if key == metric else observations(train, key, previous_start, start)
                )
                mean_val = fmean(v for _, v in pairs)
                share = (
                    (mean_val / total_loss_mean) * 100.0
                    if total_loss_mean and total_loss_mean > 0 and mean_val >= 0
                    else None
                )
                components[key] = {
                    "mean": mean_val,
                    "n": len(pairs),
                    "previous_mean": fmean(v for _, v in prior_values) if prior_values else None,
                    "previous_n": len(prior_values),
                    "share_pct": share,
                }
    result["components"] = components
    result["objective_mean"] = total_loss_mean

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
    result["derived_gap"] = None if result["paired"] else _derived_gap(train, validation)
    result["modules"] = {}
    for key, series in train.series.items():
        if key.startswith("module/") and key.count("/") == 2:
            _, module, quantity = key.split("/", 2)
            row = series.row_indices()[-1]
            value = series.values[-1]
            result["modules"].setdefault(module, {"step": int(steps[row])})[quantity] = (
                value if math.isfinite(value) else None
            )
    result["alerts"] = _build_alerts(train, validation, result, stages, data_wait_ratio)
    return result
