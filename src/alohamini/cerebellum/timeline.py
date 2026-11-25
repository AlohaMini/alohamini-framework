"""Reconstruct the observed timeline without inferring runtime cache contents."""

from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Wheel:
    stamp_ns: int
    vx: float
    vy: float
    omega: float


def clock_kind(stamp):
    if type(stamp) is not int or stamp <= 0:
        return "invalid"
    if 946684800_000000000 <= stamp < 4102444800_000000000:
        return "unix_like"
    return "other_epoch"


def load_run(evidence: Path, run: str):
    candidates = [
        evidence / "dual-camera" / run / "raw",
        evidence / f"lightnav-dual-drive-{run}",
        evidence / run,
        evidence,
    ]
    raw = next((p for p in candidates if (p / "pairs.jsonl").is_file()), None)
    if raw is None:
        raise FileNotFoundError(f"pairs.jsonl for {run} under {evidence}")
    events = [json.loads(line) for line in (raw / "pairs.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    with (raw / "wheel-odom.csv").open(encoding="utf-8-sig", newline="") as handle:
        wheels = [Wheel(int(r["stamp_ns"]), float(r["vx_m_s"]), float(r["vy_m_s"]),
                        float(r["w_rad_s"])) for r in csv.DictReader(handle)]
    return events, wheels


def audit(events, wheels, *, camera_ttl_s, feedback_age_s):
    """Previous wheel = latest header stamp <= decision. Nearest can be later."""
    stamps = [w.stamp_ns for w in wheels]
    issues = []
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        issues.append("wheel_nonmonotonic")
    if any(not all(map(math.isfinite, (w.vx, w.vy, w.omega))) for w in wheels):
        issues.append("wheel_nonfinite")
    kinds = {clock_kind(t) for t in stamps}
    numbered = [row for row in events if "seq" in row]
    result = []
    for idx, original in enumerate(numbered):
        now = original.get("decision_ns")
        kind = clock_kind(now)
        comparable = kind == "unix_like" and kinds == {"unix_like"} and not issues
        previous = nearest = None
        if comparable:
            pos = bisect_right(stamps, now)
            previous = stamps[pos - 1] if pos else None
            options = stamps[max(0, pos - 1):pos + 1]
            nearest = min(options, key=lambda t: abs(t - now)) if options else None
        age = (now - previous) / 1e6 if previous is not None else None
        nearest_delta = (now - nearest) / 1e6 if nearest is not None else None
        if nearest_delta is not None and abs(nearest_delta) > 60_000:
            status = "epochs_or_clock_offset_suspect"
        elif not stamps:
            status = "wheel_log_empty"
        elif not comparable:
            status = "timestamps_untrusted"
        elif age is None:
            status = "no_previous_sample_logged"
        else:
            status = "previous_sample_within_limit" if age <= feedback_age_s * 1000 else "previous_sample_over_limit"
        front, chest = original.get("forward_ns"), original.get("chest_ns")
        reference_age = (now - front) / 1e6 if kind == clock_kind(front) == "unix_like" and now >= front else None
        pair_age = (now - min(front, chest)) / 1e6 if (
            kind == clock_kind(front) == clock_kind(chest) == "unix_like" and now >= max(front, chest)
        ) else None
        following = numbered[idx + 1].get("decision_ns") if idx + 1 < len(numbered) else None
        interval = (following - now) / 1e6 if kind == clock_kind(following) == "unix_like" and following >= now else None
        remaining = camera_ttl_s * 1000 - pair_age if pair_age is not None else None
        moving = any(original.get(field, 0) not in (0, None) for field in ("mpc_v", "mpc_w"))
        result.append({
            "seq": original["seq"], "decision_ns": now,
            "mode": original.get("mode"), "reason": original.get("reason"),
            "wheel_previous_stamp_ns": previous, "wheel_previous_age_ms": age,
            "wheel_nearest_stamp_ns": nearest, "wheel_nearest_signed_age_ms": nearest_delta,
            "wheel_log_status": status, "wheel_consumed_stamp_ns": original.get("wheel_consumed_stamp_ns"),
            "wheel_cache_age_at_lookup_ms": original.get("wheel_age_at_lookup_ms"),
            "reference_age_ms": reference_age, "oldest_camera_age_ms": pair_age,
            "old_command_remaining_ms": remaining, "next_decision_interval_ms": interval,
            "movement_logged": moving,
            "expired_before_next": moving and remaining is not None and interval is not None and remaining < interval,
        })
    gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:]) if b > a]
    stale = [r for r in result if r["reason"] == "wheel_feedback_stale"]
    moves = [r for r in result if r["movement_logged"]]
    remaining = [r["old_command_remaining_ms"] for r in moves if r["old_command_remaining_ms"] is not None]
    intervals = [r["next_decision_interval_ms"] for r in moves if r["next_decision_interval_ms"] is not None]
    summary = {
        "numbered_decisions": len(numbered), "other_events": len(events) - len(numbered),
        "modes": dict(Counter(r["mode"] for r in result)),
        "reasons": dict(Counter(r["reason"] for r in result)),
        "wheel_samples": len(wheels), "wheel_clock_kinds": sorted(kinds), "wheel_issues": issues,
        "wheel_median_interval_ms": median(gaps) if gaps else None,
        "wheel_max_interval_ms": max(gaps, default=None),
        "logged_feedback_status_during_stale": dict(Counter(r["wheel_log_status"] for r in stale)),
        "stale_consumed_stamp_logged": sum(r["wheel_consumed_stamp_ns"] is not None for r in stale),
        "moving_with_decision_interval": len(intervals),
        "movement_remaining_median_ms": median(remaining) if remaining else None,
        "next_decision_after_motion_median_ms": median(intervals) if intervals else None,
        "movement_expired_before_next": sum(r["expired_before_next"] for r in result),
    }
    return result, summary
