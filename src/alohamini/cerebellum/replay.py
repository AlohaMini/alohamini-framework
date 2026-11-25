"""Command-line audit and deterministic, non-actuating replay."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from alohamini.schema import BodyVelocity

from .controller import Controller
from .obstacles import Grid
from .planner import components
from .scenarios import synthetic_grid
from .scheduler import Periodic
from .state import Feedback, load_config, Pose, Reference, ZERO
from .timeline import audit, clock_kind, load_run


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


FAULTS = {"wheel_stale", "reference_stale", "unknown", "stop"}


def fault_arg(value):
    try:
        fault, seconds = value.split("@", maxsplit=1)
        seconds = float(seconds)
        if fault in FAULTS and math.isfinite(seconds) and seconds >= 0:
            return fault, seconds
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected {wheel_stale,reference_stale,unknown,stop}@seconds")


def simulate(events, wheels, config, *, scene="recorded", injections=()):
    """Use recorded wheel headers as simulated receipt, not as cache proof."""
    decisions = [r for r in events if "seq" in r and clock_kind(r.get("decision_ns")) == "unix_like"]
    if not decisions:
        return [], {"status": "no_trustworthy_decision_clock"}
    decisions.sort(key=lambda r: r["decision_ns"])
    first = decisions[0]["decision_ns"]
    end = (decisions[-1]["decision_ns"] - first) / 1e9 + config["control"]["reference_ttl_s"]
    wheel = [w for w in wheels if clock_kind(w.stamp_ns) == "unix_like"
             and abs(w.stamp_ns - first) < (end + 1.) * 1e9]
    # Keep input order so a regressing wheel stamp fails the same validation
    # used by the live adapter.
    core = Controller(config)
    loop = Periodic(config["control"]["frequency_hz"])
    info = []
    decision_index = wheel_index = 0
    map_kind = None
    grid = None
    for i in range(int(end / loop.period) + 2):
        t = i * loop.period
        active = {kind for kind, since in injections if t >= since}
        while wheel_index < len(wheel) and wheel[wheel_index].stamp_ns - first <= round(t * 1e9):
            w = wheel[wheel_index]
            wheel_index += 1
            if "wheel_stale" in active:
                continue
            stamp = (w.stamp_ns - first) / 1e9
            try:
                core.feedback(Feedback(stamp, stamp, BodyVelocity(w.vx, w.vy, w.omega),
                                       source_stamp_ns=w.stamp_ns))
            except ValueError:
                core.odom.fault = "feedback_invalid_velocity"
        while decision_index < len(decisions) and decisions[decision_index]["decision_ns"] - first <= round(t * 1e9):
            row = decisions[decision_index]
            decision_index += 1
            if row.get("front_stop") is True:
                core.stop()
            raw_path, capture = row.get("front_path"), row.get("forward_ns")
            if "reference_stale" in active:
                continue
            if raw_path is not None and clock_kind(capture) == "unix_like":
                try:
                    path = tuple(Pose(*map(float, p)) for p in raw_path)
                    core.set_reference(Reference((capture-first)/1e9, (row["decision_ns"]-first)/1e9, path))
                except (TypeError, ValueError):
                    core.reference_fault("reference_invalid_geometry_frame_or_timestamp")
        if "stop" in active:
            core.stop()
        if scene != "recorded" or "unknown" in active:
            desired = "unknown" if "unknown" in active else scene
            if map_kind != desired:
                if desired == "unknown":
                    grid = Grid(t, "synthetic", "odom", .10, (0., 0.), {})
                else:
                    obstacle = ((.9, -.2, 1.1, .2),) if desired == "single-block" else ()
                    grid = synthetic_grid(t, rectangles=obstacle)
                map_kind = desired
            if core.grid is None or t - core.grid.stamp >= config["control"]["map_ttl_s"] / 2 or core.grid.cells != grid.cells:
                core.set_map(replace(grid, stamp=t))
        tick = loop.poll(t)
        result = core.tick(tick)
        vx, vy, omega = components(result.command)
        info.append({
            "tick_s": t, "jitter_s": tick.jitter_s, "missed_periods": tick.missed_periods,
            "state": result.state, "reason": result.reason,
            "arrival_status": result.arrival_status, "valid_until_s": result.valid_until,
            "vx_m_s": vx, "vy_m_s": vy, "omega_rad_s": omega,
            "side": result.selected.side if result.selected else 0,
            "clearance_m": result.selected.clearance_m if result.selected else None,
            "candidate_scores": [asdict(candidate) for candidate in result.candidates],
            "selected_path": [asdict(p) for p in result.selected.path] if result.selected else None,
            "rejected": result.rejected,
            "rejected_candidates_vx_vy_omega_reason": result.rejected_candidates,
            "feedback_consumed_stamp_ns": result.feedback_consumed_stamp_ns,
            **core.diagnostics(t),
        })
    sides = [r["side"] for r in info if r["side"]]
    clearance = [r["clearance_m"] for r in info if r["clearance_m"] is not None]
    moving = [any(r[k] for k in ("vx_m_s", "vy_m_s", "omega_rad_s")) for r in info]
    since = None
    interruptions = []
    for (before, after), row in zip(zip(moving, moving[1:]), info[1:]):
        if before and not after:
            since = row["tick_s"]
        elif not before and after and since is not None:
            interruptions.append(row["tick_s"] - since)
            since = None
    expiry_gaps = [
        max(0., after["tick_s"] - before["valid_until_s"])
        for before, after, was_moving in zip(info, info[1:], moving)
        if was_moving
    ]
    return info, {
        "scene": scene, "injections": list(injections), "ticks": len(info), "moving_ticks": sum(moving),
        "states": dict(Counter(r["state"] for r in info)),
        "reasons": dict(Counter(r["reason"] for r in info)),
        "side_reversals_between_nonzero_sides": sum(a != b for a, b in zip(sides, sides[1:])),
        "motion_to_zero_transitions": sum(a and not b for a, b in zip(moving, moving[1:])),
        "interruption_durations_s": interruptions,
        "expiry_gaps_between_ticks_s": [gap for gap in expiry_gaps if gap > 1e-9],
        "min_predicted_clearance_m": min(clearance, default=None),
    }


def render_report(run, original, simulated):
    def fields(data):
        return "\n".join(f"| `{key}` | `{json.dumps(value, ensure_ascii=False)}` |" for key, value in data.items())

    return f"""# {run} 时间轴复算与离线控制回放

## 原始实机记录

| 指标 | 复算结果 |
|---|---|
{fields(original)}

`timeline.csv` 同时列出决策前最新的轮速日志样本、绝对时间最近的样本和原控制器实际消费字段。源日志中的实际消费字段为空时，无法还原缓存内容。轮速、画面、决策之间的差值以记录的时间戳计算，历史跨设备时钟偏差仍待测量。

## 小脑离线输出

| 指标 | 回放结果 |
|---|---|
{fields(simulated)}

回放用轮速头时间作为模拟接收时间，参考路径按原始决策记录时间送达。底盘位姿来自已记录的反馈，回放速度不会改变原有位姿，也无法代表实机绕行表现。

`recorded` 场景缺少可信米制障碍图，保持停车。其他场景和故障注入属于合成验证。图像投影只用于查阅原报告。间距来自合成图的候选轨迹及其制动段，是扣除扫掠离散误差且受搜索上限截断的保守下界。空值代表没有可报告的近场障碍间距。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scene", choices=("recorded", "open", "single-block"), default="recorded")
    parser.add_argument("--inject", type=fault_arg, action="append", default=[])
    args = parser.parse_args(argv)
    events, wheels = load_run(args.evidence, args.run)
    rows, summary = audit(events, wheels, camera_ttl_s=.45, feedback_age_s=.12)
    samples, simulated = simulate(events, wheels, load_config(args.config), scene=args.scene, injections=args.inject)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "timeline.csv", rows)
    (args.out / "original-events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "controller.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(args.out / "controller.csv", [{k: v for k, v in row.items()
              if k not in {"candidate_scores", "selected_path", "rejected",
                           "rejected_candidates_vx_vy_omega_reason"}} for row in samples])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "simulation-summary.json").write_text(json.dumps(simulated, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render_report(args.run, summary, simulated), encoding="utf-8")
    print(json.dumps({"original": summary, "simulation": simulated}, indent=2))


if __name__ == "__main__":
    main()
