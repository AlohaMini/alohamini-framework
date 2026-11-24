"""Build chunk transitions from scored, acknowledged control sequences."""

import json
from pathlib import Path

import numpy as np

from alohamini.datasets.rlt.segments import training_ranges
from alohamini.datasets.rlt.storage import (
    json_lines,
    load_snapshot,
    validate_training_episode,
)
from alohamini.policies.rlt.policy import encode_physical, pad_actions, require_contract


def command_key(command):
    return tuple(
        command.get(name) for name in ("client_id", "host_session_id", "control_epoch", "sequence")
    )


def build_windows(rows, boundaries, terminal, *, chunk_size, stride, reward, encode):
    """All coordinates in a window use its own initial action context.

    Boundaries maps step indices to token/state/ref/context. A group-ending
    transition bridges fixed-policy/HOLD steps to the next critical observation,
    or to the segment's terminal observation, without learning those gap actions.
    """
    result = []
    for start in range(0, len(rows), stride):
        selected = rows[start : start + chunk_size]
        count = len(selected)
        end = start + count
        first = boundaries[start]
        last = terminal if end == len(rows) else boundaries[end]
        actual = encode(np.asarray([r["actual"] for r in selected], np.float32), first["context"])
        reference = np.asarray(first["ref"], np.float32).copy()
        human = np.asarray(
            [r["source"] == "HUMAN" for r in selected] + [False] * (chunk_size - count)
        )
        actual = pad_actions(actual, chunk_size)
        demonstration = np.where(human[:, None], actual, reference)
        rewards = np.zeros(chunk_size, np.float32)
        done = end == len(rows) and terminal["terminal"]
        gap = terminal.get("gap_steps", 0) if end == len(rows) else 0
        if done:
            rewards[count - 1] = reward
        valid = np.arange(chunk_size) < count
        remaining = len(rows) - end if end < len(rows) else terminal.get("next_length", chunk_size)
        next_valid = np.arange(chunk_size) < (0 if done else min(chunk_size, remaining))
        result.append(
            dict(
                z=first["z"],
                state=first["state"],
                ref=reference,
                bc_target=demonstration,
                action=actual,
                reward=rewards,
                valid=valid.astype(np.float32),
                elapsed_steps=np.asarray([count + gap], np.int64),
                reward_delay=np.asarray([gap if done else 0], np.int64),
                human=human,
                next_z=last["z"],
                next_state=last["state"],
                next_ref=last["ref"],
                next_valid=next_valid.astype(np.float32),
                done=np.asarray([float(done)], np.float32),
                start=np.asarray([start], np.int64),
            )
        )
    return result


def materialize_episode(policy, directory, on_boundary=lambda: None):
    """Accept a native episode or meta/rlt/episode_NNNNNN from a standalone v3 copy."""
    directory = Path(directory)
    if (directory / "v3.json").is_file():
        from contextlib import closing

        from alohamini.datasets.rlt.v3_reader import V3Observations

        cameras = getattr(policy.native, "manifest", {}).get("cameras")
        with closing(V3Observations(directory, cameras)) as observations:
            return _materialize(policy, directory, on_boundary, observations)
    return _materialize(policy, directory, on_boundary, None)


def _materialize(policy, directory, on_boundary, observations):
    summary = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "committed"
        or not summary.get("ended")
        or summary.get("evaluation")
    ):
        raise ValueError("Replay only accepts ended and manually scored episodes")
    if type(summary.get("reward")) is not int or summary["reward"] not in (0, 1):
        raise ValueError("Missing binary episode reward")
    require_contract(summary["base_contract"], policy.contract)
    validate_training_episode(directory, summary)
    rows = json_lines(directory / "steps.jsonl")
    outcomes = {command_key(row["command"]): row for row in json_lines(directory / "acks.jsonl")}
    ranges = training_ranges(summary, len(rows))
    result = []
    for start, stop, reward in ranges:
        result.extend(
            _materialize_range(
                policy,
                directory,
                on_boundary,
                observations,
                summary,
                rows[start:stop],
                outcomes,
                stop,
                len(rows),
                reward,
            )
        )
    return result


def _materialize_range(
    policy,
    directory,
    on_boundary,
    observations,
    summary,
    rows,
    outcomes,
    stop,
    episode_length,
    reward,
):
    names = policy.native.names
    learned = policy.contract["learned_names"]
    eligible = []
    whole_episode = (
        summary.get("phase_labels") == "optional-operator-v2"
        and summary.get("critical_entries") == 0
    )
    for row in rows:
        outcome = outcomes.get(command_key(row["command"]), {})
        targets = outcome.get("targets")
        if (whole_episode or row["critical"]) and row["source"] in ("VLA", "RLT", "HUMAN"):
            if outcome.get("status") in ("superseded", "rejected"):
                continue
            if outcome.get("status", "applied") != "applied":
                raise ValueError(f"Unknown disposition for critical step {row['index']}")
            if targets is None or not set(learned) <= targets.keys():
                raise ValueError(f"Missing Host acceptance for critical step {row['index']}")
            actual = list(row["requested"])
            for name in learned:
                actual[names.index(name)] = targets[name]
            row = dict(row, actual=actual)
            eligible.append(row)
    if not eligible:
        return []
    # Do not splice controls across pauses, lost observations, or source time gaps.
    groups = []
    for row in eligible:
        if not groups or row["index"] != groups[-1][-1]["index"] + 1:
            groups.append([])
        elif row["time"] - groups[-1][-1]["time"] > 2.5 / policy.native.fps:
            groups.append([])
        groups[-1].append(row)

    def boundary(path, index):
        on_boundary()
        if observations is None:
            ref = policy.reference(load_snapshot(path), index)
        else:
            snapshot, rgb = observations.read(index)
            ref = policy.reference(snapshot, index, rgb_images=rgb)
        if ref.token is None:
            raise ValueError("AC replay requires a trained token_checkpoint")
        return dict(
            z=ref.token[0].cpu().numpy(),
            state=ref.proprio[0].cpu().numpy(),
            ref=ref.reference[0][:, policy.indices].cpu().numpy(),
            context=ref.context,
        )

    all_transitions = []
    next_boundary = None
    for group_index, group in enumerate(groups):
        # Materialize one observation at a time to keep GPU memory bounded.
        needed = set(range(0, len(group), policy.config.replay_stride))
        needed.update(min(i + policy.config.chunk_size, len(group)) for i in tuple(needed))
        boundaries = {}
        for index in sorted(needed - {len(group)}):
            row = group[index]
            path = directory / "observations" / str(row["index"])
            if index == 0 and next_boundary is not None:
                boundaries[index] = next_boundary
            else:
                boundaries[index] = boundary(path, row["index"])
        last_index = group[-1]["index"]
        is_terminal = group_index == len(groups) - 1
        next_index = stop if is_terminal else groups[group_index + 1][0]["index"]
        end_path = (
            directory / "terminal"
            if next_index == episode_length
            else (directory / "observations" / str(next_index))
        )
        terminal = boundary(end_path, next_index)
        next_boundary = terminal
        terminal["terminal"] = is_terminal
        gap = next_index - last_index - 1
        if not is_terminal:
            interval = groups[group_index + 1][0]["time"] - group[-1]["time"]
            gap = max(gap, round(interval * policy.native.fps) - 1)
        terminal["gap_steps"] = gap
        terminal["next_length"] = 0 if is_terminal else len(groups[group_index + 1])

        def encode(actions, context):
            return (
                encode_physical(policy.native, actions[None], context)[0][:, policy.indices]
                .cpu()
                .numpy()
            )

        all_transitions.extend(
            build_windows(
                group,
                boundaries,
                terminal,
                chunk_size=policy.config.chunk_size,
                stride=policy.config.replay_stride,
                reward=reward,
                encode=encode,
            )
        )
    return all_transitions
