"""Train a binary gate from reviewed episode-local placement intervals."""

import hashlib
import json
from copy import deepcopy
from contextlib import closing, nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from alohamini.learning.stages import supervised_stage, save_stage_checkpoint
from alohamini.rlt.ac_training import scored_episodes, split_episodes
from alohamini.rlt.gate import GateLatch
from alohamini.rlt.gate_classifier import GATE_FORMAT, StageClassifier
from alohamini.rlt.policy import RLTPolicy, require_contract
from alohamini.rlt.storage import RunLock, atomic_json, json_lines, load_snapshot


def _signature(segment):
    return tuple(segment[k] for k in ("name", "start_frame", "end_frame_exclusive", "reward"))


def reviewed_episodes(root, manifest, excluded=None):
    if manifest.get("version") != 1 or not isinstance(manifest.get("episodes"), list):
        raise ValueError("Expected a version 1 reviewed episode intervals manifest")
    excluded = excluded or {}
    sources = {eid: directory for eid, _, directory in scored_episodes(root)}
    result = {}
    for annotation in manifest["episodes"]:
        eid = annotation["episode_id"]
        if eid in result or eid not in sources:
            raise ValueError(f"Duplicate, missing or uncommitted annotated episode {eid}")
        path = sources[eid]
        summary = json.loads((path / "episode.json").read_bytes())
        frames = len(json_lines(path / "steps.jsonl"))
        segments = [_signature(s) for s in annotation["segments"]]
        if "training_segments" in summary:
            previous = [_signature(s) for s in summary["training_segments"]["segments"]]
            if previous != segments:
                raise ValueError(f"Annotation disagrees with episode intervals in {eid}")
        previous_end = 0
        for name, start, stop, reward in segments:
            if (
                not isinstance(name, str)
                or not name
                or type(start) is not int
                or type(stop) is not int
                or not previous_end <= start < stop <= frames
                or type(reward) is not int
                or reward not in (0, 1)
            ):
                raise ValueError(f"Invalid placement intervals in {eid}")
            previous_end = stop
        reviewed = excluded.get(eid, [])
        interval_keys = {span[:3] for span in segments}
        if any(_signature(s)[:3] not in interval_keys for s in reviewed):
            raise ValueError(f"Quality review disagrees with interval boundaries in {eid}")
        excluded_keys = {_signature(s)[:3] for s in reviewed if s.get("quality") == "exclude"}
        blocked = {s for s in segments if s[:3] in excluded_keys}
        control_spans = [span for span in segments if span not in blocked]
        result[eid] = dict(
            path=path,
            frames=frames,
            summary=summary,
            spans=control_spans,
            blocked=blocked,
            first_target_start=min((s[1] for s in segments), default=None),
        )
    if len(result) < 2:
        raise ValueError("Gate requires at least two reviewed episodes for a whole-episode split")
    return result


def _split(episodes, config):
    rewards = {eid: e["summary"]["reward"] for eid, e in episodes.items()}
    labels = {s[0] for e in episodes.values() for s in e["spans"]}
    if not labels:
        raise ValueError("No reviewed positive stage intervals")
    for offset in range(100):
        train, val = split_episodes(rewards, config.validation_fraction, config.seed + offset)
        if not val:
            # For tiny corpora stratification can keep both singleton outcomes in training.
            ordered = (
                np.random.default_rng(config.seed + offset).permutation(sorted(episodes)).tolist()
            )
            val, train = ordered[:1], ordered[1:]
        if all(
            {s[0] for eid in ids for s in episodes[eid]["spans"]} == labels for ids in (train, val)
        ):
            return train, val
    raise ValueError("Each annotated phase needs coverage in both training and held-out episodes")


def _sample_frames(episode, stride, extra_strides=()):
    frames, labels = [], []
    indices = sorted(
        {i for period in (stride, *extra_strides) for i in range(0, episode["frames"], period)}
    )
    for index in indices:
        if any(a <= index < b for _, a, b, _ in episode["blocked"]):
            continue
        label = next((name for name, a, b, _ in episode["spans"] if a <= index < b), None)
        frames.append(index)
        labels.append(label)
    return frames, labels


def _encode(episode, policy, config, extra_strides=()):
    path = episode["path"]
    frames, names = _sample_frames(episode, config.gate_sample_stride, extra_strides)
    if not frames:
        raise ValueError(f"No eligible gate frames in {path}")
    require_contract(episode["summary"]["base_contract"], policy.contract)
    source = nullcontext(None)
    if (path / "v3.json").exists():
        from alohamini.rlt.v3_reader import V3Observations

        source = closing(V3Observations(path, policy.native.manifest["cameras"]))
    tokens, states = [], []
    policy.native.reset()
    with source as obs:
        for index in frames:
            if obs is None:
                token, state = policy.gate_features(
                    load_snapshot(path / "observations" / str(index))
                )
            else:
                snapshot, rgb = obs.read(index)
                token, state = policy.gate_features(snapshot, rgb_images=rgb)
            tokens.append(token.detach().cpu())
            states.append(state.detach().cpu())
    z, state = torch.cat(tokens), torch.cat(states)
    if not torch.isfinite(z).all() or not torch.isfinite(state).all():
        raise ValueError(f"Non-finite gate features in {path}")
    return dict(
        z=z,
        state=state,
        frames=frames,
        names=names,
        spans=episode["spans"],
        first_target_start=episode["first_target_start"],
        blocked=sorted(episode["blocked"]),
        y=torch.tensor([label is not None for label in names], dtype=torch.float32),
    )


def _assessment(samples, scores, config, fps, stride=None):
    stride = config.gate_sample_stride if stride is None else stride
    tp = fp = fn = tn = false_entries = early_entries = 0
    phases = {}
    for eid, e in samples.items():
        latch = GateLatch(
            config.gate_enter_threshold, config.gate_exit_threshold, config.gate_confirmations
        )
        take = [i for i, frame in enumerate(e["frames"]) if frame % stride == 0]
        frames = np.asarray([e["frames"][i] for i in take], dtype=int)
        names = [e["names"][i] for i in take]
        contiguous = np.diff(frames) == stride
        for i in range(len(contiguous)):
            if any(
                frames[i] < end and start < frames[i + 1]
                for _, start, end, _ in e.get("blocked", [])
            ):
                contiguous[i] = False
        predicted = []
        first_start = e.get("first_target_start", min((s[1] for s in e["spans"]), default=None))
        for i, (frame, name) in enumerate(zip(frames, names, strict=True)):
            score = scores[eid][take[i]]
            if i and not contiguous[i - 1]:
                latch.reset()
            before = latch.active
            active = latch.update(score)
            if active and not before and name is None:
                false_entries += 1
                early_entries += int(first_start is not None and frame < first_start)
            predicted.append(active)
        pred = np.asarray(predicted, dtype=bool)
        truth = np.asarray([name is not None for name in names], dtype=bool)
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
        # A new run begins after any unsupervised gap as well as after an inactive prediction.
        starts = np.flatnonzero(pred & ~np.r_[False, pred[:-1] & contiguous]) if len(pred) else []
        stops = (
            np.flatnonzero(pred & ~np.r_[pred[1:] & contiguous, False]) + 1 if len(pred) else []
        )
        previous_stop = None
        for name, begin, end, _ in e["spans"]:
            prior_end, previous_stop = previous_stop, end
            metric = phases.setdefault(
                name,
                dict(
                    segments=0,
                    missed=0,
                    entered_before_start=0,
                    carried_from_previous_phase=0,
                    active_at_end_after_phase=0,
                    entry_offsets_s=[],
                    exit_offsets_s=[],
                ),
            )
            metric["segments"] += 1
            options = [
                (min(end, frames[b - 1] + stride) - max(begin, frames[a]), a, b)
                for a, b in zip(starts, stops, strict=True)
                if frames[a] < end and frames[b - 1] >= begin
            ]
            if not options:
                metric["missed"] += 1
                continue
            _, a, b = max(options)
            offset = int(frames[a]) - begin
            metric["entry_offsets_s"].append(offset / fps)
            metric["entered_before_start"] += int(offset < 0)
            metric["carried_from_previous_phase"] += int(
                prior_end is not None and frames[a] < prior_end
            )
            if b < len(frames) and contiguous[b - 1]:
                metric["exit_offsets_s"].append((int(frames[b]) - end) / fps)
            elif b == len(frames) and frames[-1] >= end:
                metric["active_at_end_after_phase"] += 1
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        false_positive_rate=fp / max(fp + tn, 1),
        false_entries_outside_intervals=false_entries,
        false_entries_before_first_target=early_entries,
        by_phase=phases,
    )


def train_gate(config, *, intervals=None):
    if not config.dataset or not config.token_checkpoint:
        raise ValueError("train-gate requires a scored dataset and frozen token checkpoint")
    dataset = Path(config.dataset).resolve()
    manifest_path = Path(intervals) if intervals else dataset / "rlt-placement-intervals.json"
    if not intervals and not manifest_path.is_file() and dataset.name == "lerobot_v3":
        manifest_path = dataset.parent / manifest_path.name
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    episodes = reviewed_episodes(dataset, manifest)
    train_ids, val_ids = _split(episodes, config)
    # Select weights on separate whole episodes. The final evaluation never chooses a step.
    train_ids, selection_ids = _split({eid: episodes[eid] for eid in train_ids}, config)
    eval_strides = sorted(
        {config.gate_sample_stride, config.chunk_size, config.gate_eval_stride or config.chunk_size}
    )
    out = Path(config.output_dir).resolve()
    if out == dataset:
        raise ValueError("Use a separate gate output directory")
    with RunLock(out / "gate.lock"):
        checkpoint = out / "gate"
        if checkpoint.exists() or (out / "checkpoints").exists():
            raise FileExistsError(f"Gate checkpoint already exists at {checkpoint}")
        policy = RLTPolicy(replace(config, actor_checkpoint="", gate_checkpoint=""))
        contract, sha, fps = policy.contract, policy.token_sha256, policy.native.fps
        if manifest.get("fps") != fps:
            raise ValueError("Gate annotation FPS differs from the policy")
        data = {}
        for eid, episode in episodes.items():
            data[eid] = _encode(episode, policy, config, eval_strides if eid in val_ids else ())
            print(f"Gate {eid}: {len(data[eid]['frames'])} reviewed frames", flush=True)
        del policy
        train = {
            k: torch.cat([data[e][k] for e in train_ids]).to(config.device)
            for k in ("z", "state", "y")
        }
        val = {e: data[e] for e in val_ids}
        selection = {
            k: torch.cat([data[e][k] for e in selection_ids]).to(config.device)
            for k in ("z", "state", "y")
        }
        if selection["y"].unique().numel() != 2:
            raise ValueError("Gate checkpoint selection requires positive and negative frames")
        counts = torch.bincount(train["y"].long(), minlength=2)
        if counts.min() == 0 or not {bool(x) for e in val.values() for x in e["names"]} == {
            False,
            True,
        }:
            raise ValueError("Both gate classes must be present in each whole-episode split")
        # Unbalanced BCE preserves the empirical prior. Monitor missed and false entries separately.
        report = dict(
            train_episodes=train_ids,
            validation_episodes=val_ids,
            selection_episodes=selection_ids,
            selection_metric="episode_mean_BCE",
            eval_strides=eval_strides,
            selection_history={},
            train_frames=int(counts.sum()),
            validation_frames=sum(len(e["frames"]) for e in val.values()),
            fps=fps,
            sample_stride=config.gate_sample_stride,
            enter=config.gate_enter_threshold,
            exit=config.gate_exit_threshold,
            confirmations=config.gate_confirmations,
            interval_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            probes={},
        )
        for mode in ("state_only", "token_only", "token_state"):
            torch.manual_seed(config.seed)
            rng = torch.Generator(device=config.device).manual_seed(config.seed)
            model = StageClassifier(contract, mode).to(config.device)
            def objective(probe, batch):
                return F.binary_cross_entropy_with_logits(probe(batch["z"], batch["state"]), batch["y"])

            with supervised_stage(model, objective, device=config.device, lr=config.lr,
                                  steps=config.gate_steps, clip=10, precision=config.mixed_precision) as update:
                best_loss, best_step, stale = float("inf"), 0, 0
                history = []
                sizes = [len(data[e]["y"]) for e in selection_ids]
                for step in range(1, config.gate_steps + 1):
                    model.train()
                    idx = torch.randint(
                        len(train["y"]), (config.gate_batch_size,), generator=rng, device=config.device
                    )
                    update({"z": train["z"][idx], "state": train["state"][idx], "y": train["y"][idx]})
                    if step == 1 or step % 25 == 0 or step == config.gate_steps:
                        model.eval()
                        with torch.no_grad():
                            held = model(selection["z"], selection["state"])
                            errors = F.binary_cross_entropy_with_logits(
                                held, selection["y"], reduction="none"
                            )
                            score = torch.stack([v.mean() for v in errors.split(sizes)]).mean().item()
                        if not np.isfinite(score):
                            raise ValueError("Nonfinite gate selection loss")
                        history.append(dict(step=step, bce=score))
                        if score < best_loss:
                            best_loss, best_step, stale = score, step, 0
                            best = deepcopy(model.state_dict())
                        else:
                            stale += 1
                            if stale >= 5:
                                break
            model.load_state_dict(best)
            report["selection_history"][mode] = dict(
                best_step=best_step, best_bce=best_loss, last_step=step, scores=history
            )
            with torch.no_grad():
                model.eval()
                scores = {
                    eid: model(e["z"].to(config.device), e["state"].to(config.device))
                    .sigmoid()
                    .cpu()
                    .tolist()
                    for eid, e in val.items()
                }
            report["probes"][mode] = _assessment(val, scores, config, fps)
            report["probes"][mode]["by_stride"] = {
                str(period): _assessment(val, scores, config, fps, period)
                for period in eval_strides
            }
        atomic_json(out / "gate-validation.json", report)
        save_stage_checkpoint(
            out, step, {"classifier": model},
            dict(
                format=GATE_FORMAT,
                contract=contract,
                token_sha256=sha,
                validation=report,
            ), config.to_dict(), aliases=("gate",),
        )
        print(
            f"Gate saved at {checkpoint}; held-out results: {out / 'gate-validation.json'}",
            flush=True,
        )
