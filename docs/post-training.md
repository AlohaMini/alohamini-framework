# Policy Post-Training

**English** | [简体中文](zh-CN/post-training.md)

Activate the `alohamini` environment on the PC, prepare a trained native policy checkpoint and start the robot Host. Each collection run uses fixed policy weights. Load the new checkpoint after training to start the next round.

| Objective | Collection command | Next step |
|---|---|---|
| Improve a policy with human corrections | `alohamini collect dagger` | Merge corrections with the original demonstrations and fine-tune from the original checkpoint |
| Train a residual policy for π0.5 using success or failure rewards | `alohamini collect rlt` | Train the actor and critic offline on scored RLT episodes |
| Record complete, scored interactions from a native policy | `alohamini collect scored` | Publish a v3 dataset with command receipts and rewards |

## DAgger corrections and fine-tuning

Connect calibrated leaders and use an X11 keyboard session.

```bash
alohamini collect dagger --host HOST --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint --dataset task_corrections --task "TASK"
```

Press Space to pause and hold the followers. Select a takeover mode with `--dagger.handover_mode`.

| Mode | Start a correction from pause |
|---|---|
| `aligned` (default) | Press Tab and align the joints and grippers by hand. Each joint holds once settled. Support the leaders and press Tab again to start the correction. |
| `relative` | Press Tab once. Each arm's current pose becomes its origin. Leader displacement sets absolute follower joint targets. Leader torque stays off. |

During corrections the base stops and the lift holds. Press Tab to end and save the correction segment, then Space to resume the policy from the current observation. Esc exits.
Aligned mode defaults to a leader hold output limit of `0.2` (`--dagger.hold_output_limit`) and a joint tolerance of 3 degrees (`--dagger.alignment_tolerance_deg`). Support the leaders during holds. Faults, feedback interruption or alignment timeout release the holds.
Synchronous inference handles pause after the current computation returns. Add `--inference.type=async` for asynchronous inference. DAgger requires unfixed arm dimensions.

Merge demonstrations and corrections, then begin a new fine-tuning run:

```bash
alohamini dataset edit --operation.type merge \
  --operation.roots '["~/Alohamini_workspace/datasets/task_demo", "~/Alohamini_workspace/datasets/task_corrections"]' \
  --output ~/Alohamini_workspace/datasets/task_merged

alohamini train \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_merged \
  --output_dir ~/Alohamini_workspace/runs/act_dagger_01 \
  --policy.device cuda --steps 10000 --batch_size 8 --background
```

Both datasets must share fields, cameras, calibration, and FPS. Select an independent validation set using merged episode indices.
The command starts a new fine-tuning run. Its background log is `~/Alohamini_workspace/logs/training/act_dagger_01.log`.
Use the new checkpoint as `--policy.path` in the next `alohamini collect dagger` run.

## Offline RL with RLT

Create `run.json` in a new working directory. Set the trained native π0.5 joint-coordinate checkpoint, demonstration dataset, task prompt and robot Host address. Token training needs at least two usable demonstration episodes for the whole-episode training and validation split.

```json
{
  "checkpoint": "/path/to/pi05/pretrained_model",
  "dataset": "/path/to/demonstrations",
  "task": "Place the block in the tray",
  "host": "HOST",
  "output_dir": "token-run",
  "chunk_size": 10
}
```

From that directory, train the token, collect with the base policy, train the actor and critic offline, then deploy the residual policy.

```bash
alohamini train rlt token --config run.json --steps 5000 --batch_size 4

alohamini collect rlt --config run.json --output-dir round-1 --policy.n_action_steps 5

alohamini train rlt ac --config run.json --dataset round-1/lerobot_v3 \
  --token-checkpoint token-run/token --output-dir ac-run --steps 5000

alohamini collect rlt --config run.json --output-dir round-2 --policy.n_action_steps 5 \
  --token-checkpoint token-run/token --actor-checkpoint ac-run/actor
```

- **First collection.** Press H for relative takeover or handback and C to mark a residual phase. If no phase is marked, AC uses eligible acknowledged actions across the episode. Press → to finish. After resetting the scene, enter 0 (failure) or 1 (success) and press Enter. Scored episodes are published to `round-1/lerobot_v3/`. Keep both outcomes.
- **Residual deployment.** Press C to switch actor corrections on or off. The base policy controls the robot outside this phase. The example predicts 10 steps and executes up to 5 before replanning. Keep the base, learned action coordinates and chunk length consistent across rounds.

See the [RLT guide](../src/alohamini/policies/rlt/README.md) for AC resume, automatic phase switching with a gate, scoring a sealed episode and retrying export.

## Scored policy episodes

```bash
alohamini collect scored --host HOST --robot-model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --task "pick up the object" --output-dir ~/Alohamini_workspace/datasets/scored
```

Use Space to pause or resume. Add `--intervention` to enable relative leader takeover and handback with H. Use `--intervention left` or `--intervention right` for a single arm. Press → at task completion or T to truncate, then enter 0 or 1 and press Enter when the scene is ready for the next episode. Q exits and leaves unfinished episodes pending.

Scored episodes are published under `lerobot_v3/`. Each control step records its source, requested target and Host receipt. `experience.applied_mask` identifies acknowledged action coordinates. On the step preceding the final observation, `next.done` marks the boundary and `next.terminated` or `next.truncated` identifies the reason. Use `collect scored --resume` to append or `dataset export DIR --format lerobot-v3 --output DIR/lerobot_v3` to retry publication.

Each supervised action window requires receipts for every action coordinate used by the standard reader. A Host that reports only arm targets leaves base and lift coordinates unconfirmed. For a sealed episode awaiting a score, run `alohamini collect score --output-dir DIR --episode ID --reward 0` (or `1`).
