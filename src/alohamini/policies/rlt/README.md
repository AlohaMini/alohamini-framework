# RLT

RLT trains a residual actor on a frozen native π0.5 policy using recorded joint coordinates. Training runs on one device. Create the example `run.json` from [Policy post-training](../../../../docs/post-training.md#offline-rl-with-rlt) · [中文操作步骤](../../../../docs/zh-CN/post-training.md#rlt-离线强化学习). Run these commands from its directory.

```bash
alohamini train rlt token --config run.json --steps 5000 --batch_size 4

alohamini collect rlt --config run.json --output-dir round-1 --policy.n_action_steps 5

alohamini train rlt ac --config run.json --dataset round-1/lerobot_v3 \
  --token-checkpoint token-run/token --output-dir ac-run --steps 5000

alohamini collect rlt --config run.json --output-dir round-2 --policy.n_action_steps 5 \
  --token-checkpoint token-run/token --actor-checkpoint ac-run/actor
```

| Stage | Input | Output to use |
|---|---|---|
| `train rlt token` | Base checkpoint and demonstrations with recorded task labels | `token-run/token`, the token with lowest validation loss. See `token-metrics.jsonl` in the run directory. |
| `collect rlt` | Base policy, calibrated leaders for the selected arms, optional token and actor | Scored episodes in `round-1/`, automatically published to `round-1/lerobot_v3/` |
| `train rlt ac` | Matching token and scored RLT data (raw collection or complete v3 directory) | `ac-run/actor` for deployment, `ac-run/ac-metrics.jsonl` for training metrics |
| `train rlt gate` (optional) | Token, scored data and reviewed phase intervals | `gate-run/gate` for automatic phase switching, `gate-run/gate-validation.json` for held-out results |

## Collection and replay settings

- Set `learned_action_names` in `run.json` to coordinate names or arm groups, e.g. `["arm_left"]`. A new run defaults to arm joints and grippers present in both checkpoint state and action, excluding fixed targets. A saved token, resumed collection or scored token-training dataset supplies its recorded selection. Use the same base, coordinates and `chunk_size` throughout.
- Collection uses the checkpoint FPS. `--policy.n_action_steps` sets how many steps may execute before replanning (1 through `chunk_size`). Token training preserves the recorded task labels and base sampling rate.
- Set `--left-port` or `--right-port` for the selected leaders. Relative takeover limits joint speed to 60°/s by default (`--max-joint-velocity-deg-s`) and gripper opening to 50 percentage points/s. Space pauses or resumes. Use `--return-to-start` to return controlled joints to their initial positions before scoring.
- `--gripper-exit` enables phase exit on a measured gripper opening. If multiple learned grippers are available, specify e.g. `--gripper-exit-field arm_right_gripper.pos`.
- Token and gate support `--mixed_precision none|bfloat16|float16` on compatible devices. AC uses FP32. AC samples acknowledged learned coordinates and accounts for intervening control steps. Standard action training requires fully acknowledged targets. When copying or merging v3 data, keep the whole dataset including `meta/rlt/`. Use `alohamini dataset edit --operation.type merge` to merge datasets.

## Resume or recover

Append to an existing collection by repeating its command with `--resume`. To resume AC, use the same data, token, base, output directory and training settings. Set `--steps` to the new total. The optimizer, target networks and sampling state are restored from `ac-run/checkpoints/last/training_state/`.

```bash
alohamini train rlt ac --config run.json --dataset round-1/lerobot_v3 \
  --token-checkpoint token-run/token --output-dir ac-run --steps 10000 --resume
```

To score a sealed episode, use its 32-character hexadecimal ID from `round-1/pending/`. Retry export only if publication was interrupted.

```bash
alohamini collect score --output-dir round-1 --episode EPISODE_ID --reward 1
alohamini dataset export round-1 --format lerobot-v3 --output round-1/lerobot_v3
```

## Automatic phase switching

For hands-free phase selection, create `intervals.json` with the collection FPS and reviewed episode IDs. Label each phase from its first frame (inclusive) to its end (exclusive). The entry below shows one episode. Include at least three reviewed episodes with phase and non-phase frames so training, checkpoint selection and held-out evaluation each cover both classes.

```json
{"version":1,"fps":30,"episodes":[{"episode_id":"EPISODE_ID","segments":[{"name":"phase_1","start_frame":20,"end_frame_exclusive":45,"reward":1}]}]}
```

```bash
alohamini train rlt gate --config run.json --dataset round-1/lerobot_v3 \
  --intervals intervals.json --token-checkpoint token-run/token --output-dir gate-run

alohamini collect rlt --config run.json --output-dir round-gate --policy.n_action_steps 5 \
  --token-checkpoint token-run/token --actor-checkpoint ac-run/actor --gate-checkpoint gate-run/gate
```
