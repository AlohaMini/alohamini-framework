# RLT

RLT learns a residual action policy on top of a frozen reference policy. The supported base is a native π0.5 checkpoint using recorded joint coordinates. Training runs on one device. The robot collector supports a passive left leader.

Use a separate output directory for each stage. Paths in a JSON configuration are relative to the configuration file. A minimal configuration is:

```json
{
  "checkpoint": "/path/to/base/pretrained_model",
  "dataset": "/path/to/demonstrations",
  "output_dir": "/path/to/runs/token",
  "task": "Place the block in the tray"
}
```

| Stage | Command |
| --- | --- |
| Train the visual token on demonstrations | `alohamini rlt train-token --config run.json --steps 5000 --batch_size 4 --lr 0.0003` |
| Collect and score robot episodes | `alohamini rlt collect --config run.json --host HOST --left-port PORT --output-dir rollouts --chunk-size 10 --policy.n_action_steps 5` |
| Export scored episodes or retry an interrupted export | `alohamini rlt export-v3 --output-dir rollouts` |
| Train the actor and critic | `alohamini rlt train-ac --config run.json --dataset rollouts/lerobot_v3 --token-checkpoint /path/to/runs/token/token --output-dir actor-run --steps 5000 --save_freq 1000` |
| Resume actor and critic | Repeat the same training command with `--resume` and a larger total `--steps`. |
| Train an optional phase classifier | `alohamini rlt train-gate --config run.json --dataset rollouts/lerobot_v3 --intervals intervals.json --token-checkpoint /path/to/runs/token/token --output-dir gate-run` |
| Deploy the residual actor | Add `--token-checkpoint /path/to/runs/token/token --actor-checkpoint actor-run/actor` to collection. Add `--gate-checkpoint gate-run/gate` for automatic phase changes. |

Use the same RL chunk length, base checkpoint and learned coordinate names for collection, replay and deployment. Set `learned_action_names` to select named action and proprioception coordinates. The collector defaults to the left arm. `--policy.n_action_steps` sets how many actions from a prediction can execute before replanning. Collection FPS must match the base checkpoint. Token training preserves recorded task labels and the base sampling rate.

During collection, **H** takes over or hands control back, **C** switches the correction phase, **Space** pauses, and **→** ends the episode. Score it with **1** or **0** once the robot has returned and the scene is ready. Unscored episodes remain pending. The optional gripper-release rule for left-arm placement is configured with `--gripper-exit` or `--no-gripper-exit`.

The portable v3 dataset includes `meta/rlt` with operator scores, command outcomes and terminal observations. Copy the whole dataset. Standard action training uses windows with valid, fully acknowledged targets. AC replay uses acknowledged learned coordinates and accounts for intervening control steps. Merge scored datasets with the platform dataset merge command.

Token and gate support `--mixed_precision none|bfloat16|float16` subject to device support. AC uses FP32. The `token`, `actor` and `gate` paths select deployment weights. AC saves optimizers, target networks and sampling state under `checkpoints/<step>/training_state`.

Gate interval files use episode IDs from the scored dataset and half-open frame ranges. Supply at least three reviewed episodes covering both classes for training, selection and held-out evaluation.

```json
{"version":1,"fps":30,"episodes":[{"episode_id":"<episode ID>","segments":[{"name":"placement","start_frame":20,"end_frame_exclusive":45,"reward":1}]}]}
```
