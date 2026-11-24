# Local training monitor

[简体中文](zh-CN/training-monitor.md)

Start in an environment with AlohaMini installed. The default address is `http://127.0.0.1:7860/`. Metrics and logs use the platform package. Evaluation trajectories also use PyArrow, included in the training environment.

```bash
conda activate alohamini
alohamini train monitor --open
alohamini train monitor --runs-dir /path/to/runs --datasets-dir /path/to/datasets
```

The workspace defaults to `~/Alohamini_workspace`. Set `ALOHAMINI_WORKSPACE` to change it.

| Option | Use |
| --- | --- |
| `--runs-dir PATH` | One training output or its parent. Defaults to the workspace's `runs/`. Each resume is a separate attempt. |
| `--datasets-dir PATH` | A native AlohaMini or LeRobot v3 evaluation dataset or its parent. Repeat for multiple locations. Defaults to the workspace's `datasets/`. |
| `--logs-dir PATH` | Launcher configurations and console logs. Defaults to the workspace's `logs/training/`. |
| `--port N`, `--open` | Set the loopback port. Use `0` for a free port and `--open` to open a browser. |
| `--allow-origin ORIGIN` | Permit browser API reads from an exact Studio origin. Repeat for multiple origins. |

## View cloud training locally

Start the monitor on the training host with the actual run directory. Add `--logs-dir /path/to/logs` for a custom console directory.

```bash
alohamini train monitor --runs-dir /path/to/runs --port 7860
```

On your computer, forward the port and open `http://127.0.0.1:7861/`. Replace `user@training-host` with your SSH login. Keep both commands running while viewing. The local computer needs only SSH and a browser.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:7861:127.0.0.1:7860 user@training-host
```

## Inspect a run

- The first visit opens in English. Use the top tabs to switch views. The collapsible sidebar lists runs, policies and the selected run’s last update. Tick additional runs to compare them. Language and auto-refresh controls are at the top right, or in the sidebar on narrow screens. Use `?lang=en` or `?lang=zh` for a language-specific link.
- **Metrics** shows the objective, recorded policy components and learning rates, with up to eight charts per page. In the sidebar under **Chart settings**, choose metrics, the X axis, trend window or minimum desktop chart width (default 640 px). Faint lines show individual updates, trend lines show means of complete record windows, and dashed lines show validation. Click a legend entry to toggle it. Hover or focus a metric name for its definition.
- Drag a **Range** handle or enter both bounds and press Enter. The interval applies to all overview charts. **Full history** follows new records. **Expand chart** has its own interval, zoom, pan, undo and Y-axis settings. Available X axes are recorded steps, samples and elapsed seconds.
- **Diagnostics** compares two consecutive training windows and shows valid counts and component changes. Validation shows the latest and lowest recorded loss. Enter a target to find the first qualifying step. Expand the distribution, clipping, learning rates, timing and sampled module updates for detail.
- **Configuration** lets you search tokenizers, base models, initialization weights, frozen parameters and optimizer settings. Expand a resource to see recorded paths, revisions or hashes, or import a preflight JSON for the run. **Logs** switches between the live training console and raw JSONL.
- **Evaluation** accepts local recordings and offline reports (JSON, up to 4 MiB). Select an episode and channel for frame-index trajectories. Reports show per-channel MAE, signed bias, RMSE and per-offset MAE when supplied. The view includes available source and checkpoint manifest information. Imported selections are stored in the current browser.

## Optional training diagnostics

| Training option | Measurement |
| --- | --- |
| `--diagnostics.modules='["model.action_head"]' --diagnostics.update_every=500` | ACT action-head example. On the chosen updates, copy the selected trainable parameters and measure pre-clip gradients, actual weight changes and their relative norms. Supports single-device and replicated DDP. |
| `--diagnostics.train_eval_samples=64` | During periodic validation, evaluate a fixed training subset with the same weights and evaluation mode. Records `train_eval_loss` and `generalization_gap`. |

Both are off by default. Freeze flags are collected once at startup. Offline error summaries reuse evaluation predictions. New successful updates include sample counts, monotonic elapsed seconds and event time. Historical fields appear when recorded. Closing the monitor leaves training running.

## Files and Studio API

Run files include `train.json`, `metrics.jsonl`, `validation-metrics.jsonl`, `metrics-schema.json`, `metrics-resources.json` and the directory-wide `offline-evaluation.json`. Resumes use matching `from-STEP-ATTEMPT` metrics and preflight files and a `resume-STEP-ATTEMPT.json` configuration. Initial runs also accept `resources.json` in their output directory and a matching launch preflight in the logs directory. Bundles with `runs/RUN` and a sibling `logs/RUN.preflight.json` are discovered automatically.

Read-only HTTP GET endpoints use the `/api/v1` prefix. Studio can use a same-origin proxy with a loopback upstream Host, or an exact browser origin set with `--allow-origin`.

| Endpoint | Parameters and response |
| --- | --- |
| `/runs`, `/runs/{id}` | Listing with `id`, `version` and `metadata_version`. Details include phase summaries (including observed `policy_metrics`), configuration, `schema.runtime`, `preflight`, `launch_preflight`, console parameter counts and the offline report. Use `brief=1` for dynamic summaries. |
| `/runs/{id}/series` | Up to 32 repeated `metric` keys, `axis=step\|samples\|elapsed_s`, inclusive `min/max`, and `points=16..4096` per line (default 800). Set `mean_window=0..4096` for loss trends, off by default. |
| `/runs/{id}/diagnostics` | `metric`, `window=2..4096`, optional validation loss `target`. Returns current and previous window statistics, optimizer diagnostics and latest, previous and best validation observations. |
| `/runs/{id}/records`, `/runs/{id}/console` | Records accept `phase=train\|validation`, `limit=1..1000`, `cursor` and optional `tail=1`. Console reads use a log `name` and byte cursor. |
| `/datasets`, `/datasets/{id}`, `/datasets/{id}/trajectory` | Dataset listing, metadata (optional `run` for manifest matching), and frame-index series for `episode`, `field` and `points`. |

Series return `extent`, `cursor`, `series` and optional `means` separately for `train` and `validation`. Each metric contains `count` and `points: [[x,y], ...]`. Pass cursors as `after_train` and `after_validation` with unchanged metrics, axis, bounds, resolution and mean window. Append points and counts on `append`. Replace the phase on `replace`. Raw records support pagination and return malformed-line counts.
