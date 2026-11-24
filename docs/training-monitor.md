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

## Inspect a run

- The first visit opens in English. Switch languages in the header or use `?lang=en` or `?lang=zh`. Collapse the desktop sidebar to widen the plots. Tick additional runs to compare them.
- **Metrics** selects recorded components and optimizer groups for the policy. Hover or focus a metric name for its definition. Faint lines show individual updates, trend lines show arithmetic means of complete record windows, and dashed lines show validation. Click a legend entry to toggle it. Long histories retain endpoints, local extrema and gaps per metric.
- **Expand chart** opens an independent interval. Drag to zoom, pan, undo, restore the full range, or include zero on the Y axis. Zooming queries the selected interval again. Available X axes are recorded steps, samples and elapsed seconds.
- **Diagnostics** shows the recent distribution, P50/P95, standard deviation, slope per step, consecutive window means, learning rates, clipping and timing. Enter a validation loss target to find the first observed checkpoint at or below it. Clipping inferred from a legacy norm and threshold is labelled.
- Expand **Training setup** for startup freeze flags, parameter counts, optimizer groups and preflight results. Open a current `train check` report or a historical `RUN.preflight.json` here. Older consoles may supply aggregate parameter counts. **Configuration** supports key and value filters. **Logs** displays raw JSONL and available launch consoles.
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
| `/runs`, `/runs/{id}` | Listing with `id`, `version` and `metadata_version`. Details include phase summaries, configuration, `schema.runtime`, `preflight`, `launch_preflight`, console parameter counts and the offline report. Use `brief=1` for dynamic summaries. |
| `/runs/{id}/series` | Up to 32 repeated `metric` keys, `axis=step\|samples\|elapsed_s`, inclusive `min/max`, and `points=16..4096` per line (default 800). Set `mean_window=0..4096` for loss trends, off by default. |
| `/runs/{id}/diagnostics` | `metric`, `window=2..4096`, optional validation loss `target`. Returns raw-record window statistics, optimizer diagnostics and observed validation results. |
| `/runs/{id}/records`, `/runs/{id}/console` | Records accept `phase=train\|validation`, `limit=1..1000`, `cursor` and optional `tail=1`. Console reads use a log `name` and byte cursor. |
| `/datasets`, `/datasets/{id}`, `/datasets/{id}/trajectory` | Dataset listing, metadata (optional `run` for manifest matching), and frame-index series for `episode`, `field` and `points`. |

Series return `extent`, `cursor`, `series` and optional `means` separately for `train` and `validation`. Each metric contains `count` and `points: [[x,y], ...]`. Pass cursors as `after_train` and `after_validation` with unchanged metrics, axis, bounds, resolution and mean window. Append points and counts on `append`. Replace the phase on `replace`. Raw records support pagination and return malformed-line counts.
