# Local training monitor

[简体中文](zh-CN/training-monitor.md)

```bash
alohamini train monitor --open
```

The monitor opens at `http://127.0.0.1:7860/`. Start it before, during or after training. It reads the framework's existing output files. Closing the monitor leaves the trainer running.

| Option | Source |
| --- | --- |
| `--runs-dir PATH` | A single output directory or its parent. Default is `$ALOHAMINI_WORKSPACE/runs`. |
| `--datasets-dir PATH` | An evaluation dataset or a directory of datasets. Repeat for multiple locations. Default is the workspace's `datasets/`. |
| `--logs-dir PATH` | Detached launcher JSON and matching console logs. Default is the workspace's `logs/training/`. |
| `--port N` | Loopback port. Use `0` to choose a free port and print the URL. |
| `--allow-origin ORIGIN` | Enable browser reads for an exact Studio origin, e.g. `http://localhost:5173`. Repeat for multiple origins. |

The default workspace is `~/Alohamini_workspace`. Metrics and console viewing need only the platform package. Evaluation trajectories also use PyArrow, included in the learning and PC environments.

## Inspect and compare

- Select a run to see its latest training and validation records. Tick additional attempts to overlay curves and compare their saved configurations. Each resume has its own trace, including retries from the same checkpoint.
- Choose metrics and an X axis. Drag across a chart or enter a range to query that interval. Use **Full history** to reset. Hover over a point, or focus the plot and use the arrow keys. The question marks explain metric definitions and available aggregation rules.
- Solid lines show training and dashed lines show validation. For sparse records the tooltip gives the observed value at the selected coordinate. Overview counts are the displayed points and recorded values in the current range. Short ranges show all points. Longer ranges preserve endpoints, local extrema and gaps in each metric.
- **Logs** shows the original structured records and available detached console streams. Each console stream is labelled by its launch filename. The browser follows the latest output and bounds the displayed tail. Pause auto-refresh to read it. Hidden tabs suspend polling.
- Use the language control or open with `?lang=zh` or `?lang=en`. The URL includes the selected primary attempt.

The trainer flushes `metrics.jsonl` after each successful optimizer update. Validation uses `validation-metrics.jsonl`. A resumed attempt uses the corresponding `*-from-STEP-ATTEMPT.jsonl`, configuration and metric schema. New records include cumulative consumed samples, seconds since this attempt's training loop started (`elapsed_s`) and Unix event time (`event_time_s`). Validation and checkpoint pauses are included in elapsed time. New training records also include cumulative AMP skips, clipping decisions when clipping is enabled, and per-group rates for multi-group optimizers. Historical files use the axes they actually recorded.

## Evaluation

In **Add evaluation data**, select a local AlohaMini or LeRobot v3 dataset. The monitor reads its metadata and projects numeric Parquet columns for the selected episode and action channel. The trajectory X axis is the recorded frame index. Matching state channel names with compatible declared units can be displayed alongside actions. A missing unit is marked unspecified. A numbered checkpoint's manifest is compared with the SHA-256 in `meta/evaluation.json` when available.

**Evaluation** also shows the run directory's current `offline-evaluation.json`. Use the file chooser to import an additional offline evaluation JSON with `mae_by_action` (up to 4 MiB). MAE is shown separately per action in its recorded scale. The final report belongs to the output directory as a whole. Dataset selections and imported report values are saved for the primary attempt in this browser. Removing one from the view leaves the source intact.

## Studio API

Read-only HTTP GET endpoints under `/api/v1` return JSON. Use the IDs returned by discovery. Studio can use a same-origin proxy with a loopback upstream Host, or the exact browser origin option above.

| Endpoint | Response and options |
| --- | --- |
| `/runs` | `runs[]` with attempt `id`, `name`, resume suffix, `version` and `metadata_version`. Compare versions before fetching details. |
| `/runs/{id}` | Resolved configuration, component schema, console filenames, final offline report, and per-phase metric names, available axes, latest record and anomaly counts. `brief=1` omits configuration, schema, report and filenames. |
| `/runs/{id}/series` | Repeat `metric` up to 32 times. `axis=step\|samples\|elapsed_s`, inclusive `min` and `max`, `points=16..4096` per metric and phase. Defaults are loss, step and 800 points. |
| `/runs/{id}/records` | Unaggregated structured records, `phase=train\|validation`, `limit=1..1000`, optional `cursor`. `tail=1` starts at the latest page. Returns `records`, `cursor`, `more`, `reset` and `invalid_lines`. |
| `/runs/{id}/console` | `name` from the attempt's log list and optional byte `cursor`. Returns a UTF-8 tail with `text`, `cursor` and `reset`. |
| `/datasets`, `/datasets/{id}` | Configured dataset IDs and metadata. Pass `run={attempt_id}` on the detail endpoint to check the recorded checkpoint manifest. |
| `/datasets/{id}/trajectory` | `episode`, exact action `field`, optional `points`. Returns the frame-index series, frame count and declared unit. |

```bash
curl -s http://127.0.0.1:7860/api/v1/runs
curl -s "http://127.0.0.1:7860/api/v1/runs/$RUN_ID/series?metric=loss&metric=grad_norm&points=800"
```

Each series response contains independent `train` and `validation` objects with `mode`, `cursor`, full axis `extent`, and `series[metric] = {count, points: [[x, y], ...]}`. Pass the cursors back as `after_train` and `after_validation` with the same metric selection, range and resolution. Apply `append` by appending points and adding counts. Apply `replace` by replacing the phase. Request a fresh snapshot on zoom or to re-bin a growing trace. Cursors count records, so duplicate steps remain distinct.

The service indexes complete JSONL lines on demand and keeps the six most recently viewed attempt indexes in memory. File replacement, truncation or index eviction invalidates old cursors. Malformed lines are counted separately. Null and non-finite scalars appear as gaps. For quantitative analysis, paginate `records` or read the original JSONL files.
