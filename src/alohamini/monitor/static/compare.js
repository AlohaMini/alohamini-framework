import { $, api, describe, el, explain, fmt, fmtPct, lang, onThemeChange, opt, t } from "./ui.js";
import { paletteColor, Plot } from "./plot.js";

const valid = Number.isFinite;

function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

const HPARAM_FIELDS = [
  ["lr", "colLr"],
  ["effective_batch_size", "colBatch"],
  ["weight_decay", "colWeightDecay"],
  ["grad_clip_norm", "colClip"],
  ["warmup_steps", "colWarmup"],
  ["seed", "colSeed"],
];

const OUTCOME_FIELDS = [
  ["val_loss_best", "colBestVal"],
  ["train_loss_mean", "colTrainMean"],
  ["generalization_gap", "colGap"],
  ["offline_mae", "colMae"],
  ["samples_per_s", "colSpeed"],
  ["elapsed_s", "elapsed_s"],
];

function getVal(row, key) {
  if (key in row) return row[key];
  return row.hparams?.[key];
}

function drawScatter(canvas, rows, xKey, yKey, onSelect) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width,
    h = rect.height;
  const pad = { left: 58, right: 34, top: 16, bottom: 28 };
  const grid = cssVar("--grid", "#e8ebf0");
  const muted = cssVar("--muted", "#5a6270");
  const ink = cssVar("--ink", "#1f242d");

  const pts = rows
    .map((r, i) => ({ row: r, x: getVal(r, xKey), y: getVal(r, yKey), i }))
    .filter((p) => valid(p.x) && valid(p.y));
  ctx.font = '400 11px "Inter", "Ubuntu Sans", sans-serif';
  if (!pts.length) {
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.fillText(t("noData"), w / 2, h / 2);
    canvas._pts = [];
    return;
  }
  let xmin = Math.min(...pts.map((p) => p.x)),
    xmax = Math.max(...pts.map((p) => p.x));
  let ymin = Math.min(...pts.map((p) => p.y)),
    ymax = Math.max(...pts.map((p) => p.y));
  if (xmin === xmax) {
    xmin -= 1;
    xmax += 1;
  }
  if (ymin === ymax) {
    ymin -= 0.1;
    ymax += 0.1;
  }
  const mx = (xmax - xmin) * 0.1,
    my = (ymax - ymin) * 0.12;
  xmin -= mx;
  xmax += mx;
  ymin -= my;
  ymax += my;
  const pw = w - pad.left - pad.right,
    ph = h - pad.top - pad.bottom;
  const X = (v) => pad.left + ((v - xmin) * pw) / (xmax - xmin);
  const Y = (v) => pad.top + ((ymax - v) * ph) / (ymax - ymin);

  ctx.strokeStyle = grid;
  ctx.fillStyle = muted;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const vy = ymin + ((ymax - ymin) * i) / 4;
    const py = Y(vy);
    ctx.beginPath();
    ctx.setLineDash([3, 3]);
    ctx.moveTo(pad.left, py);
    ctx.lineTo(w - pad.right, py);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(fmt(vy, true), pad.left - 6, py);

    const vx = xmin + ((xmax - xmin) * i) / 4;
    const px = X(vx);
    ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    ctx.textBaseline = "top";
    ctx.fillText(fmt(vx, true), px, h - pad.bottom + 6);
  }

  ctx.setLineDash([]);
  const plotted = [];
  for (const p of pts) {
    const cx = X(p.x),
      cy = Y(p.y);
    const color = paletteColor(p.i);
    ctx.beginPath();
    ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = cssVar("--panel", "#ffffff");
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.font = '400 11px "Inter", "Ubuntu Sans", "Noto Sans CJK SC", sans-serif';
    ctx.textBaseline = "middle";
    if (pts.length <= 8) {
      const label = p.row.name.length > 24 ? p.row.name.slice(0, 22) + "…" : p.row.name;
      if (cx > w * 0.62) {
        ctx.textAlign = "right";
        ctx.fillText(label, cx - 9, cy);
      } else {
        ctx.textAlign = "left";
        ctx.fillText(label, cx + 9, cy);
      }
    }
    plotted.push({ ...p, cx, cy });
  }
  canvas._pts = plotted;
  canvas.onclick = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left,
      y = e.clientY - r.top;
    const hit = plotted.find((p) => Math.hypot(p.cx - x, p.cy - y) <= 10);
    if (hit && onSelect) onSelect(hit.row.id);
  };
}

function drawParallel(canvas, rows, outcomeKey) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width,
    h = rect.height;
  const muted = cssVar("--muted", "#5a6270");
  const line = cssVar("--line", "#dfe2e8");

  const axes = [
    ...HPARAM_FIELDS.filter(([k]) => rows.some((r) => valid(getVal(r, k)))),
    [outcomeKey, OUTCOME_FIELDS.find(([k]) => k === outcomeKey)?.[1] || outcomeKey],
  ];
  const active = axes.filter(([k]) => rows.some((r) => valid(getVal(r, k))));
  ctx.font = '400 11px "Inter", "Ubuntu Sans", sans-serif';
  if (active.length < 2 || rows.length < 1) {
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.fillText(t("noData"), w / 2, h / 2);
    return;
  }

  const pad = { left: 48, right: 48, top: 34, bottom: 26 };
  const pw = w - pad.left - pad.right,
    ph = h - pad.top - pad.bottom;
  const ranges = active.map(([k]) => {
    const vals = rows.map((r) => getVal(r, k)).filter(valid);
    let lo = Math.min(...vals),
      hi = Math.max(...vals);
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    return [lo, hi];
  });

  // Draw vertical axes
  active.forEach(([, labelKey], idx) => {
    const x = pad.left + (pw * idx) / (active.length - 1);
    ctx.strokeStyle = line;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, h - pad.bottom);
    ctx.stroke();
    ctx.fillStyle = muted;
    ctx.font = '500 11px "Inter", "Ubuntu Sans", "Noto Sans CJK SC", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(t(labelKey), x, pad.top - 15);
    ctx.font = '400 10.5px "Inter", "Ubuntu Sans", sans-serif';
    ctx.textBaseline = "bottom";
    ctx.fillText(fmt(ranges[idx][1], true), x, pad.top - 3);
    ctx.textBaseline = "top";
    ctx.fillText(fmt(ranges[idx][0], true), x, h - pad.bottom + 5);
  });

  // Draw each run polyline
  rows.forEach((r, rowIdx) => {
    ctx.beginPath();
    ctx.strokeStyle = paletteColor(rowIdx);
    ctx.lineWidth = 2;
    let move = true;
    active.forEach(([k], idx) => {
      const v = getVal(r, k);
      if (!valid(v)) {
        move = true;
        return;
      }
      const x = pad.left + (pw * idx) / (active.length - 1);
      const [lo, hi] = ranges[idx];
      const y = pad.top + ((hi - v) * ph) / (hi - lo);
      if (move) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      move = false;
    });
    ctx.stroke();
  });
}

function aggregateSeedBand(seriesList) {
  // Interpolate multiple seed runs onto a shared grid and compute Mean ± 1 SD
  const validSeries = seriesList.filter((pts) => pts && pts.length >= 2);
  if (validSeries.length < 2) return null;
  const lo = Math.max(...validSeries.map((pts) => pts[0][0]));
  const hi = Math.min(...validSeries.map((pts) => pts[pts.length - 1][0]));
  if (!(hi > lo)) return null;
  const steps = 160;
  const meanPts = [];
  const bandPts = [];
  for (let i = 0; i <= steps; i++) {
    const x = lo + ((hi - lo) * i) / steps;
    const ys = [];
    for (const pts of validSeries) {
      // Binary search interpolation
      let l = 0,
        r = pts.length - 1;
      while (l < r) {
        const m = (l + r) >>> 1;
        if (pts[m][0] < x) l = m + 1;
        else r = m;
      }
      const p1 = pts[l],
        p0 = l > 0 ? pts[l - 1] : p1;
      if (p0[1] == null || p1[1] == null) continue;
      const denom = p1[0] - p0[0];
      const y = denom > 0 ? p0[1] + ((x - p0[0]) / denom) * (p1[1] - p0[1]) : p1[1];
      if (valid(y)) ys.push(y);
    }
    if (ys.length >= 2) {
      const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
      const sd = Math.sqrt(ys.reduce((a, b) => a + (b - mean) ** 2, 0) / ys.length);
      meanPts.push([x, mean]);
      bandPts.push([x, mean - sd, mean + sd]);
    }
  }
  return meanPts.length ? { meanPts, bandPts } : null;
}

export class CompareView {
  constructor({ onSelectRun, onToggleCompare }) {
    this.onSelectRun = onSelectRun;
    this.onToggleCompare = onToggleCompare;
    this.sortCol = "val_loss_best";
    this.sortAsc = true;
    this.xParam = "lr";
    this.yOutcome = "val_loss_best";
    this.key = "";
    this.bandPlots = [];
    this.unsubTheme = onThemeChange(() => {
      if (this.lastRows) this.redrawCanvases();
    });
  }

  clearPlots() {
    for (const p of this.bandPlots) p.destroy();
    this.bandPlots = [];
  }

  redrawCanvases() {
    if (this.scatterCanvas && this.lastRows) {
      drawScatter(this.scatterCanvas, this.lastRows, this.xParam, this.yOutcome, this.onSelectRun);
    }
    if (this.parallelCanvas && this.lastRows) {
      drawParallel(this.parallelCanvas, this.lastRows, this.yOutcome);
    }
  }

  async render(selectedIds, signal, { stitch = true } = {}) {
    const showAll = $("compare-all-attempts").checked;
    const showBand = $("compare-seed-band").checked;
    const data = await api("compare", {}, signal);
    if (signal.aborted) return;

    let rows = (data.runs || []).filter((r) => showAll || r.is_latest_in_group);
    rows.sort((a, b) => {
      const va = getVal(a, this.sortCol),
        vb = getVal(b, this.sortCol);
      const aOk = va != null && va !== "",
        bOk = vb != null && vb !== "";
      if (aOk !== bOk) return aOk ? -1 : 1;
      if (!aOk) return 0;
      const cmp = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
      return this.sortAsc ? cmp : -cmp;
    });
    this.lastRows = rows;

    const root = $("compare-content");
    this.clearPlots();
    root.replaceChildren();

    // 1. Multi-seed Mean ± Std Aggregation Panel (when toggled)
    if (showBand) {
      const bandPanel = el("section", "diag-panel");
      bandPanel.style.marginBottom = "12px";
      const bh = el("h3", null, t("seedBand"));
      describe(bh, explain("seed"));
      bandPanel.append(bh);
      if (selectedIds.length < 2) {
        bandPanel.append(el("p", "detail", t("seedBandHint")));
      } else {
        const seriesResp = await Promise.all(
          selectedIds.map((id) =>
            api(
              "runs/" + id + "/series",
              { metric: "loss", axis: "step", smooth: "0.4", stitch: stitch ? "1" : "0", points: "400" },
              signal,
            ),
          ),
        );
        if (signal.aborted) return;
        const trainAgg = aggregateSeedBand(
          seriesResp.map((r) => r.train?.smoothed?.loss?.points || r.train?.series?.loss?.points),
        );
        const valAgg = aggregateSeedBand(seriesResp.map((r) => r.validation?.series?.loss?.points));
        if (!trainAgg && !valAgg) {
          bandPanel.append(el("p", "detail", t("noData")));
        } else {
          const p = new Plot(bandPanel);
          this.bandPlots.push(p);
          const lines = [],
            bands = [];
          if (trainAgg) {
            const c = paletteColor(0);
            lines.push({
              id: "train-mean",
              label: `${t("train")} Mean (n=${selectedIds.length})`,
              color: c,
              trend: true,
              count: trainAgg.meanPts.length,
              points: trainAgg.meanPts,
            });
            bands.push({ color: c, points: trainAgg.bandPts });
          }
          if (valAgg) {
            const c = paletteColor(2);
            lines.push({
              id: "val-mean",
              label: `${t("validation")} Mean (n=${selectedIds.length})`,
              color: c,
              dash: true,
              count: valAgg.meanPts.length,
              points: valAgg.meanPts,
            });
            bands.push({ color: c, points: valAgg.bandPts });
          }
          p.update({
            title: `${t("loss")} · Mean ± 1 SD`,
            description: explain("loss"),
            axis: "step",
            lines,
            bands,
            accent: paletteColor(0),
          });
        }
      }
      root.append(bandPanel);
    }

    // 2. Sortable HParams Experiment Matrix Table
    const wrap = el("div", "table-wrap");
    const tbl = el("table", "compare-table");
    const columns = [
      ["", ""],
      ["name", "colRun"],
      ["policy", "colPolicy"],
      ["status", "colStatus"],
      ["progress", "colProgress"],
      ["val_loss_best", "colBestVal"],
      ["train_loss_mean", "colTrainMean"],
      ["generalization_gap", "colGap"],
      ["lr", "colLr"],
      ["effective_batch_size", "colBatch"],
      ["grad_clip_norm", "colClip"],
      ["seed", "colSeed"],
      ["samples_per_s", "colSpeed"],
      ["offline_mae", "colMae"],
    ];
    const headRow = tbl.createTHead().insertRow();
    for (const [col, labelKey] of columns) {
      const th = el("th", col ? "sortable" : null, labelKey ? t(labelKey) : "");
      if (col) {
        if (this.sortCol === col) th.textContent += this.sortAsc ? " ↑" : " ↓";
        th.onclick = () => {
          if (this.sortCol === col) this.sortAsc = !this.sortAsc;
          else {
            this.sortCol = col;
            this.sortAsc = col !== "samples_per_s" && col !== "progress";
          }
          this.render(selectedIds, new AbortController().signal, { stitch });
        };
      }
      headRow.append(th);
    }
    const body = tbl.createTBody();
    for (const [idx, r] of rows.entries()) {
      const tr = body.insertRow();
      tr.className = "clickable";
      const selIdx = selectedIds.indexOf(r.id);

      const checkCell = tr.insertCell();
      const box = el("input");
      box.type = "checkbox";
      box.checked = selIdx >= 0;
      box.onclick = (e) => {
        e.stopPropagation();
        this.onToggleCompare(r.id);
      };
      checkCell.append(box);

      const nameCell = tr.insertCell();
      nameCell.className = "run-col";
      const dot = el("i", "plot-tip-dot");
      dot.style.background = selIdx >= 0 ? paletteColor(selIdx) : paletteColor(idx);
      dot.style.display = "inline-block";
      dot.style.marginRight = "6px";
      const strong = el("strong");
      for (const part of String(r.name || "").split(/([_-])/)) {
        strong.append(document.createTextNode(part));
        if (part === "_" || part === "-") strong.append(document.createElement("wbr"));
      }
      nameCell.append(dot, strong);
      if (r.attempt) nameCell.append(" ", el("span", "detail", `#${r.ordinal}`));
      tr.onclick = () => this.onSelectRun(r.id);

      tr.insertCell().textContent = r.policy || "—";
      const stCell = tr.insertCell();
      const stClass = r.status === "running" ? "ok" : r.status === "completed" ? "info" : "";
      stCell.append(
        el(
          "span",
          "tag " + stClass,
          t(r.status === "running" ? "statusRunning" : r.status === "completed" ? "statusCompleted" : "statusIdle"),
        ),
      );

      const progCell = tr.insertCell();
      if (valid(r.progress)) {
        const bar = el("span", "inline-bar");
        const fill = el("i");
        fill.style.width = `${Math.min(100, r.progress * 100)}%`;
        bar.append(fill);
        progCell.append(bar, el("span", "number", `${fmt(r.latest_step, true)}/${fmt(r.steps, true)}`));
      } else {
        progCell.textContent = valid(r.latest_step) ? fmt(r.latest_step) : "—";
      }

      const bestValCell = tr.insertCell();
      bestValCell.className = "number";
      bestValCell.textContent = valid(r.val_loss_best)
        ? `${fmt(r.val_loss_best, true)} @${fmt(r.val_best_step, true)}`
        : "—";

      const cells = [
        valid(r.train_loss_mean) ? fmt(r.train_loss_mean, true) : "—",
        valid(r.generalization_gap) ? (r.generalization_gap > 0 ? "+" : "") + fmt(r.generalization_gap, true) : "—",
        valid(r.hparams?.lr) ? fmt(r.hparams.lr, true) : "—",
        valid(r.hparams?.effective_batch_size) ? fmt(r.hparams.effective_batch_size) : "—",
        valid(r.hparams?.grad_clip_norm) ? fmt(r.hparams.grad_clip_norm, true) : "—",
        valid(r.hparams?.seed) ? String(r.hparams.seed) : "—",
        valid(r.samples_per_s) ? fmt(r.samples_per_s, true) : "—",
        valid(r.offline_mae) ? fmt(r.offline_mae, true) : "—",
      ];
      for (const txt of cells) {
        const td = tr.insertCell();
        td.className = "number";
        td.textContent = txt;
      }
    }
    wrap.append(tbl);
    root.append(wrap);

    // 3. Scatter + Parallel Coordinates Grid
    const grid = el("div", "compare-grid");
    const scCard = el("section", "diag-panel");
    const scHead = el("div", "diag-section-heading");
    scHead.append(el("h3", null, t("scatterTitle")));
    const scControls = el("div", "controls");
    const xSel = el("select");
    HPARAM_FIELDS.forEach(([k, lbl]) => xSel.add(opt(k, t(lbl))));
    xSel.value = this.xParam;
    const ySel = el("select");
    OUTCOME_FIELDS.forEach(([k, lbl]) => ySel.add(opt(k, t(lbl))));
    ySel.value = this.yOutcome;
    scControls.append(xSel, ySel);
    scHead.append(scControls);
    const scWrap = el("div", "compare-canvas-wrap");
    this.scatterCanvas = el("canvas");
    scWrap.append(this.scatterCanvas);
    scCard.append(scHead, scWrap);

    const pcCard = el("section", "diag-panel");
    pcCard.append(el("h3", null, t("parallelTitle")));
    const pcWrap = el("div", "compare-canvas-wrap");
    this.parallelCanvas = el("canvas");
    pcWrap.append(this.parallelCanvas);
    pcCard.append(pcWrap);

    grid.append(scCard, pcCard);
    root.append(grid);

    xSel.onchange = () => {
      this.xParam = xSel.value;
      this.redrawCanvases();
    };
    ySel.onchange = () => {
      this.yOutcome = ySel.value;
      this.redrawCanvases();
    };
    requestAnimationFrame(() => this.redrawCanvases());
  }
}
