import { $, api, btn, describe, disclosure, el, explain, fmt, fmtPct, lang, opt, t, table } from "./ui.js";
import { paletteColor } from "./plot.js";

const valid = Number.isFinite;

function difference(now, before) {
  if (!valid(now) || !valid(before)) return null;
  const d = now - before;
  return d > 0 ? "+" + fmt(d, true) : fmt(d, true);
}

function reading(parent, labelKey, value, note, helpKey = null) {
  const cell = el("div", "diag-reading");
  const lbl = el("span", "detail", t(labelKey));
  if (helpKey || explain(labelKey)) describe(lbl, explain(helpKey || labelKey));
  cell.append(lbl, el("strong", null, value));
  if (note) cell.append(el("small", "detail", note));
  parent.append(cell);
}

function windowNote(from, to, n, records) {
  if (!records) return t("noEarlierWindow");
  const count = (n === records ? fmt(n) : fmt(n) + "/" + fmt(records)) + " " + t("observations");
  return t("stepAt") + " " + fmt(from) + "–" + fmt(to) + " · " + count;
}

function alertMessage(alert) {
  const d = alert.data || {};
  const zh = lang === "zh";
  switch (alert.code) {
    case "overfitting":
      return zh
        ? `过拟合信号：验证损失较最佳值（步 ${fmt(d.best_step)}）上升 +${fmt(d.rise_pct, true)}%（${fmt(d.best_loss, true)} → ${fmt(d.latest_loss, true)}）`
        : `Overfitting signal: val loss rose +${fmt(d.rise_pct, true)}% since best @${fmt(d.best_step)} (${fmt(d.best_loss, true)} → ${fmt(d.latest_loss, true)})`;
    case "plateau":
      return zh
        ? `进入平台期：后 1/3 阶段均值较中段变化 ${fmt(d.change_pct, true)}%（${fmt(d.mid_mean, true)} → ${fmt(d.late_mean, true)}）`
        : `Plateau: late third changed ${fmt(d.change_pct, true)}% vs mid third (${fmt(d.mid_mean, true)} → ${fmt(d.late_mean, true)})`;
    case "heavy_clipping":
      return zh
        ? `梯度频繁裁剪：${fmt(d.fraction_pct, true)}% 的更新触及阈值 C = ${fmt(d.threshold, true)}`
        : `Frequent clipping: ${fmt(d.fraction_pct, true)}% of updates hit threshold C = ${fmt(d.threshold, true)}`;
    case "amp_skips":
      return zh
        ? `AMP 梯度溢出：本区间跳过 ${fmt(d.skipped)} 次更新`
        : `AMP overflow: ${fmt(d.skipped)} optimizer updates skipped in slice`;
    case "data_bottleneck":
      return zh
        ? `数据加载瓶颈：数据等待占单步耗时 ${fmt(d.wait_pct, true)}%`
        : `Data bottleneck: DataLoader wait takes ${fmt(d.wait_pct, true)}% of step time`;
    case "nonfinite":
      return zh
        ? `数值异常：检测到 ${fmt(d.count)} 个非有限（NaN/Inf）记录`
        : `Numeric anomaly: ${fmt(d.count)} non-finite (NaN/Inf) values observed`;
    case "module_disconnected":
      return zh
        ? `模块无梯度：${d.module} 当前 0 个张量收到梯度`
        : `Disconnected module: ${d.module} has 0 tensors receiving gradients`;
    case "module_tiny_update":
      return zh
        ? `模块更新极微弱：${d.module} 更新／参数比 ${fmt(d.relative_update, true)}`
        : `Stalled module update: ${d.module} relative update ${fmt(d.relative_update, true)}`;
    case "healthy_descent":
      return zh
        ? `收敛良好：后期均值较前期下降 ${fmt(d.drop_pct, true)}%，未见显著失稳或过拟合`
        : `Healthy descent: late mean dropped ${fmt(d.drop_pct, true)}% vs early with no instability flags`;
    default:
      return alert.code;
  }
}

function titleBlock(run, runName, compare) {
  const block = el("div", "diag-run");
  if (compare) block.append(el("h4", null, runName(run)));
  return block;
}

function renderTrendAndStages(block, data) {
  const prev = data.previous_window,
    d = data.distribution;
  const head = el("div", "diag-section-heading");
  const h = el("h3", null, t(data.metric));
  describe(h, explain(data.metric));
  head.append(h);
  if (valid(prev?.mean) && d) {
    const delta = d.mean - prev.mean;
    const direction = t(delta > 0 ? "meanRose" : delta < 0 ? "meanFell" : "meanUnchanged");
    const relative =
      prev.mean > 0 && delta ? fmt(Math.abs((100 * delta) / prev.mean), true) + "%" : null;
    const tagCls = delta < 0 ? "ok" : delta > 0 ? "warn" : "";
    head.append(el("span", "tag " + tagCls, [direction, relative].filter(Boolean).join(" ")));
  }
  block.append(head);

  const pair = el("div", "diag-pair");
  reading(
    pair,
    "previousWindow",
    fmt(prev?.mean),
    windowNote(prev?.from_step, prev?.to_step, prev?.n, prev?.records),
  );
  reading(
    pair,
    "currentWindow",
    fmt(d?.mean),
    data.records
      ? windowNote(data.from_step, data.to_step, d?.n || 0, data.records)
      : t("noObservations"),
  );
  if (d) {
    reading(pair, "median", fmt(d.p50), `P95 ${fmt(d.p95, true)} · SD ${fmt(d.sd, true)}`, "median");
  }
  block.append(pair);

  if (data.stages?.length >= 2) {
    block.append(el("h4", null, t("stageEvolution")));
    table(
      block,
      ["phase", "stepAt", "mean", "median", "percentile95", "sd"],
      data.stages.map((s) => [
        t("stage_" + s.stage),
        `${fmt(s.from_step)}–${fmt(s.to_step)}`,
        fmt(s.mean, true),
        fmt(s.p50, true),
        fmt(s.p95, true),
        fmt(s.sd, true),
      ]),
    );
  }
}

function renderComponents(block, data) {
  const entries = Object.entries(data.components || {});
  if (!entries.length) {
    block.append(el("p", "detail", t("noComponents")));
    return;
  }
  entries.sort(([, a], [, b]) => (b.share_pct ?? b.mean ?? 0) - (a.share_pct ?? a.mean ?? 0));

  // Stacked contribution share bar when shares are available
  const shareItems = entries.filter(([, v]) => valid(v.share_pct) && v.share_pct > 0);
  if (shareItems.length >= 2) {
    const bar = el("div", "share-bar");
    const totalShare = shareItems.reduce((acc, [, v]) => acc + v.share_pct, 0) || 100;
    shareItems.forEach(([k, v], idx) => {
      const seg = el("span");
      seg.style.width = `${(v.share_pct / totalShare) * 100}%`;
      seg.style.background = paletteColor(idx);
      seg.title = `${t(k)}: ${fmtPct(v.share_pct)}`;
      bar.append(seg);
    });
    block.append(bar);
  }

  table(
    block,
    ["metric", "sharePct", "earlierMean", "currentMean", "change"],
    entries.map(([k, v], idx) => {
      const name = el("span", null, t(k));
      name.title = k;
      describe(name, explain(k));
      const shareCell = el("span");
      if (valid(v.share_pct)) {
        const dot = el("i", "plot-tip-dot");
        dot.style.background = paletteColor(idx);
        dot.style.display = "inline-block";
        dot.style.marginRight = "6px";
        shareCell.append(dot, document.createTextNode(fmtPct(v.share_pct)));
      } else shareCell.textContent = "—";
      const previous = el("span", null, fmt(v.previous_mean, true)),
        current = el("span", null, fmt(v.mean, true));
      previous.title = fmt(v.previous_n) + " " + t("observations");
      current.title = fmt(v.n) + " " + t("observations");
      return [name, shareCell, previous, current, difference(v.mean, v.previous_mean)];
    }),
  );
}

function coords(point) {
  return [
    t("stepAt") + " " + fmt(point.step),
    valid(point.samples) ? fmt(point.samples) + " " + t("sampleUnit") : null,
    valid(point.elapsed_s) ? fmt(point.elapsed_s, true) + " s" : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function renderValidation(block, data, target, onJump) {
  const v = data.validation;
  if (!v.latest) {
    block.append(el("p", "detail", t("noValidation")));
    return;
  }
  const pair = el("div", "diag-pair");
  reading(pair, "last", fmt(v.latest.loss), coords(v.latest), "valLoss");
  reading(pair, "lowest", fmt(v.best.loss), coords(v.best));
  block.append(pair);
  const diff = difference(v.latest.loss, v.previous?.loss);
  const notes = [t("latestVal") + " " + fmt(v.count)];
  if (diff != null) notes.push(t("sincePreviousVal") + " " + diff);
  const noteRow = el("p", "detail", notes.join(" · "));
  if (onJump && valid(v.best?.step)) {
    const jump = btn(t("jumpToStep") + " @" + fmt(v.best.step, true), () => onJump("loss", v.best.step), "alert-jump");
    noteRow.append(" ", jump);
  }
  block.append(noteRow);
  if (target !== "")
    block.append(
      el(
        "p",
        "target-result",
        v.first_hit
          ? t("firstHit") + " ≤ " + fmt(Number(target)) + " · " + coords(v.first_hit)
          : t("noHit"),
      ),
    );
  if (data.paired) {
    const h = el("h4", null, t("paired"));
    describe(h, explain("generalization_gap"));
    const info = el("dl", "diag-facts");
    for (const k of ["generalization_gap", "train_eval_loss", "train_eval_samples", "weights", "step"]) {
      const val = data.paired[k];
      if (val != null) {
        const dt = el("dt", null, t(k));
        if (explain(k)) describe(dt, explain(k));
        info.append(dt, el("dd", null, valid(val) ? fmt(val) : val));
      }
    }
    block.append(h, info);
  } else if (data.derived_gap?.latest) {
    const g = data.derived_gap;
    const h = el("h4", null, t("approxGapNote"));
    describe(h, explain("derived/generalization_gap"));
    const info = el("dl", "diag-facts");
    info.append(
      el("dt", null, t("last")),
      el("dd", null, `${g.latest.gap > 0 ? "+" : ""}${fmt(g.latest.gap, true)} (@${fmt(g.latest.step)})`),
    );
    if (g.delta != null) {
      info.append(
        el("dt", null, t("change")),
        el("dd", null, `${g.delta > 0 ? "+" : ""}${fmt(g.delta, true)} (${fmt(g.earliest.step)} → ${fmt(g.latest.step)})`),
      );
    }
    block.append(h, info);
  }
}

function histogram(parent, d) {
  const hist = el("div", "hist");
  hist.setAttribute("role", "img");
  hist.setAttribute("aria-label", t("distribution") + " · " + d.bins.join(", "));
  const peak = Math.max(...d.bins, 1),
    width = (d.max - d.min) / d.bins.length;
  d.bins.forEach((n, i) => {
    const bar = el("span");
    bar.style.height = (100 * n) / peak + "%";
    bar.title =
      fmt(d.min + i * width) +
      " – " +
      fmt(d.min + (i + 1) * width) +
      " · " +
      n +
      " " +
      t("observations");
    hist.append(bar);
  });
  const labels = el("div", "hist-labels");
  labels.append(
    el("span", null, "min " + fmt(d.min, true)),
    el("span", null, "P50 " + fmt(d.p50, true)),
    el("span", null, "max " + fmt(d.max, true)),
  );
  parent.append(hist, labels);
}

function factsList(parent, entries) {
  const dl = el("dl", "diag-facts");
  for (const [key, value, helpKey] of entries) {
    if (value == null) continue;
    const dt = el("dt", null, t(key));
    const h = explain(helpKey || key);
    if (h) describe(dt, h);
    const dd = el("dd");
    if (value instanceof Node) dd.append(value);
    else dd.textContent = valid(value) ? fmt(value) : value;
    dl.append(dt, dd);
  }
  parent.append(dl);
}

export class Diagnostics {
  constructor(refresh, onJump) {
    this.key = "";
    this.options = "";
    this.onJump = onJump;
    for (const id of ["stat-scope", "stat-window", "stat-metric", "stat-target"]) {
      $(id).onchange = refresh;
    }
    $("stat-target").onkeydown = (e) => {
      if (e.key === "Enter") e.target.blur();
    };
  }

  async render(runs, runName, signal, { axis = "step", range = null, stitch = false } = {}) {
    const scope = $("stat-scope"),
      metric = $("stat-metric"),
      width = $("stat-window");
    const keys = [...new Set(runs.flatMap((r) => r.phases.train.metrics))];
    const opts = JSON.stringify([lang, keys, Boolean(range)]);
    if (this.options !== opts) {
      const oldScope = scope.value || "tail",
        selected = metric.value,
        oldWidth = width.value || "100";
      scope.replaceChildren(
        opt("tail", t("scope_tail")),
        ...(range ? [opt("range", t("scope_range"))] : []),
      );
      scope.value = range && oldScope === "range" ? "range" : "tail";
      metric.replaceChildren(...keys.map((k) => opt(k, t(k))));
      metric.value = keys.includes(selected) ? selected : keys.includes("loss") ? "loss" : keys[0] || "";
      width.replaceChildren(...[20, 100, 500, 1000].map((n) => opt(n, n + " " + t("recordsUnit"))));
      width.value = oldWidth;
      this.options = opts;
    }
    const useRange = scope.value === "range" && range != null;
    $("stat-window-wrap").hidden = useRange;
    const target = $("stat-target").value;
    const key = JSON.stringify([
      lang,
      runs.map((r) => [r.id, r.version]),
      scope.value,
      useRange ? [axis, range] : width.value,
      metric.value,
      target,
      stitch,
    ]);
    if (this.key === key) return;
    const params = new URLSearchParams({
      window: width.value,
      metric: metric.value || "loss",
      stitch: stitch ? "1" : "0",
    });
    if (useRange) {
      params.set("axis", axis);
      params.set("min", range[0]);
      params.set("max", range[1]);
    }
    if (target !== "" && valid(Number(target))) params.set("target", target);
    const results = await Promise.all(
      runs.map((r) => api("runs/" + r.id + "/diagnostics", params, signal)),
    );
    if (signal.aborted) return;

    // 1. Top Automated Health Alerts
    const alertBar = $("stat-alerts");
    alertBar.replaceChildren();
    const compare = runs.length > 1;
    results.forEach((data, idx) => {
      const run = runs[idx];
      for (const alert of data.alerts || []) {
        const card = el("div", "alert-card " + (alert.severity || "info"));
        const prefix = compare ? `${runName(run)} · ` : "";
        card.append(el("span", null, prefix + alertMessage(alert)));
        if (this.onJump && valid(alert.step)) {
          const jumpMetric =
            alert.code === "heavy_clipping"
              ? "grad_norm"
              : alert.code === "data_bottleneck"
                ? "derived/data_wait_ratio"
                : "loss";
          card.append(
            btn(t("jumpToStep") + " @" + fmt(alert.step, true), () => this.onJump(jumpMetric, alert.step), "alert-jump"),
          );
        }
        alertBar.append(card);
      }
    });
    alertBar.hidden = !alertBar.childElementCount;

    // 2. Main Trend + Stages, Loss Share, Validation
    const groups = [
      document.createDocumentFragment(),
      document.createDocumentFragment(),
      document.createDocumentFragment(),
    ];
    const extraGrid = $("stat-extra-grid");
    extraGrid.replaceChildren();
    const deep = $("stat-details");
    const opened = new Set([...deep.querySelectorAll("details[open]")].map((n) => n.dataset.slot));
    const deepFrag = document.createDocumentFragment();

    results.forEach((data, i) => {
      const run = runs[i];
      const b0 = titleBlock(run, runName, compare);
      renderTrendAndStages(b0, data);
      groups[0].append(b0);

      const b1 = titleBlock(run, runName, compare);
      renderComponents(b1, data);
      groups[1].append(b1);

      const b2 = titleBlock(run, runName, compare);
      renderValidation(b2, data, target, this.onJump);
      groups[2].append(b2);

      // Unfolded Card A: Distribution & Tail Spread
      const d = data.distribution;
      if (d) {
        const distCard = el("section", "diag-panel");
        const h = el("h3", null, (compare ? runName(run) + " · " : "") + t("variation"));
        describe(h, explain("percentile95"));
        distCard.append(h, el("p", "detail", t(data.metric) + " · " + t("histCount")));
        histogram(distCard, d);
        const peakCell = el("span");
        peakCell.append(document.createTextNode(fmt(d.peak_step)));
        if (this.onJump && valid(d.peak_step)) {
          peakCell.append(
            " ",
            btn(t("jumpToStep"), () => this.onJump(data.metric, d.peak_step), "alert-jump"),
          );
        }
        factsList(distCard, [
          ["median", d.p50, "median"],
          ["percentile90", d.p90, "percentile95"],
          ["percentile95", d.p95, "percentile95"],
          ["sd", d.sd],
          ["peak", peakCell],
          ["slope", d.slope_per_step, "slope"],
        ]);
        extraGrid.append(distCard);
      }

      // Unfolded Card B: Gradients, Clipping, LR & Throughput
      const clip = data.clipping;
      const optCard = el("section", "diag-panel");
      const oh = el("h3", null, (compare ? runName(run) + " · " : "") + t("optimizerTiming"));
      describe(oh, explain("grad_norm"));
      optCard.append(oh);
      const waitCell =
        data.data_wait_ratio != null
          ? (() => {
              const wrap = el("span");
              const bar = el("span", "inline-bar");
              const fill = el("i");
              fill.style.width = `${Math.min(100, data.data_wait_ratio * 100)}%`;
              bar.append(fill);
              wrap.append(bar, document.createTextNode(fmtPct(data.data_wait_ratio * 100)));
              return wrap;
            })()
          : null;
      factsList(optCard, [
        [
          "clipping",
          clip
            ? `${fmt(clip.count)}/${fmt(clip.n)} (${fmt(100 * clip.fraction, true)}%)` +
              (valid(clip.threshold) ? ` · ${t("threshold")} ${fmt(clip.threshold)}` : "") +
              (clip.source === "from_norm" ? ` · ${t("estimatedClip")}` : "")
            : null,
          "clipping",
        ],
        ["gradP95", data.gradient?.p95, "grad_norm"],
        ["skipped_updates", data.skipped, "skipped_updates"],
        ["samplesPerS", data.samples_per_s, "derived/samples_per_s"],
        ["dataWait", waitCell, "derived/data_wait_ratio"],
        ...Object.entries(data.timing || {}).map(([k, v]) => [k, v, k]),
        ["gpu_mem_gb", run.phases.train.latest.gpu_mem_gb, "gpu_mem_gb"],
      ]);
      const rates = Object.entries(data.rates || {}).filter(([k]) => k !== "lr" || !("lr/0" in data.rates));
      if (rates.length) {
        table(
          optCard,
          ["lr", "first", "last"],
          rates.map(([k, v]) => [t(k), fmt(v.first), fmt(v.last)]),
        );
      }
      extraGrid.append(optCard);

      // Module update probes (unfolded panel when present!)
      const modules = Object.entries(data.modules || {});
      if (modules.length) {
        const modCard = el("section", "diag-panel");
        const mh = el("h3", null, (compare ? runName(run) + " · " : "") + t("probe"));
        describe(mh, explain("relative_update"));
        modCard.append(mh);
        table(
          modCard,
          ["module", "stepAt", "weight_norm", "grad_norm", "update_norm", "relative_update", "gradients"],
          modules.map(([k, v]) => [
            k,
            fmt(v.step),
            fmt(v.weight_norm),
            fmt(v.grad_norm),
            fmt(v.update_norm),
            fmt(v.relative_update),
            fmt(v.grad_tensors) + " / " + fmt(v.trainable_tensors),
          ]),
        );
        extraGrid.append(modCard);
      }

      // Raw component totals stay in clean collapsible footer
      const totals = Object.entries(run.phases).flatMap(([phase, p]) =>
        Object.entries(p.latest.metric_totals || {}).map(([k, v]) => [
          t(phase),
          fmt(p.latest.step),
          t(k),
          fmt(v.sum),
          fmt(v.count),
        ]),
      );
      if (totals.length) {
        const box = disclosure(deepFrag, (compare ? runName(run) + " · " : "") + t("rawTotals"));
        box.dataset.slot = run.id + "|rawTotals";
        table(box, ["phase", "stepAt", "metric", "numerator", "count"], totals);
      }
    });

    ["stat-trends", "stat-components", "stat-validation"].forEach((id, j) =>
      $(id).replaceChildren(groups[j]),
    );
    $("stat-loss-panel").hidden = !results.some((d) => Object.keys(d.components || {}).length);
    $("stat-target").parentElement.hidden = !results.some((d) => d.validation.latest);
    deep.replaceChildren(deepFrag);
    for (const node of deep.querySelectorAll("details")) {
      node.open = opened.has(node.dataset.slot);
    }
    this.key = key;
  }
}
