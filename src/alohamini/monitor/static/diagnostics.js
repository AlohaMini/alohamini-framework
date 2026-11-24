import { $, api, disclosure, el, fmt, lang, opt, t, table } from "./ui.js";

const valid = Number.isFinite;
function difference(now, before) {
  if (!valid(now) || !valid(before)) return null;
  const d = now - before;
  return d > 0 ? "+" + fmt(d, true) : fmt(d, true);
}
function reading(parent, label, value, note) {
  const cell = el("div", "diag-reading");
  cell.append(el("span", "detail", t(label)), el("strong", null, value));
  if (note) cell.append(el("small", "detail", note));
  parent.append(cell);
}
function windowNote(from, to, n, records) {
  if (!records) return t("noEarlierWindow");
  const count = (n === records ? fmt(n) : fmt(n) + "/" + fmt(records)) + " " + t("observations");
  return t("stepAt") + " " + fmt(from) + "–" + fmt(to) + " · " + count;
}
function title(run, runName, compare) {
  const block = el("div", "diag-run");
  if (compare) block.append(el("h4", null, runName(run)));
  return block;
}
function trend(block, data) {
  const prev = data.previous_window,
    d = data.distribution;
  const head = el("div", "diag-section-heading");
  head.append(el("h3", null, t(data.metric)));
  if (valid(prev?.mean) && d) {
    const delta = d.mean - prev.mean;
    const direction = t(delta > 0 ? "meanRose" : delta < 0 ? "meanFell" : "meanUnchanged");
    const relative =
      prev.mean > 0 && delta ? fmt(Math.abs((100 * delta) / prev.mean), true) + "%" : null;
    head.append(el("span", "diag-change", [direction, relative].filter(Boolean).join(" ")));
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
  block.append(pair);
}
function components(block, data) {
  const entries = Object.entries(data.components);
  if (!entries.length) {
    block.append(el("p", "detail", t("noComponents")));
    return;
  }
  entries.sort(
    ([, a], [, b]) =>
      (valid(b.previous_mean) ? Math.abs(b.mean - b.previous_mean) : -1) -
      (valid(a.previous_mean) ? Math.abs(a.mean - a.previous_mean) : -1),
  );
  table(
    block,
    ["metric", "earlierMean", "currentMean", "change"],
    entries.map(([k, v]) => {
      const name = el("span", null, t(k));
      name.title = k;
      const previous = el("span", null, fmt(v.previous_mean, true)),
        current = el("span", null, fmt(v.mean, true));
      previous.title = fmt(v.previous_n) + " " + t("observations");
      current.title = fmt(v.n) + " " + t("observations");
      return [name, previous, current, difference(v.mean, v.previous_mean)];
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
function validation(block, data, target) {
  const v = data.validation;
  if (!v.latest) {
    block.append(el("p", "detail", t("noValidation")));
    return;
  }
  const pair = el("div", "diag-pair");
  reading(pair, "last", fmt(v.latest.loss), coords(v.latest));
  reading(pair, "lowest", fmt(v.best.loss), coords(v.best));
  block.append(pair);
  const diff = difference(v.latest.loss, v.previous?.loss);
  const notes = [t("latestVal") + " " + fmt(v.count)];
  if (diff != null) notes.push(t("sincePreviousVal") + " " + diff);
  block.append(el("p", "detail", notes.join(" · ")));
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
    const info = el("dl", "diag-facts");
    for (const k of ["generalization_gap", "train_eval_loss", "train_eval_samples", "weights", "step"]) {
      const v = data.paired[k];
      if (v != null) info.append(el("dt", null, t(k)), el("dd", null, valid(v) ? fmt(v) : v));
    }
    block.append(el("h4", null, t("paired")), info);
  }
}
function histogram(parent, d) {
  const hist = el("div", "hist");
  hist.setAttribute("role", "img");
  hist.setAttribute("aria-label", t("distribution") + " · " + d.bins.join(", "));
  const peak = Math.max(...d.bins),
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
  labels.append(el("span", null, fmt(d.min)), el("span", null, fmt(d.max)));
  parent.append(hist, labels);
}
function details(parent, run, data, runName, compare) {
  function section(key) {
    const box = disclosure(parent, (compare ? runName(run) + " · " : "") + t(key));
    box.dataset.slot = run.id + "|" + key;
    return box;
  }
  function facts(parent, entries) {
    const dl = el("dl", "diag-facts");
    for (const [key, value] of entries)
      if (value != null)
        dl.append(el("dt", null, t(key)), el("dd", null, valid(value) ? fmt(value) : value));
    parent.append(dl);
  }
  const d = data.distribution;
  if (d) {
    const box = section("variation");
    box.append(el("p", "detail", t(data.metric) + " · " + t("histCount")));
    histogram(box, d);
    facts(box, [
      ["median", d.p50],
      ["percentile95", d.p95],
      ["sd", d.sd],
      ["peak", d.peak_step],
      ["slope", d.slope_per_step],
    ]);
  }
  const clip = data.clipping;
  const measures = [
    [
      "clipping",
      clip
        ? fmt(clip.count) +
          "/" +
          fmt(clip.n) +
          " (" +
          fmt(100 * clip.fraction, true) +
          "%)" +
          (valid(clip.threshold) ? " · " + t("threshold") + " " + fmt(clip.threshold) : "") +
          (clip.source === "from_norm" ? " · " + t("estimatedClip") : "")
        : null,
    ],
    ["gradP95", data.gradient?.p95],
    ["skipped_updates", data.skipped],
    ["samplesPerS", data.samples_per_s],
    ...Object.entries(data.timing),
    ["gpu_mem_gb", run.phases.train.latest.gpu_mem_gb],
  ];
  const rates = Object.entries(data.rates).filter(([k]) => k !== "lr" || !("lr/0" in data.rates));
  if (rates.length || measures.some(([, v]) => v != null)) {
    const box = section("optimizerTiming");
    facts(box, measures);
    table(
      box,
      ["lr", "first", "last"],
      rates.map(([k, v]) => [t(k), fmt(v.first), fmt(v.last)]),
    );
  }
  const modules = Object.entries(data.modules);
  if (modules.length)
    table(
      section("probe"),
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
  const totals = Object.entries(run.phases).flatMap(([phase, p]) =>
    Object.entries(p.latest.metric_totals || {}).map(([k, v]) => [
      t(phase),
      fmt(p.latest.step),
      t(k),
      fmt(v.sum),
      fmt(v.count),
    ]),
  );
  if (totals.length)
    table(section("rawTotals"), ["phase", "stepAt", "metric", "numerator", "count"], totals);
}

export class Diagnostics {
  constructor(refresh) {
    this.key = "";
    this.options = "";
    for (const id of ["stat-window", "stat-metric", "stat-target"]) $(id).onchange = refresh;
    $("stat-target").onkeydown = (e) => {
      if (e.key === "Enter") e.target.blur();
    };
  }
  async render(runs, runName, signal) {
    const metric = $("stat-metric"),
      width = $("stat-window");
    const keys = [...new Set(runs.flatMap((r) => r.phases.train.metrics))];
    const opts = JSON.stringify([lang, keys]);
    if (this.options !== opts) {
      const selected = metric.value,
        oldWidth = width.value || "100";
      metric.replaceChildren(...keys.map((k) => opt(k, t(k))));
      metric.value = keys.includes(selected) ? selected : keys.includes("loss") ? "loss" : keys[0] || "";
      width.replaceChildren(...[20, 100, 500, 1000].map((n) => opt(n, n + " " + t("recordsUnit"))));
      width.value = oldWidth;
      this.options = opts;
    }
    const target = $("stat-target").value;
    const key = JSON.stringify([
      lang,
      runs.map((r) => [r.id, r.version]),
      width.value,
      metric.value,
      target,
    ]);
    if (this.key === key) return;
    const params = new URLSearchParams({ window: width.value, metric: metric.value || "loss" });
    if (target !== "" && valid(Number(target))) params.set("target", target);
    const results = await Promise.all(
      runs.map((r) => api("runs/" + r.id + "/diagnostics", params, signal)),
    );
    if (signal.aborted) return;
    const deep = $("stat-details"),
      oldFocus = document.activeElement;
    const focused = oldFocus?.matches("summary") ? oldFocus.parentElement.dataset.slot : null;
    const opened = new Set([...deep.querySelectorAll("details[open]")].map((n) => n.dataset.slot));
    const groups = [
      document.createDocumentFragment(),
      document.createDocumentFragment(),
      document.createDocumentFragment(),
    ];
    const extra = document.createDocumentFragment();
    const compare = runs.length > 1;
    results.forEach((data, i) => {
      const run = runs[i];
      [trend, components, validation].forEach((render, j) => {
        const block = title(run, runName, compare);
        render(block, data, target);
        groups[j].append(block);
      });
      details(extra, run, data, runName, compare);
    });
    ["stat-trends", "stat-components", "stat-validation"].forEach((id, j) =>
      $(id).replaceChildren(groups[j]),
    );
    $("stat-loss-panel").hidden = !results.some((d) => Object.keys(d.components).length);
    $("stat-target").parentElement.hidden = !results.some((d) => d.validation.latest);
    deep.replaceChildren(extra);
    for (const node of deep.querySelectorAll("details")) {
      node.open = opened.has(node.dataset.slot);
      if (focused === node.dataset.slot) node.querySelector("summary").focus({ preventScroll: true });
    }
    this.key = key;
  }
}
