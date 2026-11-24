import { $, api, disclosure, el, facts, fmt, lang, opt, t, table } from "./ui.js";

const format = (v) => (Number.isFinite(v) ? fmt(v) : null);
function observation(value) {
  if (!value) return "—";
  return [
    fmt(value.loss),
    t("stepAt") + " " + fmt(value.step),
    value.samples != null ? fmt(value.samples) + " " + t("samples") : null,
    value.elapsed_s != null ? fmt(value.elapsed_s) + " s" : null,
  ]
    .filter(Boolean)
    .join(" · ");
}
function histogram(parent, d) {
  const hist = el("div", "hist");
  hist.setAttribute("role", "img");
  hist.setAttribute("aria-label", t("distribution") + " · " + d.bins.join(", "));
  const peak = Math.max(...d.bins);
  d.bins.forEach((n, i) => {
    const bar = el("span"),
      w = (d.max - d.min) / d.bins.length;
    bar.style.height = (100 * n) / peak + "%";
    bar.title = fmt(d.min + i * w) + " – " + fmt(d.min + (i + 1) * w) + " · " + n;
    hist.append(bar);
  });
  const labels = el("div", "hist-labels");
  labels.append(el("span", null, fmt(d.min)), el("span", null, fmt(d.max)));
  parent.append(hist, labels);
}
function results(run, name, data, target) {
  const block = el("article", "card analysis-block");
  block.append(
    el("h3", null, name + " · " + t("stepAt") + " " + fmt(data.from_step) + " – " + fmt(data.to_step)),
  );
  const columns = el("div", "detail-group"),
    left = el("div"),
    right = el("div");
  columns.append(left, right);
  block.append(columns);
  const d = data.distribution;
  left.append(el("h4", null, t(data.metric) + " · " + t("distribution")));
  if (d) {
    histogram(left, d);
    facts(left, [
      ["records", fmt(d.n)],
      ["mean", fmt(d.mean)],
      ["sd", fmt(d.sd)],
      ["P50", fmt(d.p50)],
      ["P95", fmt(d.p95)],
      ["peak", fmt(d.peak_step)],
      ["slope", format(d.slope_per_step)],
    ]);
  } else left.append(el("p", "detail", t("noData")));
  right.append(el("h4", null, t("efficiency")));
  facts(right, [
    [
      "clipping",
      data.clipping
        ? fmt(100 * data.clipping.fraction) +
          "% (" +
          data.clipping.count +
          "/" +
          data.clipping.n +
          ")" +
          (data.clipping.source === "from_norm" ? " · " + t("estimatedClip") : "")
        : null,
    ],
    ["grad_norm", data.gradient ? "P95 " + fmt(data.gradient.p95) : null],
    ["skipped_updates", format(data.skipped)],
    ["samplesPerS", format(data.samples_per_s)],
    ...Object.entries(data.timing).map(([k, v]) => [k, fmt(v)]),
    ["gpu_mem_gb", format(run.phases.train.latest.gpu_mem_gb)],
  ]);
  if (data.validation.best) {
    right.append(el("h4", null, t("validation")));
    facts(right, [
      ["latestVal", fmt(data.validation.count)],
      ["best", observation(data.validation.best)],
      [
        "firstHit",
        target === ""
          ? null
          : data.validation.first_hit
            ? observation(data.validation.first_hit)
            : t("noHit"),
      ],
    ]);
  }
  if (data.paired) {
    right.append(el("h4", null, t("paired")));
    facts(
      right,
      Object.entries(data.paired).map(([k, v]) => [k, typeof v === "number" ? fmt(v) : v]),
    );
  }
  const entries = Object.entries(data.components);
  if (entries.length) {
    block.append(el("h4", null, t("components")));
    table(
      block,
      ["metric", "prevMean", "mean", "records"],
      entries.map(([k, v]) => [t(k), fmt(v.previous_mean), fmt(v.mean), fmt(v.n)]),
    );
  }
  const rates = Object.entries(data.rates).filter(([k]) => k !== "lr" || !("lr/0" in data.rates));
  if (rates.length) {
    const section = disclosure(
      block,
      t("rates") + " · " + rates.map(([k, v]) => k + " " + fmt(v.last)).join(" / "),
    );
    table(
      section,
      ["metric", "first", "last"],
      rates.map(([k, v]) => [k, fmt(v.first), fmt(v.last)]),
    );
  }
  if (Object.keys(data.modules).length) {
    const section = disclosure(block, t("probe"));
    table(
      section,
      ["module", "stepAt", "weight_norm", "grad_norm", "update_norm", "relative_update", "gradients"],
      Object.entries(data.modules).map(([k, v]) => [
        k,
        fmt(v.step),
        fmt(v.weight_norm),
        fmt(v.grad_norm),
        fmt(v.update_norm),
        fmt(v.relative_update),
        fmt(v.grad_tensors) + " / " + fmt(v.trainable_tensors),
      ]),
    );
  }
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
    table(disclosure(block, t("rawTotals")), ["phase", "stepAt", "metric", "numerator", "count"], totals);
  return block;
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
      window = $("stat-window");
    const keys = [...new Set(runs.flatMap((r) => r.phases.train.metrics))];
    const opts = JSON.stringify([lang, keys]);
    if (this.options !== opts) {
      const selected = metric.value,
        width = window.value || "100";
      metric.replaceChildren(...keys.map((k) => opt(k, t(k))));
      metric.value = keys.includes(selected) ? selected : keys.includes("loss") ? "loss" : keys[0] || "";
      window.replaceChildren(...[20, 100, 500, 1000].map((n) => opt(n, n + " " + t("recordsUnit"))));
      window.value = width;
      this.options = opts;
    }
    const target = $("stat-target").value;
    const key = JSON.stringify([
      lang,
      runs.map((r) => [r.id, r.version]),
      window.value,
      metric.value,
      target,
    ]);
    if (this.key === key) return;
    const params = new URLSearchParams({ window: window.value, metric: metric.value || "loss" });
    if (target !== "" && Number.isFinite(Number(target))) params.set("target", target);
    const data = await Promise.all(runs.map((r) => api("runs/" + r.id + "/diagnostics", params, signal)));
    if (signal.aborted) return;
    const root = $("stat-results");
    const opened = new Set([...root.querySelectorAll("details[open]")].map((n) => n.dataset.slot));
    const blocks = runs.map((r, i) => {
      const block = results(r, runName(r), data[i], target);
      for (const [j, node] of [...block.querySelectorAll("details")].entries()) {
        node.dataset.slot = r.id + "-" + j;
        node.open = opened.has(node.dataset.slot);
      }
      return block;
    });
    root.replaceChildren(...blocks);
    this.key = key;
  }
}
