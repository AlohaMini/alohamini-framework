import {
  $,
  api,
  btn,
  el,
  explain,
  fmt,
  hint,
  lang,
  modal,
  opt,
  showError,
  t,
  toggleLang,
  translate,
} from "./ui.js";
import { accents, colors, Plot } from "./plot.js";
import { Evaluation } from "./evaluation.js";

let runs = [],
  selected = [],
  tab = "metrics",
  axis = "step",
  range = null,
  live = true,
  metrics = null;
let request,
  timer,
  runListKey = "",
  configKey = "",
  evalKey = "",
  logState = {},
  axisKey = "";
const info = new Map(),
  snapshots = new Map(),
  charts = new Map();
const preferred = ["loss", "grad_norm", "lr", "loss_action", "loss_video", "loss_l1", "update_s"];
const evaluation = new Evaluation(() => {
  evalKey = "";
  switchTab("evaluation");
});
const primary = () => info.get(selected[0]);
const attemptName = (r) =>
  r.attempt ? "#" + r.ordinal + " · " + t("resume") + " " + r.attempt.split("-")[0] : t("initial");
const runName = (r) => r.name + (r.attempt ? " · " + attemptName(r) : "");
const selection = () => selected.map((id) => info.get(id)).filter(Boolean);
const unionMetrics = () => [
  ...new Set(selection().flatMap((r) => Object.values(r.phases).flatMap((p) => p.metrics))),
];
function sortedKeys(keys) {
  return [...keys].sort((a, b) => {
    const rank = (k) => (preferred.includes(k) ? preferred.indexOf(k) : preferred.length);
    return rank(a) - rank(b) || a.localeCompare(b);
  });
}
function metricSpec(r, k) {
  return r?.schema?.components?.find((s) => s.name === k);
}
function metricDescription(key) {
  const descriptions = selection()
    .filter((r) => Object.values(r.phases).some((p) => p.metrics.includes(key)))
    .map((r) => [runName(r), explain(key, metricSpec(r, key))]);
  if (new Set(descriptions.map(([, text]) => text)).size <= 1) return descriptions[0]?.[1] || explain(key);
  return descriptions.map(([name, text]) => name + " · " + text).join("\n");
}
function getBudget() {
  return Math.min(2048, Math.max(256, Math.ceil($("metrics-view").clientWidth / 64) * 64));
}
function signature(value) {
  return JSON.stringify(value);
}
function setUrl() {
  const url = new URL(location.href);
  if (selected.length) url.searchParams.set("run", selected[0]);
  else url.searchParams.delete("run");
  url.searchParams.set("lang", lang);
  history.replaceState(null, "", url);
}
function sidebar(close = false) {
  const open = !close && !document.body.classList.contains("menu-open");
  document.body.classList.toggle("menu-open", open);
  $("shade").hidden = !open;
}
function selectRun(id, compare = false) {
  if (compare) {
    if (selected.includes(id)) selected = selected.filter((k) => k !== id);
    else selected = [...selected, id];
  } else {
    if (selected[0] === id) {
      sidebar(true);
      return;
    }
    selected = [id, ...(selected.includes(id) ? selected : selected.slice(1)).filter((k) => k !== id)];
    sidebar(true);
  }
  if (!compare) {
    range = null;
    metrics = null;
    logState = {};
  }
  runListKey = "";
  lastHeader = "";
  setUrl();
  if (!compare) {
    $("run-view").hidden = true;
    $("primary-name").textContent = t("loading");
  }
  refresh();
}
function renderList() {
  const filter = $("search").value.trim().toLowerCase();
  const key = signature([lang, selected, filter, runs.map((r) => [r.id, r.name, r.policy, r.attempt])]);
  if (runListKey === key) return;
  runListKey = key;
  $("run-count").textContent = runs.length;
  const list = $("run-list");
  list.replaceChildren();
  for (const r of runs.filter((r) =>
    (runName(r) + " " + (r.policy || "")).toLowerCase().includes(filter),
  )) {
    const i = selected.indexOf(r.id),
      item = el("div", "run" + (i >= 0 ? " selected" : "") + (i === 0 ? " primary-run" : ""));
    if (i >= 0) item.style.setProperty("--ink", colors[i % colors.length]);
    const label = btn("", () => selectRun(r.id), "");
    label.title = runName(r);
    label.append(
      el("strong", null, r.name),
      el("small", null, [r.policy, attemptName(r)].filter(Boolean).join(" · ")),
    );
    label.setAttribute("aria-current", i === 0 ? "true" : "false");
    const box = el("input");
    box.type = "checkbox";
    box.checked = i >= 0;
    box.disabled = i === 0 || (i < 0 && selected.length >= colors.length);
    box.title = (i === 0 ? t("primary") : t("compare")) + " · " + runName(r);
    if (i < 0 && box.disabled) box.title = t("compareLimit");
    box.setAttribute("aria-label", box.title);
    box.onchange = () => selectRun(r.id, true);
    item.append(label, box);
    list.append(item);
  }
  if (!list.childElementCount) list.append(el("p", "side-note", filter ? t("noMatches") : t("noRuns")));
}
function renderHead(r) {
  $("primary-name").textContent = runName(r);
  $("policy-badge").textContent = r.policy || "";
  $("policy-badge").hidden = !r.policy;
  const train = r.phases.train.latest,
    val = r.phases.validation.latest;
  const when = Math.max(train.event_time_s || 0, val.event_time_s || 0);
  const date = new Date(when ? when * 1000 : Number(r.updated_ns) / 1e6);
  $("last-event").textContent =
    (when ? t("lastEvent") : t("fileUpdated")) +
    " " +
    date.toLocaleString(lang, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const cards = [
    ["step", train.step, r.steps ? t("budget") + " " + fmt(r.steps) : t("scope")],
    ["trainLoss", train.loss, t("stepAt") + " " + fmt(train.step)],
    [
      "valLoss",
      val.loss,
      val.step != null
        ? t("stepAt") + " " + fmt(val.step) + " · " + (val.weights || t("validation"))
        : t("unavailable"),
    ],
    ["grad_norm", train.grad_norm, t("stepAt") + " " + fmt(train.step)],
  ];
  const grid = $("kpis");
  cards.forEach(([key, value, note], index) => {
    let cell = grid.children[index];
    if (!cell) {
      cell = el("div", "kpi");
      const label = el("div", "kpi-label");
      label.append(el("span"), hint(""));
      cell.append(label, el("div", "kpi-value"), el("small"));
      if (index === 0) {
        const bar = el("div", "progress");
        bar.append(el("i"));
        cell.append(bar);
      } else {
        const accent = el("span", "accent");
        accent.style.setProperty("--accent", [colors[0], colors[2], colors[1]][index - 1]);
        cell.append(accent);
      }
      grid.append(cell);
    }
    cell.querySelector(".kpi-label span").textContent = t(key);
    const help = cell.querySelector(".hint");
    help.dataset.help = explain(key);
    help.setAttribute("aria-label", help.dataset.help);
    cell.querySelector(".kpi-value").textContent = fmt(value, true);
    cell.querySelector("small").textContent = note;
    if (index === 0) {
      const bar = cell.querySelector(".progress");
      bar.hidden = !r.steps || train.step == null;
      if (!bar.hidden) bar.firstChild.style.width = Math.min(100, (100 * train.step) / r.steps) + "%";
    }
  });
}
function renderAxes() {
  const present = selection()
    .flatMap((r) => Object.values(r.phases))
    .filter((p) => p.records);
  const available = ["step", "samples", "elapsed_s"].filter(
    (k) => k === "step" || (present.length && present.every((p) => p.axes.includes(k))),
  );
  if (!available.includes(axis)) {
    axis = "step";
    range = null;
  }
  const key = signature([available, lang, axis]);
  if (key !== axisKey) {
    axisKey = key;
    $("axis").replaceChildren(...available.map((k) => opt(k, t(k))));
    $("axis").value = axis;
  }
  $("active-range").hidden = !range;
  if (range)
    $("active-range").firstElementChild.textContent =
      t(axis) + "  " + fmt(range[0]) + " – " + fmt(range[1]);
}
function chooseRange(low, high) {
  if (axis !== "elapsed_s") {
    low = Math.ceil(low);
    high = Math.floor(high);
  }
  if (high < low) return;
  range = [low, high];
  refresh();
}
function rangeDialog() {
  const body = modal(t("range") + " · " + t(axis));
  const extents = selection().flatMap((r) =>
    Object.values(snapshots.get(r.id)?.phases || {})
      .map((p) => p.extent)
      .filter(Boolean),
  );
  const bounds = range || [Math.min(...extents.map((e) => e[0])), Math.max(...extents.map((e) => e[1]))];
  const inputs = bounds.map((value, i) => {
    const label = el("label", "field", t(i ? "upper" : "lower")),
      input = el("input", "input");
    input.type = "number";
    input.step = axis === "elapsed_s" ? "any" : "1";
    if (Number.isFinite(value)) input.value = value;
    label.append(input);
    body.append(label);
    return input;
  });
  const error = el("p", "notice", t("invalidRange"));
  error.hidden = true;
  body.append(error);
  const actions = el("div", "dialog-actions");
  actions.append(
    btn(t("resetRange"), () => {
      range = null;
      $("dialog").close();
      refresh();
    }),
    btn(
      t("apply"),
      () => {
        const [low, high] = inputs.map((i) => i.valueAsNumber);
        if (
          !Number.isFinite(low) ||
          !Number.isFinite(high) ||
          low > high ||
          inputs.some((i) => !i.validity.valid)
        ) {
          error.hidden = false;
          return;
        }
        $("dialog").close();
        chooseRange(low, high);
      },
      "primary",
    ),
  );
  body.append(actions);
}
function metricDialog() {
  const body = modal(t("chooseMetrics")),
    keys = sortedKeys(unionMetrics()),
    chosen = new Set(metrics || []);
  const options = el("div");
  for (const key of keys) {
    const label = el("label", "metric-choice"),
      input = el("input");
    input.type = "checkbox";
    input.checked = chosen.has(key);
    input.onchange = () => (input.checked ? chosen.add(key) : chosen.delete(key));
    label.append(input, document.createTextNode(t(key)));
    options.append(label);
  }
  body.append(options);
  const message = el("p", "notice", t("maxMetrics"));
  message.hidden = true;
  body.append(message);
  const actions = el("div", "dialog-actions");
  actions.append(
    btn(
      t("apply"),
      () => {
        if (!chosen.size || chosen.size > 32) {
          message.hidden = false;
          return;
        }
        metrics = sortedKeys(chosen);
        $("dialog").close();
        refresh();
      },
      "primary",
    ),
  );
  body.append(actions);
}
async function fetchSeries(r, keys, queryKey, budget, signal) {
  const old = snapshots.get(r.id);
  if (old?.version === r.version && old.key === queryKey) return false;
  const params = new URLSearchParams({ axis, points: budget });
  keys.forEach((k) => params.append("metric", k));
  if (range) {
    params.set("min", range[0]);
    params.set("max", range[1]);
  }
  const delta = old && old.key === queryKey && old.extra < budget / 4;
  if (delta) for (const [phase, p] of Object.entries(old.phases)) params.set("after_" + phase, p.cursor);
  const phases = await api("runs/" + r.id + "/series", params, signal);
  if (signal.aborted) return false;
  let extra = 0,
    appended = false;
  for (const [phase, p] of Object.entries(phases)) {
    if (p.mode !== "append" || !delta) continue;
    const previous = old.phases[phase].series;
    let added = 0;
    for (const [k, s] of Object.entries(p.series)) {
      added = Math.max(added, s.points.length);
      if (previous[k]) {
        s.points = previous[k].points.concat(s.points);
        s.count += previous[k].count;
      }
    }
    appended = true;
    extra = Math.max(extra, added);
  }
  snapshots.set(r.id, {
    key: queryKey,
    version: r.version,
    phases,
    extra: appended ? old.extra + extra : 0,
  });
  return true;
}
function drawCharts(keys) {
  const box = $("charts");
  box.querySelector(".empty-state")?.remove();
  for (const [key, plot] of charts)
    if (!keys.includes(key)) {
      plot.destroy();
      charts.delete(key);
    }
  $("chart-count").textContent = keys.length;
  if (!keys.length) {
    box.append(el("div", "empty-state", t("noneYet")));
    return;
  }
  keys.forEach((key, index) => {
    let plot = charts.get(key);
    if (!plot) {
      plot = new Plot(box, chooseRange);
      charts.set(key, plot);
    }
    if (box.children[index] !== plot.card) box.insertBefore(plot.card, box.children[index] || null);
    const lines = [];
    for (const [i, r] of selection().entries()) {
      const snap = snapshots.get(r.id);
      for (const [phase, p] of Object.entries(snap?.phases || {})) {
        const s = p.series[key];
        if (!s?.count) continue;
        const color =
          selected.length === 1
            ? phase === "train"
              ? accents[index % accents.length]
              : colors[2]
            : colors[i % colors.length];
        lines.push({
          ...s,
          color,
          dash: phase === "validation",
          label: (selected.length > 1 ? runName(r) + " · " : "") + t(phase),
        });
      }
    }
    plot.update({
      title: t(key),
      description: metricDescription(key),
      axis,
      lines,
      featured: index === 0,
      accent: accents[index % accents.length],
      range,
    });
  });
}
function flatten(object, prefix = "", map = new Map()) {
  for (const [key, value] of Object.entries(object)) {
    const path = prefix ? prefix + "." + key : key;
    if (value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length)
      flatten(value, path, map);
    else map.set(path, typeof value === "string" ? value : JSON.stringify(value));
  }
  return map;
}
function renderConfig() {
  const filter = $("config-filter").value.toLowerCase(),
    list = selection();
  const stamp = signature([lang, filter, list.map((r) => [r.id, r.metadata_version])]);
  if (stamp === configKey) return;
  configKey = stamp;
  const table = $("config-table");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  head.append(el("th", null, t("configKey")));
  list.forEach((r) => head.append(el("th", null, runName(r))));
  const data = list.map((r) => flatten(r.config || {})),
    keys = new Set(data.flatMap((d) => [...d.keys()]));
  const body = table.createTBody();
  for (const k of keys) {
    if (
      filter &&
      !k.toLowerCase().includes(filter) &&
      !data.some((d) => d.get(k)?.toLowerCase().includes(filter))
    )
      continue;
    const row = body.insertRow(),
      name = el("td", null, k);
    const explanation = explain(k, null, true);
    if (explanation) name.append(hint(explanation));
    row.append(name);
    data.forEach((d) => row.append(el("td", "config-value", d.get(k) ?? "—")));
  }
  if (!body.rows.length) body.insertRow().insertCell().textContent = t("noConfig");
}
async function renderLogs(r, signal) {
  const sources = [
    ...r.logs.map((name) => [name, name]),
    ["train", t("rawTrain")],
    ["validation", t("rawVal")],
  ];
  const preferredLog = r.schema?.console_log || (!r.attempt ? r.name + ".log" : null);
  const selector = $("log-source"),
    sig = signature([r.id, sources]);
  if (logState.sources !== sig) {
    selector.replaceChildren(...sources.map(([k, label]) => opt(k, label)));
    selector.value = sources.some(([k]) => k === logState.key)
      ? logState.key
      : r.logs.includes(preferredLog)
        ? preferredLog
        : "train";
    logState = { sources: sig };
  }
  const key = selector.value,
    raw = ["train", "validation"].includes(key);
  const cursor = logState.key === key ? logState.cursor : "";
  const params = new URLSearchParams(raw ? { phase: key, limit: 150, tail: 1 } : { name: key });
  if (cursor) params.set("cursor", cursor);
  const value = await api("runs/" + r.id + (raw ? "/records" : "/console"), params, signal);
  if (signal.aborted) return;
  const more = raw ? value.records.map((x) => JSON.stringify(x) + "\n").join("") : value.text;
  const previous = cursor && !value.reset ? logState.text : "";
  // Bound both DOM and memory even for progress bars using carriage returns.
  const text = (previous + more).slice(-64000);
  const output = $("log-output");
  const stick = !cursor || output.scrollHeight - output.scrollTop - output.clientHeight < 25;
  if (!cursor || value.reset || more) output.textContent = text.replace(/\r\n?/g, "\n") || t("logEmpty");
  if (stick) output.scrollTop = output.scrollHeight;
  logState = { key, cursor: value.cursor, text, sources: sig };
}
function switchTab(next) {
  tab = next;
  document.querySelectorAll("[data-tab]").forEach((n) => {
    if (n.dataset.tab === tab) n.setAttribute("aria-current", "page");
    else n.removeAttribute("aria-current");
    $(n.dataset.tab + "-view").hidden = n.dataset.tab !== tab;
  });
  if (next !== "evaluation") {
    evaluation.clear();
    evalKey = "";
  }
  refresh();
}
function showLive() {
  $("live-label").textContent = t(live ? "follow" : "paused");
  $("live").setAttribute("aria-pressed", String(live));
  $("live").title = t(live ? "pauseHint" : "resumeHint");
  $("live-icon").setAttribute("href", live ? "#icon-pause" : "#icon-play");
  $("language-label").textContent = lang === "zh" ? "English" : "中文";
}
let lastPaint = "",
  lastHeader = "";
async function sync(signal) {
  const listing = await api("runs", {}, signal);
  if (signal.aborted) return;
  const order = new Map(runs.map((r, i) => [r.id, i]));
  runs = listing.runs.sort((a, b) => (order.get(a.id) ?? -1) - (order.get(b.id) ?? -1));
  selected = selected.filter((id) => runs.some((r) => r.id === id));
  if (!selected.length && runs.length) {
    const requested = new URL(location.href).searchParams.get("run");
    selected = [runs.find((r) => r.id === requested)?.id || runs[0].id];
    setUrl();
  }
  for (const map of [info, snapshots])
    for (const id of map.keys()) if (!selected.includes(id)) map.delete(id);
  renderList();
  const empty = !selected.length;
  $("empty").hidden = !empty;
  $("import").disabled = empty;
  if (empty) {
    lastHeader = "";
    $("run-view").hidden = true;
    $("policy-badge").hidden = true;
    $("primary-name").textContent = t("localWorkspace");
    $("last-event").textContent = "";
    $("feedback").hidden = true;
    return;
  }
  await Promise.all(
    selected.map(async (id) => {
      const item = runs.find((r) => r.id === id),
        prev = info.get(id);
      if (prev?.version === item.version) return;
      const brief = prev?.metadata_version === item.metadata_version;
      const value = await api("runs/" + id, brief ? { brief: 1 } : {}, signal);
      // The catalog revision precedes these reads. A write during a read must
      // still invalidate the next poll, even if it was the last training step.
      if (!signal.aborted)
        info.set(id, { ...prev, ...value, version: item.version, metadata_version: item.metadata_version });
    }),
  );
  if (signal.aborted) return;
  const r = primary();
  $("run-view").hidden = false;
  const headerSig = signature([r.id, r.version, lang]);
  if (headerSig !== lastHeader) {
    renderHead(r);
    lastHeader = headerSig;
  }
  const notices = [];
  for (const run of selection())
    for (const [phase, p] of Object.entries(run.phases)) {
      const name = (selected.length > 1 ? run.name + " · " : "") + t(phase);
      if (p.invalid_lines) notices.push(name + " · " + t("recordIssues") + " " + p.invalid_lines);
      if (Object.keys(p.nonfinite).length)
        notices.push(
          name +
            " · " +
            t("nonfinite") +
            "  " +
            Object.entries(p.nonfinite)
              .map(([k, n]) => k + " × " + n)
              .join(", "),
        );
    }
  $("feedback").textContent = notices.join(" / ");
  $("feedback").hidden = !notices.length;
  if (tab === "metrics") {
    renderAxes();
    const all = unionMetrics();
    if (!metrics?.length) metrics = sortedKeys(all).slice(0, 5);
    const keys = metrics.filter((k) => all.includes(k)),
      budget = getBudget();
    const queryKey = signature([axis, range, keys, budget]);
    const changed = await Promise.all(
      selection().map((run) => fetchSeries(run, keys, queryKey, budget, signal)),
    );
    if (signal.aborted) return;
    const paint = signature([queryKey, lang, selected, r.version]);
    if (lastPaint !== paint || changed.some(Boolean)) {
      drawCharts(keys);
      lastPaint = paint;
    }
  } else if (tab === "config") renderConfig();
  else if (tab === "logs") await renderLogs(r, signal);
  else if (tab === "evaluation") {
    const key = signature([r.id, r.metadata_version, lang]);
    if (key !== evalKey) {
      await evaluation.render(r);
      if (!signal.aborted) evalKey = key;
    }
  }
}
function refresh() {
  clearTimeout(timer);
  request?.abort();
  const c = new AbortController();
  request = c;
  sync(c.signal)
    .catch((e) => {
      if (!c.signal.aborted) showError(e);
    })
    .finally(() => {
      if (request === c && live && !document.hidden) timer = setTimeout(refresh, 1800);
    });
}
$("live").onclick = () => {
  live = !live;
  showLive();
  if (live) refresh();
  else {
    clearTimeout(timer);
    request?.abort();
  }
};
$("language").onclick = () => {
  toggleLang();
  setUrl();
  translate();
  showLive();
  refresh();
};
$("import").onclick = () => {
  if (primary()) evaluation.picker(primary()).catch(showError);
};
$("axis").onchange = (e) => {
  axis = e.target.value;
  range = null;
  refresh();
};
$("metrics-btn").onclick = metricDialog;
$("range-btn").onclick = rangeDialog;
$("reset-range").onclick = () => {
  range = null;
  refresh();
};
$("search").oninput = renderList;
$("config-filter").oninput = renderConfig;
$("log-source").onchange = () => {
  logState.key = $("log-source").value;
  logState.cursor = "";
  refresh();
};
$("menu").onclick = () => sidebar();
$("shade").onclick = () => sidebar(true);
document
  .querySelectorAll("[data-tab]")
  .forEach((node) => (node.onclick = () => switchTab(node.dataset.tab)));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearTimeout(timer);
    request?.abort();
  } else if (live) refresh();
});
let sizeTimer, lastWidth;
new ResizeObserver(() => {
  const w = getBudget();
  if (
    lastWidth != null &&
    w !== lastWidth &&
    tab === "metrics" &&
    !document.hidden &&
    !$("run-view").hidden
  ) {
    clearTimeout(sizeTimer);
    sizeTimer = setTimeout(refresh, 200);
  }
  lastWidth = w;
}).observe($("metrics-view"));
translate();
showLive();
switchTab("metrics");
