import {
  $,
  api,
  btn,
  el,
  explain,
  fmt,
  describe,
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
import { inspectChart } from "./inspector.js";
import { Diagnostics } from "./diagnostics.js";
import { configurationFacts, factCell, reportControls } from "./resources.js";

let runs = [],
  selected = [],
  tab = "metrics",
  axis = "step",
  range = null,
  live = true,
  metrics = null,
  trend = 100,
  chartPage = 0;
let request,
  timer,
  runListKey = "",
  configKey = "",
  configCache,
  evalKey = "",
  logState = {},
  axisKey = "";
const info = new Map(),
  snapshots = new Map(),
  charts = new Map();
const chartsPerPage = 8;
const evaluation = new Evaluation(() => {
  evalKey = "";
  switchTab("evaluation");
});
const diagnostics = new Diagnostics(refresh);
const primary = () => info.get(selected[0]);
const attemptName = (r) =>
  r.attempt ? "#" + r.ordinal + " · " + t("resume") + " " + r.attempt.split("-")[0] : t("initial");
const runName = (r) => r.name + (r.attempt ? " · " + attemptName(r) : "");
const selection = () => selected.map((id) => info.get(id)).filter(Boolean);
const unionMetrics = () => [
  ...new Set(selection().flatMap((r) => Object.values(r.phases).flatMap((p) => p.metrics))),
];
function policyMetrics(r) {
  const phases = Object.values(r.phases);
  const available = new Set(phases.flatMap((p) => p.metrics));
  const specs = r.schema?.components || [];
  return [
    ...new Set([
      ...specs.filter((s) => s.console).map((s) => s.name),
      ...specs.map((s) => s.name),
      ...phases.flatMap((p) => p.policy_metrics || []),
      ...[...available].filter(
        (k) => k.startsWith("loss_") || k.endsWith("_loss") || k.startsWith("losses_after_"),
      ),
    ]),
  ].filter((k) => available.has(k) && k !== "loss");
}
function metricSpec(r, k) {
  return r?.schema?.components?.find((s) => s.name === k);
}
function metricDescription(key) {
  const descriptions = selection()
    .filter((r) => Object.values(r.phases).some((p) => p.metrics.includes(key)))
    .map((r) => [runName(r), [explain(key, metricSpec(r, key), r), key].filter(Boolean).join("\n")]);
  if (new Set(descriptions.map(([, text]) => text)).size <= 1)
    return descriptions[0]?.[1] || explain(key);
  return descriptions.map(([name, text]) => name + " · " + text).join("\n");
}
function defaultMetrics() {
  const available = unionMetrics();
  const policy = selection().flatMap(policyMetrics);
  const groupRates = available.filter((k) => k.startsWith("lr/"));
  const defaults = [
    "loss",
    ...policy,
    "grad_norm",
    ...(groupRates.length > 1 || !available.includes("lr") ? groupRates : ["lr"]),
  ];
  const recorded = [...new Set(defaults)].filter((k) => available.includes(k));
  return recorded.length ? recorded : available.slice(0, 5);
}
function extent() {
  const extents = selection().flatMap((r) =>
    Object.values(snapshots.get(r.id)?.phases || {})
      .map((p) => p.extent)
      .filter(Boolean),
  );
  return extents.length
    ? [Math.min(...extents.map((v) => v[0])), Math.max(...extents.map((v) => v[1]))]
    : null;
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
const mobile = matchMedia("(max-width:850px)");
let collapsed = localStorage.getItem("am:sidebarCollapsed") === "1",
  drawer = false;
function showSidebar() {
  const open = mobile.matches ? drawer : !collapsed;
  document.body.classList.toggle("sidebar-collapsed", !mobile.matches && collapsed);
  document.body.classList.toggle("menu-open", mobile.matches && drawer);
  $("sidebar").inert = !open;
  $("shade").hidden = !mobile.matches || !drawer;
  $("menu").title = t(open ? "hideRuns" : "showRuns");
  $("menu").setAttribute("aria-label", $("menu").title);
  $("menu").setAttribute("aria-expanded", String(open));
}
function sidebar(close = false) {
  if (mobile.matches) {
    drawer = close ? false : !drawer;
    if (!drawer && $("sidebar").contains(document.activeElement)) $("menu").focus();
  } else if (!close) {
    collapsed = !collapsed;
    localStorage.setItem("am:sidebarCollapsed", collapsed ? "1" : "0");
  }
  showSidebar();
  if (mobile.matches && drawer) $("search").focus();
}
mobile.addEventListener("change", () => {
  drawer = false;
  showSidebar();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer) sidebar(true);
});
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
  chartPage = 0;
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
    if (i === 0) {
      const mark = el("span", "primary-mark", "✓");
      mark.title = t("primary");
      item.append(label, mark);
    } else item.append(label, box);
    list.append(item);
  }
  if (!list.childElementCount) list.append(el("p", "side-note", filter ? t("noMatches") : t("noRuns")));
}
function renderHead(r) {
  $("primary-name").textContent = $("primary-name").title = runName(r);
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
  const readings = [
    ["step", train.step, r.steps ? " / " + fmt(r.steps) : ""],
    ["trainLoss", train.loss, ""],
    ["valLoss", val.loss, val.step != null && val.step !== train.step ? " @ " + fmt(val.step) : ""],
    ["grad_norm", train.grad_norm, ""],
  ];
  const grid = $("kpis");
  for (const [i, [key, value, note]] of readings.entries()) {
    let cell = grid.children[i];
    if (!cell) {
      cell = el("div");
      cell.append(el("dt"), el("dd"));
      grid.append(cell);
    }
    cell.firstChild.textContent = t(key);
    cell.lastChild.replaceChildren(document.createTextNode(fmt(value, true)), el("small", null, note));
  }
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
    all = unionMetrics(),
    policy = new Set(selection().flatMap(policyMetrics)),
    chosen = new Set(metrics || defaultMetrics());
  const filter = el("input", "input");
  filter.type = "search";
  filter.placeholder = t("findMetric");
  filter.setAttribute("aria-label", filter.placeholder);
  body.append(filter);
  const options = el("div");
  function group(title, keys) {
    if (!keys.length) return;
    const field = el("fieldset", "metric-group");
    field.append(el("legend", null, t(title)));
    for (const key of keys) {
      const label = el("label", "metric-choice"),
        input = el("input");
      input.type = "checkbox";
      input.checked = chosen.has(key);
      input.onchange = () => (input.checked ? chosen.add(key) : chosen.delete(key));
      label.append(input, document.createTextNode(t(key)));
      label.dataset.search = (key + " " + t(key)).toLowerCase();
      field.append(label);
    }
    options.append(field);
  }
  group(
    "policyMetrics",
    [...new Set(["loss", ...policy])].filter((k) => all.includes(k)),
  );
  group(
    "trainingMetrics",
    all.filter((k) => k !== "loss" && !policy.has(k) && !k.startsWith("module/")),
  );
  group(
    "probe",
    all.filter((k) => k.startsWith("module/") && !policy.has(k)),
  );
  filter.oninput = () => {
    const needle = filter.value.trim().toLowerCase();
    for (const field of options.children) {
      for (const label of field.querySelectorAll("label"))
        label.hidden = !label.dataset.search.includes(needle);
      field.hidden = !field.querySelector("label:not([hidden])");
    }
  };
  body.append(options);
  const message = el("p", "notice", t("maxMetrics"));
  message.hidden = true;
  body.append(message);
  const actions = el("div", "dialog-actions");
  actions.append(
    btn(t("byPolicy"), () => {
      metrics = null;
      chartPage = 0;
      $("dialog").close();
      refresh();
    }),
    btn(
      t("apply"),
      () => {
        if (!chosen.size) {
          message.hidden = false;
          return;
        }
        metrics = [...chosen];
        chartPage = 0;
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
  const params = new URLSearchParams({ axis, points: budget, mean_window: trend });
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
    let added = 0;
    for (const group of ["series", "means"]) {
      for (const [k, series] of Object.entries(p[group] || {})) {
        added = Math.max(added, series.points.length);
        const previous = old.phases[phase][group]?.[k];
        if (previous) {
          series.points = previous.points.concat(series.points);
          series.count += previous.count;
        }
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
function seriesLines(key, index, runs, phases, windowSize) {
  const lines = [];
  for (const [i, r] of runs.entries()) {
    for (const [phase, p] of Object.entries(phases[i] || {})) {
      const series = p.series[key];
      if (!series?.count) continue;
      const mean = p.means?.[key],
        hasMean = mean?.points.some((point) => point[1] != null);
      const slot = index % accents.length;
      const phaseColor = phase === "train" ? accents[slot] : slot === 2 ? colors[3] : colors[2];
      const base = runs.length === 1 ? phaseColor : colors[i % colors.length];
      const prefix = runs.length > 1 ? runName(r) + " · " : "";
      lines.push({
        ...series,
        id: r.id + "/" + phase,
        color: hasMean ? base + "58" : base,
        faint: hasMean,
        dash: phase === "validation",
        label: prefix + t(hasMean ? "rawLine" : phase),
      });
      if (hasMean)
        lines.push({
          ...mean,
          id: r.id + "/mean",
          trend: true,
          color: base,
          label: prefix + t("mean") + " · " + fmt(windowSize) + " " + t("recordsUnit"),
        });
    }
  }
  return lines;
}
function drawCharts(keys) {
  const box = $("charts");
  box.querySelector(".empty-state")?.remove();
  for (const [key, plot] of charts)
    if (!keys.includes(key)) {
      plot.destroy();
      charts.delete(key);
    }
  const runs = selection(),
    policy = new Set(runs.flatMap(policyMetrics));
  const groups = new Map();
  for (const [index, key] of keys.entries()) {
    let group = "trainingMetrics";
    if (key === "loss" || policy.has(key)) group = "policyMetrics";
    else if (/^(lr($|\/)|grad_|skipped_updates$)/.test(key)) group = "optimizer";
    else if (key.startsWith("module/")) group = "probe";
    else if (["generalization_gap", "train_eval_loss", "train_eval_samples"].includes(key))
      group = "paired";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([key, index]);
  }
  for (const section of [...box.children]) if (!groups.has(section.dataset.group)) section.remove();
  if (!keys.length) {
    box.append(el("div", "empty-state", t("noneYet")));
    return;
  }
  let groupIndex = 0;
  for (const [group, items] of groups) {
    let section = box.querySelector('[data-group="' + group + '"]');
    if (!section) {
      section = el("section", "chart-section");
      section.dataset.group = group;
      section.append(el("h2", "section-bar"), el("div", "chart-grid"));
    }
    if (box.children[groupIndex] !== section)
      box.insertBefore(section, box.children[groupIndex] || null);
    groupIndex++;
    section.firstChild.textContent = t(group);
    const grid = section.lastChild;
    items.forEach(([key, index], position) => {
      let plot = charts.get(key);
      if (!plot) {
        plot = new Plot(grid, chooseRange, () => {
          const index = plot.seriesIndex,
            currentTrend = trend,
            runs = selection();
          inspectChart({
            key,
            description: metricDescription(key),
            runs,
            axis,
            bounds: extent(),
            initial: range,
            trend: currentTrend,
            linesFor: (phases) => seriesLines(key, index, runs, phases, currentTrend),
            accent: accents[index % accents.length],
          });
        });
        charts.set(key, plot);
      }
      if (grid.children[position] !== plot.card)
        grid.insertBefore(plot.card, grid.children[position] || null);
      plot.seriesIndex = index;
      const lines = seriesLines(
        key,
        index,
        runs,
        runs.map((r) => snapshots.get(r.id)?.phases),
        trend,
      );
      plot.update({
        title: t(key),
        description: metricDescription(key),
        axis,
        lines,
        accent: accents[index % accents.length],
        range: range || extent(),
      });
    });
  }
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
  const meta = signature([lang, list.map((r) => [r.id, r.metadata_version])]);
  const stamp = signature([meta, filter]);
  if (stamp === configKey) return;
  configKey = stamp;
  const table = $("config-table");
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  head.append(el("th", null, t("configKey")));
  list.forEach((r) => head.append(el("th", null, runName(r))));
  if (configCache?.meta !== meta) {
    const data = list.map((r) => flatten(r.config || {})),
      facts = list.map(configurationFacts);
    facts.forEach((map, i) => {
      for (const item of map.values()) for (const alias of item.aliases) data[i].delete(alias);
    });
    configCache = { meta, data, facts };
    reportControls(list[0], () => {
      configKey = "";
      configCache = null;
      renderConfig();
    });
  }
  const { data, facts } = configCache;
  const keys = new Set([...facts.flatMap((m) => [...m.keys()]), ...data.flatMap((m) => [...m.keys()])]);
  const body = table.createTBody();
  for (const k of keys) {
    const fact = facts.map((m) => m.get(k)).find(Boolean);
    const label = fact?.label || k;
    const searchable = [
      k,
      label,
      ...data.map((d) => d.get(k)),
      ...facts.flatMap((m) => {
        const item = m.get(k);
        return item ? [item.value, ...item.details.flat(), ...item.aliases] : [];
      }),
    ];
    if (filter && !searchable.some((v) => v != null && String(v).toLowerCase().includes(filter)))
      continue;
    const row = body.insertRow(),
      name = el("td", null, label);
    if (fact) name.title = k;
    else describe(name, explain(k));
    row.append(name);
    data.forEach((d, i) => {
      const value = facts[i].get(k),
        cell = el("td", "config-value");
      if (value) cell.append(factCell(value));
      else cell.textContent = d.get(k) ?? "—";
      row.append(cell);
    });
  }
  if (!body.rows.length) body.insertRow().insertCell().textContent = t("noConfig");
}
async function renderLogs(r, signal) {
  const sources = [
    ...r.logs.filter((name) => !name.endsWith(".preflight.log")).map((name) => [name, name]),
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
      : sources.some(([k]) => k === preferredLog)
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
  const url = new URL(location.href);
  if (next === "metrics") url.searchParams.delete("view");
  else url.searchParams.set("view", next);
  history.replaceState(null, "", url);
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
  showSidebar();
  $("trend").replaceChildren(
    ...[0, 20, 100, 500].map((n) => opt(n, n ? fmt(n) + " " + t("recordsUnit") : t("off"))),
  );
  $("trend").value = trend;
  $("live").setAttribute("aria-label", $("live").title);
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
  $("import").hidden = empty;
  if (empty) {
    lastHeader = "";
    $("run-view").hidden = true;
    $("policy-badge").hidden = true;
    $("primary-name").textContent = t("monitor");
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
        info.set(id, {
          ...prev,
          ...value,
          version: item.version,
          metadata_version: item.metadata_version,
        });
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
    $("metrics-view").querySelector(".toolbar").hidden = !all.length;
    const chosen = (metrics || defaultMetrics()).filter((k) => all.includes(k));
    chartPage = Math.max(0, Math.min(chartPage, Math.ceil(chosen.length / chartsPerPage) - 1));
    const offset = chartPage * chartsPerPage;
    const keys = chosen.slice(offset, offset + chartsPerPage),
      budget = getBudget();
    $("chart-pages").hidden = chosen.length <= chartsPerPage;
    $("charts-prev").disabled = chartPage === 0;
    $("charts-next").disabled = offset + keys.length >= chosen.length;
    $("chart-page-status").textContent =
      offset + 1 + "–" + (offset + keys.length) + " / " + chosen.length;
    const queryKey = signature([axis, range, keys, budget, trend]);
    const changed = await Promise.all(
      selection().map((run) => fetchSeries(run, keys, queryKey, budget, signal)),
    );
    if (signal.aborted) return;
    const paint = signature([queryKey, lang, selected, r.version]);
    if (lastPaint !== paint || changed.some(Boolean)) {
      drawCharts(keys);
      lastPaint = paint;
    }
  } else if (tab === "diagnostics") await diagnostics.render(selection(), runName, signal);
  else if (tab === "config") renderConfig();
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
$("trend").onchange = () => {
  trend = Number($("trend").value);
  refresh();
};
const settings = $("display-settings");
document.addEventListener("click", (e) => {
  if (settings.open && !settings.contains(e.target)) settings.open = false;
});
settings.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    settings.open = false;
    settings.querySelector("summary").focus();
  }
});
const chartWidth = $("chart-width");
const storedWidth = Number(localStorage.getItem("am:chart-width"));
if (storedWidth >= Number(chartWidth.min) && storedWidth <= Number(chartWidth.max))
  chartWidth.value = storedWidth;
function sizeCharts() {
  const width = chartWidth.valueAsNumber;
  $("charts").style.setProperty("--chart-width", width + "px");
  $("width-value").textContent = width + " px";
  chartWidth.setAttribute("aria-valuetext", width + " px");
}
chartWidth.oninput = sizeCharts;
chartWidth.onchange = () => localStorage.setItem("am:chart-width", chartWidth.value);
sizeCharts();
$("metrics-btn").onclick = metricDialog;
$("charts-prev").onclick = () => {
  chartPage--;
  refresh();
};
$("charts-next").onclick = () => {
  chartPage++;
  refresh();
};
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
$("close-sidebar").onclick = () => sidebar(true);
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
const initialView = new URL(location.href).searchParams.get("view");
switchTab(
  [...document.querySelectorAll("[data-tab]")].some((n) => n.dataset.tab === initialView)
    ? initialView
    : "metrics",
);
