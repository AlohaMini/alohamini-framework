import {
  $,
  api,
  btn,
  describe,
  el,
  explain,
  fmt,
  fmtPct,
  lang,
  modal,
  onThemeChange,
  opt,
  showError,
  t,
  toggleLang,
  toggleTheme,
  translate,
} from "./ui.js";
import { colors, paletteColor, Plot } from "./plot.js";
import { Evaluation } from "./evaluation.js";
import { inspectChart } from "./inspector.js";
import { rangeSlider } from "./range.js";
import { Diagnostics } from "./diagnostics.js";
import { CompareView } from "./compare.js";
import { configurationFacts, factCell, reportControls } from "./resources.js";

let runs = [],
  selected = [],
  tab = "metrics",
  axis = "step",
  range = null,
  live = true,
  metrics = null,
  smooth = Number(localStorage.getItem("am:smooth") ?? "0.90"),
  yScale = localStorage.getItem("am:yScale") || "robust",
  showRaw = localStorage.getItem("am:showRaw") !== "0",
  showEvents = localStorage.getItem("am:showEvents") !== "0",
  stitchResumes = localStorage.getItem("am:stitch") !== "0",
  metricFilter = "";

if (!Number.isFinite(smooth) || smooth < 0 || smooth > 0.99) smooth = 0.9;
if (!["fit", "robust", "log", "zero"].includes(yScale)) yScale = "robust";

let request,
  timer,
  runListKey = "",
  configKey = "",
  configCache,
  evalKey = "",
  logState = {},
  axisKey = "",
  lastPaint = "",
  lastHeader = "";

const info = new Map(),
  snapshots = new Map(),
  eventsCache = new Map(),
  charts = new Map(),
  expandedGroups = new Set(),
  collapsedSections = new Set(JSON.parse(localStorage.getItem("am:collapsedSections") || "[]"));

const evaluation = new Evaluation(() => {
  evalKey = "";
  switchTab("evaluation");
});

function jumpToMetricStep(metricKey, step) {
  if (metricKey) {
    const all = unionMetrics();
    const cur = metrics || defaultMetrics();
    if (all.includes(metricKey) && !cur.includes(metricKey)) {
      metrics = [...cur, metricKey];
    }
  }
  switchTab("metrics");
  setTimeout(() => {
    const plot = charts.get(metricKey) || charts.get("loss");
    if (plot && Number.isFinite(step)) plot.flashStep(step);
  }, 150);
}

const diagnostics = new Diagnostics(() => refresh(), jumpToMetricStep);
const compareView = new CompareView({
  onSelectRun: (id) => {
    selectRun(id, false);
    switchTab("metrics");
  },
  onToggleCompare: (id) => {
    selectRun(id, true);
  },
});

const overviewRange = rangeSlider(chooseRange, () => {
  range = null;
  refresh();
});
$("interval-tools").append(overviewRange.element);

const primary = () => info.get(selected[0]);
const attemptName = (r) =>
  r.attempt ? "#" + r.ordinal + " · " + t("resume") + " " + r.attempt.split("-")[0] : t("initial");
const runName = (r) => r.name + (r.attempt && !r.stitched ? " · " + attemptName(r) : "");
const selection = () => selected.map((id) => info.get(id)).filter(Boolean);

function runAvailableMetrics(r) {
  return [
    ...new Set([
      ...Object.values(r.phases).flatMap((p) => p.metrics || []),
      ...(r.phases.train?.derived_metrics || []),
    ]),
  ];
}
const unionMetrics = () => [...new Set(selection().flatMap(runAvailableMetrics))];

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
    .filter((r) => runAvailableMetrics(r).includes(key))
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
    "generalization_gap",
    "derived/generalization_gap",
    "grad_norm",
    "derived/clip_fraction",
    ...(groupRates.length > 1 || !available.includes("lr") ? groupRates : ["lr"]),
    "derived/samples_per_s",
    "derived/data_wait_ratio",
  ];
  const recorded = [...new Set(defaults)].filter((k) => available.includes(k));
  return recorded.length ? recorded : available.slice(0, 8);
}

function extent() {
  const extents = selection().flatMap((r) =>
    Object.values(snapshots.get(r.id)?.phases || {})
      .map((p) => p?.extent)
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

// Sidebar & responsive layout
const mobile = matchMedia("(max-width:850px)");
const topbar = document.querySelector(".topbar"),
  tabs = topbar.querySelector(".tabs"),
  topActions = document.querySelector(".top-actions");
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
  $("close-sidebar").title = t("hideRuns");
  $("close-sidebar").setAttribute("aria-expanded", String(open));
  const host = mobile.matches ? $("sidebar") : topbar;
  if (topActions.parentElement !== host) host.append(topActions);
  topbar.inert = document.querySelector("main").inert = mobile.matches && drawer;
}
function sidebar(close = false) {
  const fromSidebar = $("sidebar").contains(document.activeElement);
  if (mobile.matches) {
    drawer = close ? false : !drawer;
  } else if (!close) {
    collapsed = !collapsed;
    localStorage.setItem("am:sidebarCollapsed", collapsed ? "1" : "0");
  }
  showSidebar();
  if (mobile.matches) {
    if (drawer) $("close-sidebar").focus({ preventScroll: true });
    else if (fromSidebar) $("menu").focus();
  } else if (!close) $(collapsed ? "menu" : "close-sidebar").focus();
}
mobile.addEventListener("change", () => {
  drawer = false;
  showSidebar();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer && !e.defaultPrevented && !$("dialog").open) sidebar(true);
});

function selectRun(id, compare = false, explicitStitch = null) {
  if (explicitStitch !== null) {
    stitchResumes = explicitStitch;
    $("toggle-stitch").checked = stitchResumes;
    localStorage.setItem("am:stitch", stitchResumes ? "1" : "0");
    snapshots.clear();
    eventsCache.clear();
    info.clear();
  }
  if (compare) {
    if (selected.includes(id)) {
      if (selected.length > 1) selected = selected.filter((k) => k !== id);
    } else if (selected.length < colors.length) {
      selected = [...selected, id];
    }
  } else {
    if (selected[0] === id && explicitStitch === null) {
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
  lastPaint = "";
  setUrl();
  if (!compare) {
    $("run-view").hidden = true;
    $("display-settings").hidden = true;
  }
  refresh();
}

function breakableName(text) {
  const frag = document.createDocumentFragment();
  const parts = String(text || "").split(/([_-])/);
  for (const part of parts) {
    frag.append(document.createTextNode(part));
    if (part === "_" || part === "-") frag.append(document.createElement("wbr"));
  }
  return frag;
}

function renderRunItem(r, { isChild = false, groupItems = null } = {}) {
  const i = selected.indexOf(r.id);
  const item = el("div", "run" + (i >= 0 ? " selected" : "") + (i === 0 ? " primary-run" : ""));
  const swatchColor = i >= 0 ? paletteColor(i) : null;
  if (swatchColor) item.style.setProperty("--run-color", swatchColor);

  const swatch = el("span", "run-swatch");
  const label = btn("", () => selectRun(r.id, false, isChild ? false : groupItems?.length > 1 ? true : null), "run-main");
  label.title = r.name + (r.attempt ? " · " + attemptName(r) : "");
  label.setAttribute("aria-current", i === 0 ? "true" : "false");

  const titleRow = el("div", "run-title-row");
  const st = info.get(r.id)?.status || r.status || "idle";
  const stDot = el("span", "run-status-dot " + st);
  stDot.title = t(st === "running" ? "statusRunning" : st === "completed" ? "statusCompleted" : "statusIdle");
  const displayTitle = isChild ? attemptName(r) : r.name;
  const strong = el("strong");
  strong.append(breakableName(displayTitle));
  titleRow.append(stDot, strong);
  label.append(titleRow);

  const subParts = isChild
    ? [r.policy, r.steps ? fmt(r.steps, true) + " steps" : null]
    : [
        r.policy,
        groupItems && groupItems.length > 1 ? `${groupItems.length} ${t("resumesCount")}` : attemptName(r),
      ];
  label.append(el("small", null, subParts.filter(Boolean).join(" · ")));

  const loaded = info.get(r.id);
  const curStep = loaded?.phases?.train?.latest?.step;
  if (Number.isFinite(curStep) && Number.isFinite(r.steps) && r.steps > 0) {
    const prog = el("div", "run-progress");
    const bar = el("i");
    bar.style.width = `${Math.min(100, (100 * curStep) / r.steps)}%`;
    prog.append(bar);
    label.append(prog);
  }

  if (i === 0) {
    const time = el("time");
    time.id = "last-event";
    label.append(time);
    if (primary()) updateRunTime(primary(), time);
    const mark = el("span", "primary-mark", "✓");
    mark.title = t("primary");
    item.append(swatch, label, mark);
  } else {
    const box = el("input");
    box.type = "checkbox";
    box.checked = i >= 0;
    box.disabled = i < 0 && selected.length >= colors.length;
    box.title = (i >= 0 ? t("compare") : t("compare")) + " · " + r.name;
    if (i < 0 && box.disabled) box.title = t("compareLimit");
    box.setAttribute("aria-label", box.title);
    box.onchange = () => selectRun(r.id, true);
    item.append(swatch, label, box);
  }
  return item;
}

function renderList() {
  const filter = $("search").value.trim().toLowerCase();
  const key = signature([
    lang,
    selected,
    filter,
    [...expandedGroups],
    runs.map((r) => [r.id, r.name, r.policy, r.attempt, r.status, info.get(r.id)?.phases?.train?.latest?.step]),
  ]);
  if (runListKey === key) return;
  runListKey = key;

  // Group runs by group_id (same output directory -> initial + resumes)
  const grouped = new Map();
  for (const r of runs) {
    const gid = r.group_id || r.id;
    if (!grouped.has(gid)) grouped.set(gid, []);
    grouped.get(gid).push(r);
  }
  $("run-count").textContent = grouped.size;

  const list = $("run-list");
  list.replaceChildren();
  let shown = 0;
  for (const [gid, items] of grouped) {
    const match = items.some((r) =>
      (r.name + " " + (r.policy || "") + " " + (r.attempt || "")).toLowerCase().includes(filter),
    );
    if (!match) continue;
    shown++;
    // Latest updated item is representative head of group
    const head = items[0];
    const wrap = el("div", "run-group");
    wrap.append(renderRunItem(head, { isChild: false, groupItems: items }));
    if (items.length > 1) {
      const childSelected = items.slice(1).some((c) => selected.includes(c.id));
      const isOpen = expandedGroups.has(gid) || childSelected;
      const toggle = el("button", "group-toggle");
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", String(isOpen));
      toggle.innerHTML = `<svg class="icon"><use href="#icon-chevron" /></svg>`;
      toggle.append(document.createTextNode(`${items.length} ${t("resumesCount")}`));
      toggle.onclick = () => {
        if (expandedGroups.has(gid)) expandedGroups.delete(gid);
        else expandedGroups.add(gid);
        runListKey = "";
        renderList();
      };
      wrap.append(toggle);
      if (isOpen) {
        const kids = el("div", "run-children");
        const orderedKids = [...items].sort((a, b) => (a.ordinal || 0) - (b.ordinal || 0));
        for (const child of orderedKids) kids.append(renderRunItem(child, { isChild: true }));
        wrap.append(kids);
      }
    }
    list.append(wrap);
  }
  if (!shown) list.append(el("p", "side-note", filter ? t("noMatches") : t("noRuns")));
}

function updateRunTime(r, time) {
  const when = Math.max(
    r.phases.train.latest.event_time_s || 0,
    r.phases.validation.latest.event_time_s || 0,
  );
  const date = new Date(when ? when * 1000 : Number(r.updated_ns) / 1e6);
  if (!date.getTime()) return;
  time.dateTime = date.toISOString();
  time.textContent =
    (when ? t("lastEvent") : t("fileUpdated")) +
    " " +
    date.toLocaleString(lang, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function renderHead(r) {
  const time = $("last-event");
  if (time) updateRunTime(r, time);
  const grid = $("kpis");
  // Show compact KPI strip ONLY on Metrics & Diagnostics views
  const showKpis = tab === "metrics" || tab === "diagnostics";
  grid.hidden = !showKpis;
  if (!showKpis) return;

  const train = r.phases.train.latest || {},
    val = r.phases.validation.latest || {},
    best = r.phases.validation.best;
  const tailMean = r.phases.train.tail_mean_loss;
  const clipThr = r.config?.optimizer?.grad_clip_norm;

  // Build 5 high-signal compact KPI cells, each with gray-bg white-text hover explanation
  const pct =
    Number.isFinite(train.step) && Number.isFinite(r.steps) && r.steps > 0
      ? ` (${fmt((100 * train.step) / r.steps, true)}%)`
      : "";
  const statusTag = el(
    "span",
    "tag " + (r.status === "running" ? "ok" : r.status === "completed" ? "info" : ""),
    t(r.status === "running" ? "statusRunning" : r.status === "completed" ? "statusCompleted" : "statusIdle"),
  );

  // Validation delta vs best
  let valNote = "",
    valDeltaClass = "";
  if (Number.isFinite(val.loss) && best && Number.isFinite(best.loss)) {
    const diff = val.loss - best.loss;
    const bestLabel = best.step >= 1000 && best.step % 1000 === 0 ? `${best.step / 1000}k` : fmt(best.step, true);
    if (Math.abs(diff) < 1e-9 || val.step === best.step) {
      valNote = `★ best @${bestLabel}`;
      valDeltaClass = "delta-good";
    } else {
      valNote = `+${fmt(diff, true)} (@${bestLabel})`;
      valDeltaClass = "delta-bad";
    }
  } else if (val.step != null) {
    valNote = `@ ${fmt(val.step)}`;
  }

  // Gradient + LR combined readout
  const lrVal = train.lr ?? (train.lr_groups ? Object.values(train.lr_groups)[0] : null);
  const clippedNow =
    Number.isFinite(clipThr) && clipThr > 0 && Number.isFinite(train.grad_norm) && train.grad_norm > clipThr;
  const gradNote = [
    Number.isFinite(lrVal) ? `lr ${fmt(lrVal, true)}` : null,
    clippedNow ? `> C(${fmt(clipThr, true)})` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  // 5th cell: Generalization gap if validation exists, else throughput
  let fifthTitle = "gapOrSpeed",
    fifthVal = "—",
    fifthNote = "",
    fifthHelp = "derived/generalization_gap";
  if (Number.isFinite(val.generalization_gap)) {
    fifthTitle = "generalization_gap";
    fifthVal = (val.generalization_gap > 0 ? "+" : "") + fmt(val.generalization_gap, true);
    fifthNote = "paired";
    fifthHelp = "generalization_gap";
  } else if (Number.isFinite(val.loss) && Number.isFinite(tailMean)) {
    const g = val.loss - tailMean;
    fifthTitle = "derived/generalization_gap";
    fifthVal = (g > 0 ? "+" : "") + fmt(g, true);
    fifthNote = "val − train";
  } else if (Number.isFinite(train.samples_per_s)) {
    fifthTitle = "samplesPerS";
    fifthVal = fmt(train.samples_per_s, true);
    fifthNote = "smp/s";
    fifthHelp = "derived/samples_per_s";
  }

  const cells = [
    {
      title: t("step"),
      help: explain("step"),
      badge: statusTag,
      value: fmt(train.step, false),
      note: (r.steps ? `/ ${fmt(r.steps)}${pct}` : "") + (r.stitched ? " · stitched" : ""),
    },
    {
      title: t("trainLoss"),
      help: explain("trainLoss"),
      value: fmt(tailMean ?? train.loss, true),
      note: tailMean != null && Number.isFinite(train.loss) ? `last ${fmt(train.loss, true)}` : "",
    },
    {
      title: t("valLoss"),
      help: explain("valLoss"),
      value: fmt(val.loss, true),
      note: valNote,
      noteClass: valDeltaClass,
    },
    {
      title: t("gradAndLr"),
      help: explain("grad_norm"),
      value: fmt(train.grad_norm, true),
      note: gradNote,
      noteClass: clippedNow ? "delta-bad" : "",
    },
    {
      title: t(fifthTitle),
      help: explain(fifthHelp),
      value: fifthVal,
      note: fifthNote,
    },
  ];

  grid.replaceChildren();
  for (const c of cells) {
    const cell = el("div");
    const dt = el("dt");
    const lbl = el("span", null, c.title);
    if (c.help) describe(lbl, c.help);
    dt.append(lbl);
    if (c.badge) dt.append(c.badge);
    const dd = el("dd");
    dd.append(document.createTextNode(c.value));
    if (c.note) dd.append(el("small", c.noteClass || null, c.note));
    cell.append(dt, dd);
    grid.append(cell);
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
    describe($("axis").previousElementSibling, explain(axis));
  }
}

function chooseRange(low, high) {
  if (axis !== "elapsed_s") {
    low = Math.ceil(low);
    high = Math.floor(high);
  }
  if (high < low) return;
  const full = extent();
  range = full && low <= full[0] && high >= full[1] ? null : [low, high];
  refresh();
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
    "generalizationGroup",
    all.filter((k) =>
      ["derived/generalization_gap", "generalization_gap", "train_eval_loss", "train_eval_samples"].includes(k),
    ),
  );
  group(
    "optimizer",
    all.filter((k) => /^(lr($|\/)|grad_|skipped_updates$|nonfinite_count$|derived\/clip_fraction$)/.test(k)),
  );
  group(
    "systemGroup",
    all.filter((k) =>
      /^(derived\/(samples_per_s|step_s|data_wait_ratio)|samples_per_s|step_s|dataloading_s|update_s|gpu_mem_gb)$/.test(k),
    ),
  );
  group(
    "probe",
    all.filter((k) => k.startsWith("module/") && !policy.has(k)),
  );
  const categorized = new Set(
    [...options.querySelectorAll("label")].map((l) => l.dataset.search.split(" ")[0]),
  );
  group(
    "trainingMetrics",
    all.filter((k) => !categorized.has(k.toLowerCase())),
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
  const params = new URLSearchParams({
    axis,
    points: budget,
    smooth: String(smooth),
    stitch: stitchResumes ? "1" : "0",
  });
  keys.forEach((k) => params.append("metric", k));
  if (range) {
    params.set("min", range[0]);
    params.set("max", range[1]);
  }
  const delta = !stitchResumes && old && old.key === queryKey && old.extra < budget / 4;
  if (delta) {
    for (const phase of ["train", "validation"]) {
      if (old.phases[phase]?.cursor) params.set("after_" + phase, old.phases[phase].cursor);
    }
  }
  const data = await api("runs/" + r.id + "/series", params, signal);
  if (signal.aborted) return false;
  let extra = 0,
    appended = false;
  for (const phase of ["train", "validation"]) {
    const p = data[phase];
    if (!p || p.mode !== "append" || !delta) continue;
    let added = 0;
    for (const group of ["series", "smoothed"]) {
      for (const [k, series] of Object.entries(p[group] || {})) {
        added = Math.max(added, series.points.length);
        const previous = old.phases[phase]?.[group]?.[k];
        if (previous) {
          series.points = previous.points.concat(series.points);
          series.count += previous.count;
        }
      }
    }
    if (!p.bounds && old.phases[phase]?.bounds) p.bounds = old.phases[phase].bounds;
    appended = true;
    extra = Math.max(extra, added);
  }
  snapshots.set(r.id, {
    key: queryKey,
    version: r.version,
    phases: data,
    extra: appended ? old.extra + extra : 0,
  });
  return true;
}

async function fetchEvents(r, signal) {
  const key = signature([r.version, stitchResumes]);
  const old = eventsCache.get(r.id);
  if (old?.key === key) return old.events;
  const data = await api("runs/" + r.id + "/events", { stitch: stitchResumes ? "1" : "0" }, signal);
  if (signal.aborted) return old?.events || [];
  const list = data.events || [];
  eventsCache.set(r.id, { key, events: list });
  return list;
}

function seriesLines(key, index, runList, phasesList, opts = {}) {
  const curSmooth = opts.smooth ?? smooth;
  const curShowRaw = opts.showRaw ?? showRaw;
  const lines = [];
  for (const [i, r] of runList.entries()) {
    const data = phasesList[i];
    if (!data) continue;
    for (const phase of ["train", "validation"]) {
      const p = data[phase];
      const series = p?.series?.[key];
      if (!series?.count) continue;
      const smoothed = curSmooth > 0 ? p.smoothed?.[key] : null;
      const hasSmooth = Boolean(smoothed?.points?.some((pt) => pt[1] != null));
      const base = runList.length === 1 ? paletteColor(phase === "validation" ? (index + 2) % 6 : index) : paletteColor(i);
      const prefix = runList.length > 1 ? runName(r) + " · " : "";
      if (!hasSmooth || curShowRaw || phase === "validation") {
        lines.push({
          ...series,
          id: r.id + "/" + phase,
          color: base,
          displayColor: base,
          faint: hasSmooth && phase === "train",
          dash: phase === "validation",
          label: prefix + t(hasSmooth && phase === "train" ? "rawLine" : phase),
        });
      }
      if (hasSmooth) {
        lines.push({
          ...smoothed,
          id: r.id + "/ema",
          trend: true,
          color: base,
          displayColor: base,
          label: `${prefix}${t("emaLine")} (${curSmooth.toFixed(2)})`,
        });
      }
    }
  }
  return lines;
}

function renderLossShareStrip(r, policyKeys) {
  const strip = $("loss-share-strip");
  if (!r || selection().length > 1 || policyKeys.length < 2) {
    strip.hidden = true;
    return;
  }
  const trainPhase = r.phases.train || {};
  const latest = trainPhase.latest || {};
  const means = trainPhase.tail_component_means || {};
  const fallbackMap = { ...(latest.metrics || {}), ...latest };
  const items = policyKeys
    .map((k) => ({ key: k, val: Number.isFinite(means[k]) ? means[k] : fallbackMap[k] }))
    .filter((it) => Number.isFinite(it.val) && it.val >= 0);
  const total = Number.isFinite(trainPhase.tail_mean_loss) ? trainPhase.tail_mean_loss : latest.loss;
  const sum = Number.isFinite(total) && total > 0 ? total : items.reduce((a, b) => a + b.val, 0);
  if (items.length < 2 || !(sum > 0)) {
    strip.hidden = true;
    return;
  }
  items.sort((a, b) => b.val - a.val);
  strip.hidden = false;
  strip.replaceChildren();
  const head = el("div", "share-head");
  const title = el("strong", null, t("lossShareTitle"));
  describe(title, explain("loss", null, r));
  head.append(title, el("span", "detail number", `${t("loss")}: ${fmt(sum, true)}`));
  const bar = el("div", "share-bar");
  const leg = el("div", "share-legend");
  const normSum = items.reduce((a, b) => a + b.val, 0) || sum;
  items.forEach((it, idx) => {
    const pct = (100 * it.val) / normSum;
    const seg = el("span");
    seg.style.width = `${pct}%`;
    seg.style.background = paletteColor(idx);
    bar.append(seg);
    const chip = el("span");
    const dot = el("i", "plot-tip-dot");
    dot.style.background = paletteColor(idx);
    dot.style.display = "inline-block";
    dot.style.marginRight = "5px";
    chip.append(dot, document.createTextNode(`${t(it.key)} `), el("b", null, `${fmtPct(pct)} (${fmt(it.val, true)})`));
    leg.append(chip);
  });
  strip.append(head, bar, leg);
}

function classifyGroup(key, policySet) {
  if (key === "loss" || policySet.has(key)) return "policyMetrics";
  if (
    ["derived/generalization_gap", "generalization_gap", "train_eval_loss", "train_eval_samples"].includes(key)
  )
    return "generalizationGroup";
  if (/^(lr($|\/)|grad_|skipped_updates$|nonfinite_count$|derived\/clip_fraction$)/.test(key))
    return "optimizer";
  if (
    /^(derived\/(samples_per_s|step_s|data_wait_ratio)|samples_per_s|step_s|dataloading_s|update_s|gpu_mem_gb)$/.test(key)
  )
    return "systemGroup";
  if (key.startsWith("module/")) return "probe";
  return "trainingMetrics";
}

function drawCharts(keys) {
  const box = $("charts");
  box.querySelector(".empty-state")?.remove();
  const needle = metricFilter.trim().toLowerCase();
  const visibleKeys = needle
    ? keys.filter((k) => (k + " " + t(k)).toLowerCase().includes(needle))
    : keys;

  for (const [key, plot] of charts) {
    if (!visibleKeys.includes(key)) {
      plot.destroy();
      charts.delete(key);
    }
  }
  const runList = selection(),
    policySet = new Set(runList.flatMap(policyMetrics));
  renderLossShareStrip(runList[0], [...policySet]);

  const groupOrder = [
    "policyMetrics",
    "generalizationGroup",
    "optimizer",
    "systemGroup",
    "probe",
    "trainingMetrics",
  ];
  const groups = new Map();
  for (const [index, key] of visibleKeys.entries()) {
    const g = classifyGroup(key, policySet);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push([key, index]);
  }
  const orderedGroups = groupOrder.filter((g) => groups.has(g)).map((g) => [g, groups.get(g)]);
  for (const section of [...box.children]) {
    if (!groups.has(section.dataset.group)) section.remove();
  }
  if (!visibleKeys.length) {
    box.append(el("div", "empty-state", needle ? t("noMatches") : t("noneYet")));
    return;
  }

  const primaryEvents = showEvents ? eventsCache.get(runList[0]?.id)?.events || [] : [];
  let groupIndex = 0;
  for (const [group, items] of orderedGroups) {
    let section = box.querySelector('[data-group="' + group + '"]');
    if (!section) {
      section = el("section", "chart-section");
      section.dataset.group = group;
      const toggle = el("button", "section-toggle");
      toggle.type = "button";
      toggle.innerHTML = `<svg class="icon"><use href="#icon-chevron" /></svg>`;
      const titleSpan = el("span", "section-name");
      const countSpan = el("span", "count");
      toggle.append(titleSpan, countSpan);
      toggle.onclick = () => {
        if (collapsedSections.has(group)) collapsedSections.delete(group);
        else collapsedSections.add(group);
        localStorage.setItem("am:collapsedSections", JSON.stringify([...collapsedSections]));
        section.classList.toggle("collapsed", collapsedSections.has(group));
      };
      section.append(toggle, el("div", "chart-grid"));
    }
    section.classList.toggle("collapsed", collapsedSections.has(group));
    if (box.children[groupIndex] !== section) {
      box.insertBefore(section, box.children[groupIndex] || null);
    }
    groupIndex++;
    section.querySelector(".section-name").textContent = t(group);
    section.querySelector(".count").textContent = String(items.length);
    const grid = section.lastChild;

    items.forEach(([key, index], position) => {
      let plot = charts.get(key);
      if (!plot) {
        plot = new Plot(
          grid,
          chooseRange,
          () => {
            const idx = plot.seriesIndex,
              curRuns = selection(),
              curEvs = eventsCache.get(curRuns[0]?.id)?.events || [],
              curRefs = refsForMetric(key, curRuns);
            inspectChart({
              key,
              description: metricDescription(key),
              runs: curRuns,
              axis,
              bounds: extent(),
              initial: range,
              smooth,
              yScale,
              showRaw,
              showEvents,
              stitch: stitchResumes,
              events: curEvs,
              refs: curRefs,
              linesFor: (phases, opts) => seriesLines(key, idx, curRuns, phases, opts),
              accent: paletteColor(idx),
            });
          },
          {
            syncGroup: "metrics-overview",
            onReset: () => {
              if (range) {
                range = null;
                refresh();
              }
            },
          },
        );
        charts.set(key, plot);
      }
      if (grid.children[position] !== plot.card) {
        grid.insertBefore(plot.card, grid.children[position] || null);
      }
      plot.seriesIndex = index;
      const phasesList = runList.map((r) => snapshots.get(r.id)?.phases);
      const lines = seriesLines(key, index, runList, phasesList);
      const bList = phasesList
        .flatMap((p) => [p?.train?.bounds?.[key], p?.validation?.bounds?.[key]])
        .filter(Boolean);
      const robustBounds = bList.length
        ? [Math.min(...bList.map((b) => b[0])), Math.max(...bList.map((b) => b[1]))]
        : null;
      const cardEvents =
        key === "loss" || key === "grad_norm" || key === "derived/generalization_gap"
          ? primaryEvents
          : primaryEvents.filter((e) => e.type === "resume" || e.type === "warmup_end");
      plot.update({
        title: t(key),
        description: metricDescription(key),
        axis,
        lines,
        events: cardEvents,
        refs: refsForMetric(key, runList),
        robustBounds,
        accent: paletteColor(index),
        range: range || extent(),
        yScale,
      });
    });
  }
}

function refsForMetric(key, runList) {
  const refs = [];
  if (key === "grad_norm") {
    const clip = runList[0]?.config?.optimizer?.grad_clip_norm;
    if (Number.isFinite(clip) && clip > 0) {
      refs.push({ value: clip, label: `clip C=${fmt(clip, true)}` });
    }
  } else if (key === "derived/generalization_gap" || key === "generalization_gap") {
    refs.push({ value: 0, label: "0" });
  }
  return refs;
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
    diffOnly = $("config-diff").checked,
    list = selection();
  $("config-diff-wrap").hidden = list.length < 2;
  const meta = signature([lang, list.map((r) => [r.id, r.metadata_version])]);
  const stamp = signature([meta, filter, diffOnly]);
  if (stamp === configKey) return;
  configKey = stamp;
  const tbl = $("config-table");
  tbl.replaceChildren();
  const head = tbl.createTHead().insertRow();
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
  const body = tbl.createTBody();
  for (const k of keys) {
    const fact = facts.map((m) => m.get(k)).find(Boolean);
    const label = fact?.label || k;
    const rawVals = data.map((d, i) => facts[i].get(k)?.value ?? d.get(k) ?? "—");
    const differs = list.length > 1 && new Set(rawVals).size > 1;
    if (diffOnly && list.length > 1 && !differs) continue;
    const searchable = [
      k,
      label,
      ...rawVals,
      ...facts.flatMap((m) => {
        const item = m.get(k);
        return item ? [...item.details.flat(), ...item.aliases] : [];
      }),
    ];
    if (filter && !searchable.some((v) => v != null && String(v).toLowerCase().includes(filter)))
      continue;
    const row = body.insertRow();
    if (differs) row.className = "diff-row";
    const name = el("td", null, label);
    const helpText = explain(fact?.helpKey || k);
    if (helpText) describe(name, helpText);
    else if (fact) name.title = k;
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
  const text = (previous + more).slice(-64000);
  const output = $("log-output");
  const stick = !cursor || output.scrollHeight - output.scrollTop - output.clientHeight < 25;
  if (!cursor || value.reset || more) output.textContent = text.replace(/\r\n?/g, "\n") || t("logEmpty");
  if (stick) output.scrollTop = output.scrollHeight;
  logState = { key, cursor: value.cursor, text, sources: sig };
}

function revealTab() {
  const active = tabs.querySelector('[aria-current="page"]');
  if (!active || tabs.scrollWidth <= tabs.clientWidth) return;
  const tabBox = active.getBoundingClientRect(),
    bar = tabs.getBoundingClientRect();
  if (tabBox.left < bar.left) tabs.scrollLeft += tabBox.left - bar.left;
  else if (tabBox.right > bar.right) tabs.scrollLeft += tabBox.right - bar.right;
}
new ResizeObserver(revealTab).observe(tabs);

function switchTab(next) {
  if (next !== tab) window.scrollTo(0, 0);
  tab = next;
  if (tab !== "metrics") $("display-settings").hidden = true;
  const url = new URL(location.href);
  if (next === "metrics") url.searchParams.delete("view");
  else url.searchParams.set("view", next);
  history.replaceState(null, "", url);
  document.querySelectorAll("[data-tab]").forEach((n) => {
    if (n.dataset.tab === tab) n.setAttribute("aria-current", "page");
    else n.removeAttribute("aria-current");
    $(n.dataset.tab + "-view").hidden = n.dataset.tab !== tab;
  });
  revealTab();
  if (next !== "evaluation") {
    evaluation.clear();
    evalKey = "";
  }
  if (next !== "compare") compareView.clearPlots();
  if (primary()) renderHead(primary());
  refresh();
}

function hasActiveRun() {
  return selection().some((r) => r.status === "running");
}

function showLive() {
  const active = hasActiveRun();
  $("live-label").textContent = t(!live ? "paused" : active ? "follow" : "idleLive");
  $("live").dataset.state = !live ? "paused" : active ? "active" : "idle";
  $("live").setAttribute("aria-pressed", String(live));
  $("live").title = t(live ? "pauseHint" : "resumeHint");
  $("live-icon").setAttribute("href", live ? "#icon-pause" : "#icon-play");
  showSidebar();
  $("y-scale").replaceChildren(
    opt("fit", t("scale_fit")),
    opt("robust", t("scale_robust")),
    opt("log", t("scale_log")),
    opt("zero", t("scale_zero")),
  );
  $("y-scale").value = yScale;
  describe($("y-scale").previousElementSibling, explain("yScale"));
  describe($("smooth-row").firstElementChild, explain("smoothing"));
  $("live").setAttribute("aria-label", $("live").title);
  $("language-label").textContent = lang === "zh" ? "English" : "中文";
  revealTab();
}

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
  for (const map of [info, snapshots, eventsCache])
    for (const id of map.keys()) if (!selected.includes(id)) map.delete(id);
  const empty = !selected.length;
  $("empty").hidden = !empty;
  $("import").hidden = empty;
  if (empty) {
    renderList();
    lastHeader = "";
    $("display-settings").hidden = true;
    $("run-view").hidden = true;
    $("feedback").hidden = true;
    return;
  }

  await Promise.all(
    selected.map(async (id) => {
      const item = runs.find((r) => r.id === id),
        prev = info.get(id);
      if (prev?.version === item.version && prev._stitched === stitchResumes) return;
      const brief = prev?.metadata_version === item.metadata_version && prev._stitched === stitchResumes;
      const params = { stitch: stitchResumes ? "1" : "0" };
      if (brief) params.brief = "1";
      const value = await api("runs/" + id, params, signal);
      if (!signal.aborted)
        info.set(id, {
          ...prev,
          ...value,
          version: item.version,
          metadata_version: item.metadata_version,
          _stitched: stitchResumes,
        });
    }),
  );
  if (signal.aborted) return;
  renderList();
  showLive();

  const r = primary();
  $("run-view").hidden = false;
  $("stitch-pill").hidden = !selection().some((run) => run.has_resumes);
  const headerSig = signature([r.id, r.version, lang, tab, stitchResumes]);
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
    $("display-settings").hidden = !all.length;
    const keys = (metrics || defaultMetrics()).filter((k) => all.includes(k));
    const budget = getBudget();
    const queryKey = signature([axis, range, keys, budget, smooth, stitchResumes]);
    const [changed] = await Promise.all([
      Promise.all(selection().map((run) => fetchSeries(run, keys, queryKey, budget, signal))),
      showEvents && r ? fetchEvents(r, signal) : Promise.resolve([]),
    ]);
    if (signal.aborted) return;
    const paint = signature([
      queryKey,
      lang,
      selected,
      r.version,
      yScale,
      showRaw,
      showEvents,
      metricFilter,
    ]);
    if (lastPaint !== paint || changed.some(Boolean)) {
      const full = extent();
      $("interval-tools").hidden = !full;
      overviewRange.update(full, range, axis, selected.join("/"));
      drawCharts(keys);
      lastPaint = paint;
    }
  } else if (tab === "diagnostics") {
    await diagnostics.render(selection(), runName, signal, { axis, range, stitch: stitchResumes });
  } else if (tab === "compare") {
    await compareView.render(selected, signal, { stitch: stitchResumes });
  } else if (tab === "config") {
    renderConfig();
  } else if (tab === "logs") {
    await renderLogs(r, signal);
  } else if (tab === "evaluation") {
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
      if (request === c && live && !document.hidden) {
        // Fast poll (1.8s) while a run is active; gentle poll (8s) when all selected runs are idle
        const interval = hasActiveRun() ? 1800 : 8000;
        timer = setTimeout(refresh, interval);
      }
    });
}

// Event wiring
$("live").onclick = () => {
  live = !live;
  showLive();
  if (live) refresh();
  else {
    clearTimeout(timer);
    request?.abort();
  }
};
$("theme").onclick = () => {
  toggleTheme();
};
onThemeChange(() => {
  lastPaint = "";
  if (tab === "metrics" && primary()) {
    const all = unionMetrics();
    drawCharts((metrics || defaultMetrics()).filter((k) => all.includes(k)));
  }
});
$("language").onclick = () => {
  toggleLang();
  setUrl();
  translate();
  showLive();
  updateSmoothFill();
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
$("y-scale").onchange = (e) => {
  yScale = e.target.value;
  localStorage.setItem("am:yScale", yScale);
  lastPaint = "";
  refresh();
};

const smoothSlider = $("smooth-slider");
smoothSlider.value = smooth.toFixed(2);
function updateSmoothFill() {
  const val = smoothSlider.valueAsNumber;
  const frac = val / Number(smoothSlider.max);
  smoothSlider.style.setProperty("--width-fill", frac * 100 + "%");
  $("smooth-value").textContent = val > 0 ? val.toFixed(2) : t("off");
}
smoothSlider.oninput = () => {
  smooth = smoothSlider.valueAsNumber;
  updateSmoothFill();
};
smoothSlider.onchange = () => {
  localStorage.setItem("am:smooth", smooth.toFixed(2));
  refresh();
};
updateSmoothFill();

const chartWidth = $("chart-width");
const storedWidth = Number(localStorage.getItem("am:chart-width"));
if (storedWidth >= Number(chartWidth.min) && storedWidth <= Number(chartWidth.max))
  chartWidth.value = storedWidth;
function sizeCharts() {
  const width = chartWidth.valueAsNumber;
  const fraction = (width - Number(chartWidth.min)) / (Number(chartWidth.max) - Number(chartWidth.min));
  chartWidth.style.setProperty("--width-fill", fraction * 100 + "%");
  $("charts").style.setProperty("--chart-width", width + "px");
  $("width-value").textContent = width + " px";
  chartWidth.setAttribute("aria-valuetext", width + " px");
}
chartWidth.oninput = sizeCharts;
chartWidth.onchange = () => localStorage.setItem("am:chart-width", chartWidth.value);
sizeCharts();

$("toggle-raw").checked = showRaw;
$("toggle-raw").onchange = () => {
  showRaw = $("toggle-raw").checked;
  localStorage.setItem("am:showRaw", showRaw ? "1" : "0");
  lastPaint = "";
  refresh();
};
$("toggle-events").checked = showEvents;
$("toggle-events").onchange = () => {
  showEvents = $("toggle-events").checked;
  localStorage.setItem("am:showEvents", showEvents ? "1" : "0");
  lastPaint = "";
  refresh();
};
$("toggle-stitch").checked = stitchResumes;
$("toggle-stitch").onchange = () => {
  stitchResumes = $("toggle-stitch").checked;
  localStorage.setItem("am:stitch", stitchResumes ? "1" : "0");
  snapshots.clear();
  eventsCache.clear();
  info.clear();
  lastPaint = "";
  lastHeader = "";
  refresh();
};

$("metric-filter").oninput = () => {
  metricFilter = $("metric-filter").value;
  lastPaint = "";
  if (tab === "metrics" && primary()) {
    const all = unionMetrics();
    drawCharts((metrics || defaultMetrics()).filter((k) => all.includes(k)));
  }
};
$("metrics-btn").onclick = metricDialog;
$("search").oninput = renderList;
$("config-filter").oninput = renderConfig;
$("config-diff").onchange = renderConfig;
$("compare-all-attempts").onchange = () => refresh();
$("compare-seed-band").onchange = () => refresh();
$("log-source").onchange = () => {
  logState.key = $("log-source").value;
  logState.cursor = "";
  refresh();
};
$("menu").onclick = () => sidebar();
$("close-sidebar").onclick = () => sidebar();
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
