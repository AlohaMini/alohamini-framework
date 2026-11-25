import { $, api, btn, el, modal, opt, t } from "./ui.js";
import { Plot } from "./plot.js";
import { rangeSlider } from "./range.js";

export function inspectChart({
  key,
  description,
  runs,
  axis,
  bounds,
  initial,
  smooth: initialSmooth,
  yScale: initialYScale,
  showRaw: initialShowRaw,
  showEvents: initialShowEvents,
  stitch,
  events,
  refs,
  linesFor,
  accent,
}) {
  if (!bounds) return;
  const body = modal(t(key)),
    dialog = $("dialog");
  dialog.classList.add("inspector");
  const controls = el("div", "controls"),
    history = [],
    status = el("span", "detail");
  const y = el("select");
  y.setAttribute("aria-label", t("yScale"));
  y.append(
    opt("fit", t("scale_fit")),
    opt("robust", t("scale_robust")),
    opt("log", t("scale_log")),
    opt("zero", t("scale_zero")),
  );
  y.value = initialYScale || "fit";

  let smooth = Number.isFinite(initialSmooth) ? initialSmooth : 0.6;
  let showRaw = initialShowRaw !== false;
  let showEvents = initialShowEvents !== false;
  let range = normalize(initial || bounds),
    controller,
    lines = [],
    robustBounds = null;

  function normalize([lo, hi]) {
    const width = Math.max(axis === "elapsed_s" ? 0.001 : 1, Math.min(hi - lo, bounds[1] - bounds[0]));
    const low = Math.max(bounds[0], Math.min(lo, bounds[1] - width)),
      high = Math.min(bounds[1], low + width);
    return axis === "elapsed_s" ? [low, high] : [Math.floor(low), Math.ceil(high)];
  }
  function set(next, remember = true) {
    const value = normalize(next);
    if (value[0] === range[0] && value[1] === range[1]) {
      slider.update(bounds, range, axis);
      return;
    }
    if (remember) history.push(range);
    range = value;
    load();
  }
  function scale(factor) {
    const center = (range[0] + range[1]) / 2,
      half = ((range[1] - range[0]) * factor) / 2;
    set([center - half, center + half]);
  }
  function move(sign) {
    const d = (sign * (range[1] - range[0])) / 2;
    set([range[0] + d, range[1] + d]);
  }

  const back = btn(t("undo"), () => set(history.pop(), false)),
    full = btn(t("resetRange"), () => set(bounds));
  const earlier = btn(t("previous"), () => move(-1)),
    later = btn(t("next"), () => move(1));
  const zoomIn = btn("＋", () => scale(0.5)),
    zoomOut = btn("−", () => scale(2));
  for (const [button, label] of [
    [zoomIn, "zoomIn"],
    [zoomOut, "zoomOut"],
  ]) {
    button.title = t(label);
    button.setAttribute("aria-label", t(label));
  }

  // Independent EMA smoothing slider in modal
  const smoothLabel = el("label", "axis", t("smoothing"));
  const smoothInput = el("input");
  smoothInput.type = "range";
  smoothInput.min = "0";
  smoothInput.max = "0.99";
  smoothInput.step = "0.01";
  smoothInput.value = String(smooth);
  smoothInput.style.width = "88px";
  const smoothOut = el("span", "number", smooth.toFixed(2));
  smoothLabel.append(smoothInput, smoothOut);
  smoothInput.oninput = () => {
    smooth = smoothInput.valueAsNumber;
    smoothOut.textContent = smooth.toFixed(2);
  };
  smoothInput.onchange = load;

  // Show raw & events checkboxes
  const rawPill = el("label", "check-pill");
  const rawBox = el("input");
  rawBox.type = "checkbox";
  rawBox.checked = showRaw;
  rawBox.onchange = () => {
    showRaw = rawBox.checked;
    load();
  };
  rawPill.append(rawBox, el("span", null, t("showRaw")));

  const evPill = el("label", "check-pill");
  const evBox = el("input");
  evBox.type = "checkbox";
  evBox.checked = showEvents;
  evBox.onchange = () => {
    showEvents = evBox.checked;
    paint();
  };
  evPill.append(evBox, el("span", null, t("showEvents")));

  // Export PNG & CSV
  const pngBtn = btn(t("exportPng"), () => {
    const a = document.createElement("a");
    a.href = plot.toPngDataUrl();
    a.download = `${(runs[0]?.name || "chart").replace(/[^a-zA-Z0-9._-]/g, "_")}-${key.replace(/[^a-zA-Z0-9._-]/g, "_")}.png`;
    a.click();
  });
  const csvBtn = btn(t("exportCsv"), () => {
    const r = runs[0];
    if (!r) return;
    const params = new URLSearchParams({
      metric: key,
      axis,
      min: range[0],
      max: range[1],
      stitch: stitch ? "1" : "0",
    });
    const a = document.createElement("a");
    a.href = `./api/v1/runs/${r.id}/export.csv?${params}`;
    a.download = "";
    a.click();
  });

  const ylabel = el("label", "axis", t("yScale"));
  ylabel.append(y);
  controls.append(
    zoomIn,
    zoomOut,
    earlier,
    later,
    back,
    full,
    ylabel,
    smoothLabel,
    rawPill,
    evPill,
    pngBtn,
    csvBtn,
  );
  body.append(controls);
  const plot = new Plot(body, (lo, hi) => set([lo, hi]), null, { onReset: () => set(bounds) });
  const slider = rangeSlider((lo, hi) => set([lo, hi]));
  status.setAttribute("role", "status");
  body.append(slider.element, status);

  function paint() {
    plot.update({
      title: t(key),
      description,
      axis,
      lines,
      events: showEvents ? events : [],
      refs,
      robustBounds,
      accent,
      range,
      yScale: y.value,
    });
  }

  async function load() {
    controller?.abort();
    const current = new AbortController();
    controller = current;
    const width = range[1] - range[0],
      total = bounds[1] - bounds[0];
    zoomOut.disabled = full.disabled = width >= total;
    zoomIn.disabled = width <= (axis === "elapsed_s" ? 0.001 : 1);
    earlier.disabled = range[0] <= bounds[0];
    later.disabled = range[1] >= bounds[1];
    back.disabled = !history.length;
    slider.update(bounds, range, axis);
    status.hidden = false;
    status.textContent = t("loading");
    const params = new URLSearchParams({
      metric: key,
      axis,
      min: range[0],
      max: range[1],
      smooth: String(smooth),
      stitch: stitch ? "1" : "0",
      points: Math.min(4096, Math.max(512, Math.round(plot.area.clientWidth * devicePixelRatio * 2))),
    });
    try {
      const phases = await Promise.all(
        runs.map((r) => api("runs/" + r.id + "/series", params, current.signal)),
      );
      if (current.signal.aborted) return;
      lines = linesFor(phases, { smooth, showRaw });
      const bList = phases
        .flatMap((p) => [p.train?.bounds?.[key], p.validation?.bounds?.[key]])
        .filter(Boolean);
      robustBounds = bList.length
        ? [Math.min(...bList.map((b) => b[0])), Math.max(...bList.map((b) => b[1]))]
        : null;
      paint();
      status.hidden = true;
    } catch (e) {
      if (!current.signal.aborted) status.textContent = t("error") + " · " + e.message;
    }
  }
  y.onchange = paint;
  dialog.addEventListener(
    "close",
    () => {
      controller?.abort();
      plot.destroy();
      dialog.classList.remove("inspector");
    },
    { once: true },
  );
  load();
}
