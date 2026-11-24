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
  trend,
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
  y.append(opt("fit", t("autoScale")), opt("zero", t("zeroScale")));
  let range = normalize(initial || bounds),
    controller,
    lines = [];
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
  const ylabel = el("label", "axis", t("yScale"));
  ylabel.append(y);
  controls.append(zoomIn, zoomOut, earlier, later, back, full, ylabel);
  body.append(controls);
  const plot = new Plot(body, (lo, hi) => set([lo, hi]));
  const slider = rangeSlider((lo, hi) => set([lo, hi]));
  status.setAttribute("role", "status");
  body.append(slider.element, status);
  function paint() {
    plot.update({ title: t(key), description, axis, lines, accent, range, zeroY: y.value === "zero" });
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
      mean_window: trend,
      points: Math.min(4096, Math.max(512, Math.round(plot.area.clientWidth * devicePixelRatio * 2))),
    });
    try {
      const phases = await Promise.all(
        runs.map((r) => api("runs/" + r.id + "/series", params, current.signal)),
      );
      if (current.signal.aborted) return;
      lines = linesFor(phases);
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
