import { btn, el, fmt, t } from "./ui.js";

// Native sliders share a scale. Drag previews stay local until a handle is released.
export function rangeSlider(onChange, onEdit) {
  const element = el("div", "interval-control"),
    label = el("span", "interval-label"),
    rail = el("div", "interval-rail"),
    readout = onEdit ? btn("", onEdit, "interval-values") : el("span", "interval-values");
  element.setAttribute("role", "group");
  element.append(label, rail, readout);
  rail.append(el("span", "interval-track"));
  const handles = [el("input"), el("input")];
  let dragStart, axis, scope, full, selected;

  function preview() {
    const [lo, hi] = handles.map((h) => h.valueAsNumber);
    const min = Number(handles[0].min),
      span = Number(handles[0].max) - min;
    const low = span ? (100 * (lo - min)) / span : 0,
      high = span ? (100 * (hi - min)) / span : 100;
    rail.style.setProperty("--low", low + "%");
    rail.style.setProperty("--high", high + "%");
    rail.style.setProperty("--split", (low + high) / 2 + "%");
    handles[0].setAttribute("aria-valuemax", hi);
    handles[1].setAttribute("aria-valuemin", lo);
    readout.textContent = lo === hi ? fmt(lo) : fmt(lo) + " – " + fmt(hi);
    if (onEdit) readout.setAttribute("aria-label", t("editRange") + " · " + readout.textContent);
  }
  function render() {
    const min = Math.min(full[0], selected[0]),
      max = Math.max(full[1], selected[1]);
    const digits = Math.max(fmt(min).length, fmt(max).length, axis === "elapsed_s" ? 8 : 1);
    readout.style.setProperty("--value-width", 2 * digits + 3 + "ch");
    label.textContent = t("range");
    element.setAttribute("aria-label", t("range") + " · " + t(axis));
    if (onEdit) readout.title = t("editRange");
    handles.forEach((h, i) => {
      h.min = min;
      h.max = max;
      h.step = axis === "elapsed_s" ? "any" : "1";
      h.value = selected[i];
      h.disabled = min === max;
      h.setAttribute("aria-label", t(axis) + " · " + t(i ? "upper" : "lower"));
    });
    preview();
  }
  handles.forEach((h, i) => {
    h.type = "range";
    rail.append(h);
    h.onpointerdown = () => {
      dragStart = handles.map((handle) => handle.value);
    };
    h.onpointerup = h.onblur = () => {
      const unchanged = dragStart?.every((value, index) => value === handles[index].value);
      dragStart = null;
      if (unchanged && full) render();
    };
    h.onpointercancel = () => {
      dragStart = null;
      if (full) render();
    };
    h.oninput = () => {
      const other = handles[1 - i].valueAsNumber;
      h.value = i ? Math.max(other, h.valueAsNumber) : Math.min(other, h.valueAsNumber);
      preview();
    };
    h.onchange = () => {
      dragStart = null;
      onChange(handles[0].valueAsNumber, handles[1].valueAsNumber);
    };
  });
  function update(bounds, range, nextAxis, nextScope = "") {
    if (scope !== nextScope || axis !== nextAxis || !bounds) dragStart = null;
    scope = nextScope;
    axis = nextAxis;
    full = bounds;
    selected = range || full;
    element.hidden = !full;
    if (full && !dragStart) render();
  }
  return { element, readout, update };
}
