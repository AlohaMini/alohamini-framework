import { btn, el, t } from "./ui.js";

// The two handles share one scale. Only completed edits request chart data.
export function rangeSlider(onChange, onReset) {
  const element = el("div", "interval-control"),
    heading = el("div", "interval-heading"),
    label = el("span"),
    fields = el("div", "interval-fields"),
    rail = el("div", "interval-rail"),
    error = el("p", "field-error");
  const inputs = [el("input", "input"), el("input", "input")],
    handles = [el("input"), el("input")];
  let full,
    selected,
    axis,
    scope,
    dragStart,
    draft = false;

  element.setAttribute("role", "group");
  heading.append(label);
  const reset =
    onReset &&
    btn(
      "",
      () => {
        discardDraft();
        onReset();
      },
      "quiet interval-reset",
    );
  if (reset) heading.append(reset);
  const dash = el("span", null, "–");
  dash.setAttribute("aria-hidden", "true");
  fields.append(inputs[0], dash, inputs[1]);
  rail.append(el("span", "interval-track"), ...handles);
  error.hidden = true;
  error.setAttribute("role", "status");
  element.append(heading, fields, rail, error);

  function clearError() {
    error.hidden = true;
    for (const input of inputs) input.removeAttribute("aria-invalid");
  }
  function discardDraft() {
    draft = false;
    clearError();
    if (full) render();
  }
  function commitFields() {
    if (!draft || !full) return;
    const [lo, hi] = inputs.map((input) => input.valueAsNumber);
    if (inputs.some((input) => !input.validity.valid) || lo > hi) {
      error.textContent = t("invalidRange");
      error.hidden = false;
      for (const input of inputs) input.setAttribute("aria-invalid", "true");
      return;
    }
    draft = false;
    clearError();
    if (lo !== selected[0] || hi !== selected[1]) onChange(lo, hi);
  }
  inputs.forEach((input) => {
    input.type = "number";
    input.required = true;
    input.oninput = () => {
      draft = true;
      clearError();
    };
    input.onkeydown = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitFields();
      } else if (event.key === "Escape" && draft) {
        event.preventDefault();
        discardDraft();
      }
    };
  });
  fields.addEventListener("focusout", (event) => {
    if (fields.contains(event.relatedTarget)) return;
    if (rail.contains(event.relatedTarget) || (reset && reset.contains(event.relatedTarget))) {
      discardDraft();
    } else commitFields();
  });

  function preview() {
    const [lo, hi] = handles.map((handle) => handle.valueAsNumber);
    const min = Number(handles[0].min),
      span = Number(handles[0].max) - min;
    const low = span ? (100 * (lo - min)) / span : 0,
      high = span ? (100 * (hi - min)) / span : 100;
    rail.style.setProperty("--low", low + "%");
    rail.style.setProperty("--high", high + "%");
    rail.style.setProperty("--split", (low + high) / 2 + "%");
    handles[0].setAttribute("aria-valuemax", hi);
    handles[1].setAttribute("aria-valuemin", lo);
    if (!draft) {
      inputs[0].value = lo;
      inputs[1].value = hi;
    }
  }
  function render() {
    const min = Math.min(full[0], selected[0]),
      max = Math.max(full[1], selected[1]);
    handles.forEach((handle, i) => {
      handle.min = min;
      handle.max = max;
      handle.value = selected[i];
      handle.disabled = min === max;
    });
    preview();
  }
  handles.forEach((handle, i) => {
    handle.type = "range";
    handle.onpointerdown = () => {
      if (draft) discardDraft();
      dragStart = handles.map((h) => h.value);
    };
    handle.onpointerup = handle.onblur = () => {
      const unchanged = dragStart?.every((value, index) => value === handles[index].value);
      dragStart = null;
      if (unchanged && full) render();
    };
    handle.onpointercancel = () => {
      dragStart = null;
      if (full) render();
    };
    handle.oninput = () => {
      const other = handles[1 - i].valueAsNumber;
      handle.value = i ? Math.max(other, handle.valueAsNumber) : Math.min(other, handle.valueAsNumber);
      preview();
    };
    handle.onchange = () => {
      dragStart = null;
      onChange(handles[0].valueAsNumber, handles[1].valueAsNumber);
    };
  });
  function update(bounds, range, nextAxis, nextScope = "") {
    if (scope !== nextScope || axis !== nextAxis || !bounds) {
      dragStart = null;
      draft = false;
      clearError();
    }
    scope = nextScope;
    axis = nextAxis;
    full = bounds;
    selected = range || full;
    element.hidden = !full;
    if (!full) return;
    label.textContent = t("range");
    element.setAttribute("aria-label", t("range") + " · " + t(axis));
    if (reset) {
      reset.hidden = !range;
      reset.textContent = t("resetRange");
    }
    inputs.forEach((input, i) => {
      const name = t(axis) + " · " + t(i ? "upper" : "lower");
      input.setAttribute("aria-label", name);
      input.title = name;
      handles[i].setAttribute("aria-label", name);
      input.step = handles[i].step = axis === "elapsed_s" ? "any" : "1";
    });
    if (!dragStart) render();
  }
  return { element, update };
}
