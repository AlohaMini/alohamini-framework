import { $, api, btn, describe, el, explain, fmt, fmtPct, modal, opt, t, table } from "./ui.js";
import { paletteColor, Plot } from "./plot.js";

function stored(id) {
  try {
    return (
      JSON.parse(localStorage.getItem("am:eval:" + id)) || {
        datasets: [],
        reports: [],
        closedLoop: [],
      }
    );
  } catch {
    return { datasets: [], reports: [], closedLoop: [] };
  }
}
function field(label, control) {
  const node = el("label", "field", label);
  node.append(control);
  return node;
}
function fact(parent, label, value, helpKey = null) {
  const item = el("span");
  const lbl = el("span", null, label + " ");
  if (helpKey && explain(helpKey)) describe(lbl, explain(helpKey));
  item.append(lbl, el("b", null, value));
  parent.append(item);
}

function readReport(data, name) {
  if (!data || typeof data !== "object" || Array.isArray(data)) throw Error(t("reportBad"));
  // Support closed-loop rollout report JSON
  if (Number.isFinite(data.success_rate)) {
    return {
      kind: "closed_loop",
      name,
      success_rate: data.success_rate,
      episodes: data.episodes ?? data.num_episodes ?? null,
      task: typeof data.task === "string" ? data.task : null,
      checkpoint: typeof data.checkpoint === "string" ? data.checkpoint : null,
      failure_modes:
        data.failure_modes && typeof data.failure_modes === "object" ? data.failure_modes : null,
      notes: typeof data.notes === "string" ? data.notes : null,
    };
  }
  const values = data?.mae_by_action;
  const valid = (map, signed = false) =>
    map &&
    typeof map === "object" &&
    !Array.isArray(map) &&
    Object.keys(map).length &&
    Object.values(map).every((v) => Number.isFinite(v) && (signed || v >= 0));
  if (
    !valid(values) ||
    (data.bias_by_action != null && !valid(data.bias_by_action, true)) ||
    (data.rmse_by_action != null && !valid(data.rmse_by_action)) ||
    (data.p50_by_action != null && !valid(data.p50_by_action)) ||
    (data.p90_by_action != null && !valid(data.p90_by_action)) ||
    (data.p95_by_action != null && !valid(data.p95_by_action)) ||
    (data.by_horizon != null &&
      (!Array.isArray(data.by_horizon) ||
        !data.by_horizon.every(
          (h) =>
            Number.isInteger(h?.index) &&
            (h.offset == null || Number.isFinite(h.offset)) &&
            valid(h.mae_by_action),
        )))
  )
    throw Error(t("reportBad"));
  return {
    kind: "offline",
    name,
    mae_by_action: values,
    bias_by_action: data.bias_by_action,
    rmse_by_action: data.rmse_by_action,
    p50_by_action: data.p50_by_action,
    p90_by_action: data.p90_by_action,
    p95_by_action: data.p95_by_action,
    by_horizon: data.by_horizon,
    valid_action_steps: data.valid_action_steps,
    checkpoint: typeof data.checkpoint === "string" ? data.checkpoint : null,
    dataset: typeof data.dataset === "string" ? data.dataset : null,
    weights: data.weights,
  };
}

export class Evaluation {
  constructor(onChange) {
    this.onChange = onChange;
    this.plots = [];
    this.choices = new Map();
  }
  save(run, value) {
    localStorage.setItem("am:eval:" + run.id, JSON.stringify(value));
    this.onChange();
  }
  async picker(run) {
    const body = modal(t("addEval"));
    const select = document.createElement("select");
    const message = el("p", "detail", t("loading"));
    const add = btn(
      t("add"),
      () => {
        const data = stored(run.id);
        if (!data.datasets.includes(select.value)) data.datasets.push(select.value);
        $("dialog").close();
        this.save(run, data);
      },
      "primary",
    );
    add.disabled = true;
    body.append(field(t("selectDataset"), select), message, add, el("hr", "dialog-divider"));
    const file = document.createElement("input");
    file.type = "file";
    file.accept = ".json,application/json";
    file.className = "input";
    const error = el("p", "notice");
    error.hidden = true;
    file.onchange = async () => {
      try {
        const source = file.files[0];
        if (!source) return;
        if (source.size > 4 * 1024 * 1024) throw Error(t("reportBad"));
        let raw;
        try {
          raw = JSON.parse(await source.text());
        } catch {
          throw Error(t("reportBad"));
        }
        const report = readReport(raw, source.name);
        const data = stored(run.id);
        data.closedLoop = data.closedLoop || [];
        if (report.kind === "closed_loop") {
          data.closedLoop = data.closedLoop.filter((r) => r.name !== report.name);
          data.closedLoop.push(report);
        } else {
          data.reports = data.reports.filter((r) => r.name !== report.name);
          data.reports.push(report);
        }
        this.save(run, data);
        $("dialog").close();
      } catch (e) {
        error.hidden = false;
        error.textContent = e.message;
      }
    };
    body.append(field(t("offlineFile"), file), el("p", "detail", t("fileHint")), error);
    try {
      const { datasets } = await api("datasets");
      for (const d of datasets) select.add(opt(d.id, d.name));
      add.disabled = !datasets.length;
      message.textContent = (datasets.length ? "" : t("noDataset") + " ") + t("folderHint");
    } catch (e) {
      message.textContent = e.message;
    }
  }
  clear() {
    this.controller?.abort();
    this.controller = null;
    for (const p of this.plots) p.destroy();
    this.plots = [];
  }
  async render(run) {
    this.clear();
    this.controller = new AbortController();
    const signal = this.controller.signal;
    const view = $("evaluation-results");
    view.replaceChildren();
    const saved = stored(run.id);
    const closedLoop = saved.closedLoop || [];
    if (saved.datasets.length) {
      const h = el("div", "toolbar");
      h.append(el("h2", null, t("datasets")));
      view.append(h);
    }
    if (
      !saved.datasets.length &&
      !saved.reports.length &&
      !closedLoop.length &&
      !run.offline?.mae_by_action
    ) {
      const empty = el("div", "card evaluation-block");
      empty.append(el("h3", null, t("evalEmpty")), el("p", "detail", t("evalHint")));
      view.append(empty);
    }
    const recordings = saved.datasets.map((id) => {
      const block = el("div", "card evaluation-block");
      view.append(block);
      return this.recording(block, id, run, signal).catch((e) => {
        if (signal.aborted) return;
        block.replaceChildren(el("p", "notice", e.message), this.removeDataset(run, id));
      });
    });

    // Closed-loop evaluation reports
    if (closedLoop.length) {
      const h = el("div", "toolbar");
      h.append(el("h2", null, t("closedLoop")));
      view.append(h);
      for (const r of closedLoop) {
        const card = el("div", "card evaluation-block");
        const head = el("div", "section-title");
        head.append(
          el("h3", null, r.name),
          btn(
            t("remove"),
            () => {
              const data = stored(run.id);
              data.closedLoop = (data.closedLoop || []).filter((x) => x.name !== r.name);
              this.save(run, data);
            },
            "quiet",
          ),
        );
        const facts = el("div", "eval-facts");
        const pct = r.success_rate <= 1 ? r.success_rate * 100 : r.success_rate;
        fact(facts, t("successRate"), fmtPct(pct));
        if (r.episodes != null) fact(facts, t("episodes"), fmt(r.episodes));
        if (r.task) fact(facts, t("task"), r.task);
        card.append(head, facts);
        if (r.checkpoint) card.append(el("p", "path", r.checkpoint));
        if (r.failure_modes && Object.keys(r.failure_modes).length) {
          table(
            card,
            ["failureModes", "count"],
            Object.entries(r.failure_modes).map(([k, v]) => [k, fmt(v)]),
          );
        }
        view.append(card);
      }
    }

    // Offline action-error reports
    const reports = [];
    if (run.offline?.mae_by_action) reports.push({ ...run.offline, name: t("finalReport") });
    reports.push(...saved.reports);
    if (reports.length) {
      const title = el("div", "toolbar");
      title.append(el("h2", null, t("offline")));
      const controls = el("div", "controls"),
        measure = el("select"),
        horizon = el("select");
      const available = [
        "mae_by_action",
        "bias_by_action",
        "rmse_by_action",
        "p50_by_action",
        "p90_by_action",
        "p95_by_action",
      ].filter((k) => reports.some((r) => r[k]));
      measure.replaceChildren(...available.map((k) => opt(k, t(k))));
      measure.setAttribute("aria-label", t("reportMetric"));
      horizon.setAttribute("aria-label", t("horizon"));
      const offsets = [
        ...new Set(
          reports.flatMap((r) =>
            (r.by_horizon || []).filter((h) => h.offset != null).map((h) => h.offset),
          ),
        ),
      ].sort((a, b) => a - b);
      const indices = [
        ...new Set(
          reports.flatMap((r) =>
            (r.by_horizon || []).filter((h) => h.offset == null).map((h) => h.index),
          ),
        ),
      ].sort((a, b) => a - b);
      horizon.replaceChildren(
        opt("all", t("all")),
        ...offsets.map((v) => opt("offset:" + v, v)),
        ...indices.map((v) => opt("index:" + v, "#" + v)),
      );
      const hl = el("label", "axis", t("horizon"));
      hl.append(horizon);
      if (available.length > 1) controls.append(measure);
      else controls.append(el("span", "detail", t(available[0])));
      if (offsets.length + indices.length) controls.append(hl);
      title.append(controls);
      const wrap = el("div", "table-wrap"),
        tbl = el("table"),
        head = tbl.createTHead().insertRow();
      head.append(el("th", null, t("channel")));
      for (const report of reports) {
        const th = el("th", null, report.name);
        const facts = el("div", "eval-facts");
        fact(facts, t("validSteps"), fmt(report.valid_action_steps));
        if (report.weights) fact(facts, t("weights"), report.weights);
        th.append(facts);
        if (report.checkpoint || report.dataset) {
          const details = el("details");
          details.append(el("summary", null, t("source")));
          for (const path of [report.checkpoint, report.dataset])
            if (path) details.append(el("p", "path", path));
          th.append(details);
        }
        if (saved.reports.includes(report))
          th.append(
            btn(
              t("remove"),
              () => {
                const data = stored(run.id);
                data.reports = data.reports.filter((r) => r.name !== report.name);
                this.save(run, data);
              },
              "quiet",
            ),
          );
        head.append(th);
      }
      const rows = tbl.createTBody();
      function fill() {
        hl.hidden = measure.value !== "mae_by_action";
        const perHorizon = !hl.hidden && horizon.value !== "all";
        const [axis, value] = horizon.value.split(":");
        const maps = reports.map((r) =>
          perHorizon
            ? r.by_horizon?.find(
                (h) => (axis === "offset" || h.offset == null) && h[axis] === Number(value),
              )?.mae_by_action
            : r[measure.value],
        );
        rows.replaceChildren();
        const names = new Set(maps.flatMap((m) => Object.keys(m || {})));
        for (const name of names) {
          const row = rows.insertRow();
          row.append(el("td", null, name));
          for (const m of maps) row.append(el("td", "number", fmt(m?.[name])));
        }
      }
      measure.onchange = horizon.onchange = fill;
      fill();
      wrap.append(tbl);
      view.append(title, wrap);
    }
    await Promise.all(recordings);
  }
  removeDataset(run, id) {
    return btn(
      t("remove"),
      () => {
        const data = stored(run.id);
        data.datasets = data.datasets.filter((k) => k !== id);
        this.save(run, data);
      },
      "quiet",
    );
  }
  async recording(block, id, run, signal) {
    block.textContent = t("loading");
    const d = await api("datasets/" + id, { run: run.id }, signal);
    if (signal.aborted) return;
    block.replaceChildren();
    const head = el("div", "section-title");
    head.append(el("h3", null, d.name), this.removeDataset(run, id));
    const facts = el("div", "eval-facts");
    fact(facts, t("episodes"), fmt(d.episode_count));
    fact(facts, t("frames"), fmt(d.frames));
    fact(facts, t("fps"), fmt(d.fps));
    fact(facts, t("source"), d.format);
    block.append(head, facts);
    if (d.task) block.append(el("p", "detail", d.task));
    const matchCls =
      d.manifest_match === "match" ? "ok" : d.manifest_match === "different" ? "warn" : "";
    block.append(el("span", "tag " + matchCls, t("manifest_" + d.manifest_match)));
    if (d.checkpoint) block.append(el("p", "path", d.checkpoint));
    if (!d.episode_count || !d.fields.length) return;
    const controls = el("div", "toolbar"),
      group = el("div", "controls");
    const episode = el("input", "input");
    episode.type = "number";
    episode.step = "1";
    episode.min = "0";
    episode.style.width = "6.5em";
    const previous = this.choices.get(id) || {};
    episode.max = d.episode_ids?.at(-1) ?? d.episode_count - 1;
    episode.value = previous.episode ?? d.episode_ids?.[0] ?? 0;
    const select = document.createElement("select");
    d.fields.forEach((k) => select.add(opt(k, k)));
    if (d.fields.includes(previous.field)) select.value = previous.field;
    const modeSelect = document.createElement("select");
    for (const m of ["position", "residual", "velocity", "acceleration", "jerk"]) {
      modeSelect.add(opt(m, t("mode_" + m)));
    }
    modeSelect.value = previous.mode || "position";
    group.append(
      field(t("episode"), episode),
      field(t("actionField"), select),
      field(t("trajMode"), modeSelect),
    );
    controls.append(group);
    block.append(controls);
    const statsStrip = el("div", "eval-facts");
    block.append(statsStrip);
    const plot = new Plot(block);
    this.plots.push(plot);
    const status = el("p", "detail");
    block.append(status);
    let revision = 0;
    const fetchPlot = async () => {
      const seq = ++revision;
      const ep = episode.valueAsNumber;
      if (
        !Number.isInteger(ep) ||
        ep < 0 ||
        ep > Number(episode.max) ||
        (d.episode_ids && !d.episode_ids.includes(ep))
      ) {
        status.textContent = t("episode") + " · " + t("unavailable");
        plot.card.hidden = true;
        return;
      }
      this.choices.set(id, { episode: ep, field: select.value, mode: modeSelect.value });
      status.textContent = t("loading");
      plot.card.hidden = true;
      statsStrip.replaceChildren();
      try {
        const data = await api(
          "datasets/" + id + "/trajectory",
          {
            episode: ep,
            field: select.value,
            mode: modeSelect.value,
            points: Math.min(2048, Math.max(256, Math.round(block.clientWidth))),
          },
          signal,
        );
        if (signal.aborted || seq !== revision) return;
        const lines = Object.entries(data.series).map(([k, s], i) => ({
          ...s,
          id: k,
          color: paletteColor(i),
          dash: i > 0,
          label: t(k),
        }));
        const res = data.summary?.residual;
        const kin = data.summary?.kinematics;
        if (kin) {
          fact(statsStrip, t("rms"), fmt(kin.rms, true));
          fact(statsStrip, t("p95"), fmt(kin.p95_abs, true));
          fact(statsStrip, t("maxAbs"), fmt(kin.max_abs, true));
        } else if (res) {
          fact(statsStrip, t("bias"), (res.bias > 0 ? "+" : "") + fmt(res.bias, true));
          fact(statsStrip, t("mae"), fmt(res.mae, true));
          fact(statsStrip, t("rmse"), fmt(res.rmse, true));
          fact(statsStrip, t("p50"), fmt(res.p50, true));
          fact(statsStrip, t("p90"), fmt(res.p90, true));
          fact(statsStrip, t("p95"), fmt(res.p95, true));
          fact(statsStrip, t("maxAbs"), fmt(res.max_abs, true));
        }
        status.textContent = "";
        plot.card.hidden = false;
        const refs =
          modeSelect.value === "residual" ? [{ value: 0, label: "0" }] : [];
        plot.update({
          title: `${t("mode_" + modeSelect.value)} · ${data.field}`,
          description: `${t("frame")} · ${data.unit || t("recordedUnit")}`,
          axis: "frame",
          unit: data.unit || t("recordedUnit"),
          lines,
          refs,
        });
      } catch (e) {
        if (!signal.aborted && seq === revision) status.textContent = e.message;
      }
    };
    select.onchange = fetchPlot;
    episode.onchange = fetchPlot;
    modeSelect.onchange = fetchPlot;
    await fetchPlot();
  }
}
