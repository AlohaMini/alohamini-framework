import { $, api, btn, el, fmt, modal, opt, t } from "./ui.js";
import { colors, Plot } from "./plot.js";

function stored(id) {
  try {
    return JSON.parse(localStorage.getItem("am:eval:" + id)) || { datasets: [], reports: [] };
  } catch {
    return { datasets: [], reports: [] };
  }
}
function field(label, control) {
  const node = el("label", "field", label);
  node.append(control);
  return node;
}
function fact(parent, label, value) {
  const item = el("span");
  item.append(document.createTextNode(label + " "), el("b", null, value));
  parent.append(item);
}
function readReport(data, name) {
  const values = data?.mae_by_action;
  if (
    !values ||
    typeof values !== "object" ||
    Array.isArray(values) ||
    !Object.keys(values).length ||
    !Object.values(values).every((v) => typeof v === "number" && Number.isFinite(v) && v >= 0)
  ) {
    throw Error(t("reportBad"));
  }
  return {
    name,
    mae_by_action: values,
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
        data.reports = data.reports.filter((r) => r.name !== report.name);
        data.reports.push(report);
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
    const view = $("evaluation-view");
    view.replaceChildren();
    const h = el("div", "toolbar");
    h.append(el("h2", null, t("datasets")));
    view.append(h);
    const saved = stored(run.id);
    if (!saved.datasets.length) {
      const empty = el("div", "card evaluation-block");
      empty.append(
        el("h3", null, t("evalEmpty")),
        el("p", "detail", t("evalHint")),
        btn(t("addEval"), () => this.picker(run)),
      );
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
    const reports = [];
    if (run.offline?.mae_by_action) reports.push({ ...run.offline, name: t("finalReport") });
    reports.push(...saved.reports);
    if (reports.length) {
      const title = el("div", "toolbar");
      title.append(el("h2", null, t("offline") + " · MAE"));
      const wrap = el("div", "table-wrap"),
        table = el("table"),
        head = table.createTHead().insertRow();
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
      const rows = table.createTBody();
      const names = new Set(reports.flatMap((report) => Object.keys(report.mae_by_action)));
      for (const name of names) {
        const row = rows.insertRow();
        row.append(el("td", null, name));
        for (const report of reports) row.append(el("td", "number", fmt(report.mae_by_action[name])));
      }
      wrap.append(table);
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
    block.append(el("span", "tag", t("manifest_" + d.manifest_match)));
    if (d.checkpoint) block.append(el("p", "path", d.checkpoint));
    if (!d.episode_count || !d.fields.length) return;
    const controls = el("div", "toolbar"),
      group = el("div", "controls");
    const episode = el("input", "input");
    episode.type = "number";
    episode.step = "1";
    episode.min = "0";
    episode.style.width = "7em";
    const previous = this.choices.get(id) || {};
    episode.max = d.episode_ids?.at(-1) ?? d.episode_count - 1;
    episode.value = previous.episode ?? d.episode_ids?.[0] ?? 0;
    const select = document.createElement("select");
    d.fields.forEach((k) => select.add(opt(k, k)));
    if (d.fields.includes(previous.field)) select.value = previous.field;
    group.append(field(t("episode"), episode), field(t("actionField"), select));
    controls.append(group);
    block.append(controls);
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
      this.choices.set(id, { episode: ep, field: select.value });
      status.textContent = t("loading");
      plot.card.hidden = true;
      try {
        const data = await api(
          "datasets/" + id + "/trajectory",
          {
            episode: ep,
            field: select.value,
            points: Math.min(2048, Math.max(256, Math.round(block.clientWidth))),
          },
          signal,
        );
        if (signal.aborted || seq !== revision) return;
        const lines = Object.entries(data.series).map(([k, s], i) => ({
          ...s,
          color: colors[i],
          dash: i > 0,
          label: t(k),
        }));
        status.textContent = "";
        plot.card.hidden = false;
        plot.update({
          title: t("trajectory") + " · " + data.field,
          description: t("frame") + " · " + (data.unit || t("recordedUnit")),
          axis: "frame",
          unit: data.unit || t("recordedUnit"),
          lines,
          featured: true,
        });
      } catch (e) {
        if (!signal.aborted && seq === revision) status.textContent = e.message;
      }
    };
    select.onchange = fetchPlot;
    episode.onchange = fetchPlot;
    await fetchPlot();
  }
}
