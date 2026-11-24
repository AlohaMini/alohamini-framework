import { $, btn, el, fmt, t } from "./ui.js";

function saved(run) {
  try {
    return JSON.parse(localStorage.getItem("am:preflight:" + run.id));
  } catch {
    return null;
  }
}
function matches(report, run) {
  if (typeof report?.passed === "boolean")
    return (
      typeof report.dataset === "string" &&
      (!run.config?.dataset || report.dataset === run.config.dataset)
    );
  return (
    report?.schema_version === 1 &&
    typeof report.ok === "boolean" &&
    (!report.policy || !run.policy || report.policy === run.policy) &&
    (!report.resolved_settings?.dataset ||
      !run.config?.dataset ||
      report.resolved_settings.dataset === run.config.dataset) &&
    Array.isArray(report.resources) &&
    report.resources.every((r) => r && typeof r.name === "string" && typeof r.kind === "string")
  );
}
function shortPath(path) {
  const parts = path.replaceAll("\\", "/").replace(/\/+$/, "").split("/").filter(Boolean);
  if (!parts.length) return path;
  const end = parts.at(-1);
  if (parts.length < 4) return parts.join("/");
  return parts.slice(/^(assets|pretrained_model|model\.safetensors)$/i.test(end) ? -3 : -2).join("/");
}

// Values retain their recorded origin. Configuration fallbacks never query a live model or filesystem.
export function configurationFacts(run) {
  const rows = new Map(),
    cfg = run.config || {},
    rt = run.schema?.runtime || {};
  function add(key, label, value, details = [], aliases = []) {
    if (value == null || value === "" || rows.has(key)) return;
    rows.set(key, {
      label: t(label) + (key.startsWith("import.") ? " · " + t("supplied") : ""),
      value: String(value),
      details: details.filter(([, v]) => v != null && v !== ""),
      aliases,
    });
  }
  function read(report, source, imported = false) {
    if (!report || !matches(report, run)) return;
    const prefix = imported ? "import." : "";
    const origin = [t("source"), source];
    const base = report.initialization;
    if (typeof base?.repository === "string") {
      add(prefix + "base.model", "baseModel", base.repository, [
        [t("revision"), base.revision],
        [t("weights"), base.file],
        ["SHA-256", base.sha256],
        [t("path"), report.base_path],
        origin,
      ]);
    }
    if (typeof report.checkpoint === "string") {
      add(prefix + "weights.policy", "initialWeights", shortPath(report.checkpoint), [
        [t("path"), report.checkpoint],
        [t("headerHash"), report.checkpoint_header_sha256],
        origin,
      ]);
    }
    const resources = Array.isArray(report.resources)
      ? report.resources.filter((r) => r && typeof r.name === "string" && r.status !== "not_required")
      : [];
    for (const r of resources) {
      let kind = r.kind;
      if (!["tokenizer", "processor", "weights"].includes(kind)) {
        if (/tokenizer/i.test(r.name)) kind = "tokenizer";
        else if (/backbone|base_model/i.test(r.name)) kind = "weights";
        else continue;
      }
      const paths = Array.isArray(r.paths) ? r.paths.filter((p) => typeof p === "string") : [];
      const chosen = paths.length === 1 ? paths[0] : r.requested_path || paths[0];
      const label =
        kind === "weights" && r.name === "policy"
          ? "initialWeights"
          : kind === "weights"
            ? "modelWeights"
            : kind;
      const value = typeof chosen === "string" ? shortPath(chosen) : r.name;
      const detail = [
        [t("configKey"), r.name],
        [t("requestedPath"), r.requested_path],
        [t("resolvedFiles"), paths.join("\n")],
        [t("reportNote"), r.error?.message || r.reason],
        origin,
      ];
      const aliases = [];
      if (!imported && r.requested_path) {
        if (cfg.model?.[r.name] === r.requested_path) aliases.push("model." + r.name);
        if (r.name === "policy" && cfg.pretrained_path === r.requested_path)
          aliases.push("pretrained_path");
      }
      const shared = resources.some((other) => other !== r && other.kind === r.kind);
      add(
        prefix + kind + "." + r.name,
        shared && r.name !== "policy" ? t(label) + " · " + r.name : label,
        value,
        detail,
        aliases,
      );
    }
  }
  read(run.preflight, t("recorded"));
  read(run.launch_preflight, run.name + ".preflight.json");
  const checked = [...rows.keys()];
  const model = {
    ...run.recorded_asset_paths,
    ...run.preflight?.resolved_settings?.model,
    ...cfg.model,
  };
  for (const [field, value] of Object.entries(model)) {
    if (typeof value !== "string" || !value) continue;
    let kind;
    if (/tokenizer/.test(field) && /(path|name|model|_id)$/.test(field)) kind = "tokenizer";
    else if (field === "assets_path" && run.policy === "openwam") kind = "tokenizer";
    else if (/(base_model|backbone|vlm_model).*(name|path|id)$/.test(field)) kind = "baseModel";
    else continue;
    if (
      checked.includes(kind + "." + field) ||
      checked.includes("processor." + field) ||
      checked.includes("weights." + field) ||
      (kind === "baseModel" && rows.has("base.model"))
    )
      continue;
    const key = kind + "." + field;
    const source = cfg.model?.[field]
      ? "model." + field
      : run.preflight?.resolved_settings?.model?.[field]
        ? "resolved_settings.model." + field
        : "statistics.json · contract.model." + field;
    add(
      key,
      kind,
      shortPath(value),
      [
        [t("path"), value],
        [t("source"), source],
      ],
      cfg.model?.[field] === value ? ["model." + field] : [],
    );
  }
  if (!run.attempt && !cfg.resume && typeof cfg.pretrained_path === "string" && !rows.has("base.model"))
    add(
      "weights.policy",
      "initialWeights",
      shortPath(cfg.pretrained_path),
      [
        [t("path"), cfg.pretrained_path],
        [t("source"), "pretrained_path"],
      ],
      ["pretrained_path"],
    );
  const imported = saved(run);
  if (imported) read(imported.report, t("supplied") + " · " + imported.name, true);

  const flags = rt.trainability,
    old = run.console_parameters;
  if (flags || old) {
    const total = flags ? flags.trainable + flags.frozen : old.num_total_params;
    const trained = flags ? flags.trainable : old.num_learnable_params,
      frozen = total - trained;
    const breakdown = [
      [t("trainable"), fmt(trained)],
      [t("allParameters"), fmt(total)],
      [t("source"), flags ? t("flags") : t("console")],
    ];
    for (const m of flags?.modules || [])
      breakdown.push([
        m.name,
        t("trainable") + " " + fmt(m.trainable) + " · " + t("frozen") + " " + fmt(m.frozen),
      ]);
    add(
      "parameters.frozen",
      "frozenParams",
      fmt(frozen) + (total ? " (" + fmt((100 * frozen) / total, true) + "%)" : ""),
      breakdown,
    );
  }
  for (const key of [
    "effective_batch_size",
    "training_samples",
    "validation_samples",
    "excluded_samples",
    "torch",
    "accelerate",
  ])
    if (rt[key] != null && cfg[key] == null)
      add("runtime." + key, key, typeof rt[key] === "number" ? fmt(rt[key]) : rt[key]);
  for (const [i, group] of (rt.optimizer_groups || []).entries()) {
    add(
      "optimizer.group." + i,
      t("rates") + " " + i,
      group.lr != null ? t("lr") + " " + fmt(group.lr) : "#" + i,
      Object.entries(group).map(([k, v]) => [k, JSON.stringify(v)]),
    );
  }
  return rows;
}
export function factCell(item) {
  if (!item.details.length) return document.createTextNode(item.value);
  const node = el("details", "config-fact");
  node.append(el("summary", null, item.value));
  node.ontoggle = () => {
    if (!node.open || node.children.length > 1) return;
    const dl = el("dl");
    for (const [k, v] of item.details) dl.append(el("dt", null, k), el("dd", null, v));
    node.append(dl);
  };
  return node;
}

export function reportControls(run, onChange) {
  const root = $("config-import");
  root.replaceChildren();
  if (!run) return;
  const file = el("input");
  file.type = "file";
  file.accept = ".json,application/json";
  file.hidden = true;
  file.setAttribute("aria-label", t("importCheck"));
  const message = el("span", "detail");
  message.setAttribute("role", "status");
  const imported = saved(run);
  if (imported)
    root.append(
      btn(
        t("remove") + " · " + imported.name,
        () => {
          localStorage.removeItem("am:preflight:" + run.id);
          onChange();
        },
        "quiet",
      ),
    );
  const action = btn(t("importCheck"), () => file.click(), "quiet");
  action.title = run.name;
  root.append(action, file, message);
  file.onchange = async () => {
    const selected = file.files[0];
    if (!selected) return;
    try {
      if (selected.size > 4 * 1024 * 1024) throw Error(t("badCheck"));
      let report;
      try {
        report = JSON.parse(await selected.text());
      } catch {
        throw Error(t("badCheck"));
      }
      if (!matches(report, run)) throw Error(t("badCheck"));
      localStorage.setItem("am:preflight:" + run.id, JSON.stringify({ name: selected.name, report }));
      onChange();
    } catch (e) {
      message.textContent = e.message;
    } finally {
      file.value = "";
    }
  };
}
