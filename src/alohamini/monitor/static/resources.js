import { $, btn, disclosure, el, facts, fmt, lang, t, table } from "./ui.js";

let version = "",
  revision = 0;
function stored(run) {
  try {
    return JSON.parse(localStorage.getItem("am:preflight:" + run.id));
  } catch {
    return null;
  }
}
function check(report, run) {
  if (typeof report?.passed === "boolean")
    return (
      typeof report.dataset === "string" && (!run.config?.dataset || report.dataset === run.config.dataset)
    );
  return (
    report?.schema_version === 1 &&
    typeof report.ok === "boolean" &&
    typeof report.complete === "boolean" &&
    (!report.policy || !run.policy || report.policy === run.policy) &&
    Array.isArray(report.resources) &&
    report.resources.every((r) => r && typeof r.name === "string" && typeof r.status === "string") &&
    Array.isArray(report.errors)
  );
}
function showReport(parent, report, source) {
  if (typeof report.passed === "boolean") {
    const detail = disclosure(
      parent,
      t("launchCheck") + " · " + t(report.passed ? "successfulCheck" : "failedCheck") + " · " + source,
    );
    const values = [
      ["episodes", report.episodes],
      ["frames", report.frames],
      ["train_episodes", report.train_episodes?.length],
      ["val_episodes", report.val_episodes?.length],
      ["training_samples", report.train_samples],
      ["validation_samples", report.val_samples],
      ["precision", report.checkpoint_dtypes && Object.keys(report.checkpoint_dtypes).join(" / ")],
      ["torch", report.torch],
      ["CUDA", report.cuda_version],
      ["cpuOnly", report.cpu_only_preflight === true ? t("yes") : null],
      ["gpuTest", report.gpu_training_test],
      ["stage", report.stage],
    ];
    facts(
      detail,
      values.map(([k, v]) => [k, typeof v === "number" ? fmt(v) : v]),
    );
    disclosure(detail, t("rawReport")).append(el("pre", null, JSON.stringify(report, null, 2)));
    return;
  }
  const state = !report.ok ? "failedCheck" : !report.complete ? "incompleteCheck" : "successfulCheck";
  const detail = disclosure(parent, t("resources") + " · " + t(state) + " · " + source);
  const rows = report.resources.map((r) => {
    const path = el(
      "span",
      "path",
      [...new Set([r.requested_path, ...(Array.isArray(r.paths) ? r.paths : [])])]
        .filter(Boolean)
        .join("\n"),
    );
    if (r.reason || r.error)
      path.append(el("p", null, r.reason || r.error?.message || JSON.stringify(r.error)));
    return [r.name, r.kind, r.status, r.validation, path];
  });
  table(detail, ["module", "kind", "status", "checkMode", "paths"], rows);
  for (const err of report.errors)
    detail.append(el("p", "notice", [err.stage, err.message].filter(Boolean).join(" · ")));
  disclosure(detail, t("rawReport")).append(el("pre", null, JSON.stringify(report, null, 2)));
}
function addReport(parent, run, redraw) {
  const file = el("input", "report-input");
  file.type = "file";
  file.accept = ".json,application/json";
  file.setAttribute("aria-label", t("importCheck"));
  file.hidden = true;
  const info = el("div", "resource-tools"),
    action = btn(t("importCheck"), () => file.click(), "quiet");
  const error = el("span", "detail");
  error.setAttribute("role", "status");
  const imported = stored(run);
  if (!run.preflight && !run.launch_preflight && !imported) info.append(el("span", "detail", t("noCheck")));
  info.append(action, file, error);
  if (imported)
    info.append(
      btn(
        t("remove") + " · " + imported.name,
        () => {
          localStorage.removeItem("am:preflight:" + run.id);
          revision++;
          redraw();
        },
        "quiet",
      ),
    );
  parent.append(info);
  file.onchange = async () => {
    const selected = file.files?.[0];
    if (!selected) return;
    try {
      if (selected.size > 4 * 1024 * 1024) throw Error(t("badCheck"));
      let report;
      try {
        report = JSON.parse(await selected.text());
      } catch {
        throw Error(t("badCheck"));
      }
      if (!check(report, run)) throw Error(t("badCheck"));
      localStorage.setItem("am:preflight:" + run.id, JSON.stringify({ name: selected.name, report }));
      revision++;
      redraw();
    } catch (e) {
      error.textContent = e.message;
    } finally {
      file.value = "";
    }
  };
  if (run.preflight && check(run.preflight, run)) showReport(parent, run.preflight, t("recorded"));
  if (run.launch_preflight) showReport(parent, run.launch_preflight, run.name + ".preflight.json");
  if (imported) showReport(parent, imported.report, t("supplied") + " · " + imported.name);
}
export function renderResources(runs, redraw) {
  const sig = JSON.stringify([lang, revision, runs.map((r) => [r.id, r.metadata_version])]);
  if (version === sig) return;
  version = sig;
  const root = $("run-resources");
  root.replaceChildren();
  for (const run of runs) {
    const cfg = run.config || {},
      rt = run.schema?.runtime || {},
      flags = rt.trainability;
    const old = run.console_parameters;
    const frozen = flags?.frozen ?? (old ? old.num_total_params - old.num_learnable_params : null);
    const total = flags ? flags.trainable + flags.frozen : old?.num_total_params;
    const report = run.preflight || run.launch_preflight;
    const checkLabel = report
      ? t(run.preflight ? "resources" : "launchCheck") +
        " · " +
        t(
          (report.ok ?? report.passed) === false
            ? "failedCheck"
            : report.complete === false
              ? "incompleteCheck"
              : "successfulCheck",
        )
      : null;
    const parent = disclosure(
      root,
      [
        runs.length > 1 ? run.name : null,
        t("runtime"),
        cfg.optimizer?.type,
        frozen != null && total ? t("frozen") + " " + fmt((100 * frozen) / total, true) + "%" : null,
        checkLabel,
      ]
        .filter(Boolean)
        .join("  ·  "),
    );
    facts(parent, [
      ["optimizer", cfg.optimizer?.type],
      ["precision", cfg.mixed_precision],
      ...[
        "training_samples",
        "validation_samples",
        "effective_batch_size",
        "excluded_samples",
        "torch",
        "accelerate",
      ]
        .filter((k) => rt[k] != null)
        .map((k) => [k, typeof rt[k] === "number" ? fmt(rt[k]) : rt[k]]),
    ]);
    if (flags) {
      const d = disclosure(
        parent,
        t("flags") +
          " · " +
          t("frozen") +
          " " +
          fmt(flags.frozen) +
          " / " +
          fmt(flags.trainable + flags.frozen),
      );
      table(
        d,
        ["module", "trainable", "frozen"],
        flags.modules.map((m) => [m.name, fmt(m.trainable), fmt(m.frozen)]),
      );
    } else {
      if (old)
        facts(parent, [
          ["trainable", fmt(old.num_learnable_params)],
          ["frozen", fmt(frozen)],
          ["source", t("console")],
        ]);
      parent.append(el("p", "detail", t("flagMissing")));
      const scope = [
        ...Object.entries(cfg.model || {}).filter(([k]) =>
          /freez|frozen|unfreeze|train_only|finetune_/i.test(k),
        ),
        ...Object.entries(cfg.paper_preset?.reference_recipe || {})
          .filter(([, v]) => typeof v === "string" && /\b(frozen|freeze)\b/i.test(v))
          .map(([k, v]) => ["paper_preset.reference_recipe." + k, v]),
      ];
      if (scope.length) {
        const d = disclosure(parent, t("configuredScope"));
        table(
          d,
          ["configKey", "setting"],
          scope.map(([k, v]) => [k, typeof v === "string" ? v : JSON.stringify(v)]),
        );
      }
    }
    if (rt.optimizer_groups?.length) {
      const d = disclosure(parent, t("rates"));
      const keys = [...new Set(rt.optimizer_groups.flatMap((g) => Object.keys(g)))];
      table(
        d,
        ["configKey", ...rt.optimizer_groups.map((_, i) => "#" + i)],
        keys.map((k) => [k, ...rt.optimizer_groups.map((g) => JSON.stringify(g[k]) ?? "—")]),
      );
    }
    addReport(parent, run, redraw);
  }
}
