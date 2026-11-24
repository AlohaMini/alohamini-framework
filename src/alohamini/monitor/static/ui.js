// Shared display strings, metric definitions and DOM helpers.
const words = {
  compareHint: ["Select a run to inspect. Tick others to compare.", "点击查看任务，勾选叠加对比。"],
  compareLimit: ["Compare up to six attempts at a time.", "可同时对比六次训练。"],
  lastEvent: ["Last record", "最近记录"],
  fileUpdated: ["Files updated", "文件更新"],
  runs: ["Training runs", "训练任务"],
  searchRuns: ["Find a run", "搜索训练任务"],
  localWorkspace: ["Local workspace", "本地工作区"],
  workspace: ["Workspace", "工作区"],
  monitor: ["Training", "训练监控"],
  overview: ["Training overview", "训练总览"],
  addEval: ["Add evaluation data", "导入评估数据"],
  noRuns: ["Waiting for a training run", "等待训练任务"],
  noRunsHint: [
    "New runs appear here automatically. You can also open an existing output directory.",
    "新训练会自动出现在这里，也可以打开已有的训练输出目录。",
  ],
  views: ["Run views", "运行视图"],
  metrics: ["Metrics", "指标"],
  evaluation: ["Evaluation", "评估"],
  config: ["Configuration", "配置"],
  logs: ["Logs", "日志"],
  curves: ["Metric curves", "指标曲线"],
  axis: ["X axis", "横轴"],
  range: ["Range", "区间"],
  chooseMetrics: ["Metrics", "选择指标"],
  resetRange: ["Full history", "完整历史"],
  savedConfig: ["Run configuration", "训练配置"],
  filterConfig: ["Filter parameters", "筛选参数"],
  logStream: ["Live log", "实时日志"],
  logSource: ["Log source", "日志来源"],
  close: ["Close", "关闭"],
  follow: ["Auto-refresh", "自动刷新"],
  paused: ["Paused", "已暂停"],
  pauseHint: ["Pause updates", "暂停刷新"],
  resumeHint: ["Resume updates", "恢复刷新"],
  compare: ["Compare", "对比"],
  primary: ["Primary run", "主任务"],
  initial: ["Initial run", "初始训练"],
  resume: ["Resume from", "续训起点"],
  noMatches: ["No matching runs", "没有匹配的任务"],
  step: ["Optimizer step", "优化器步数"],
  samples: ["Samples processed", "已处理样本"],
  elapsed_s: ["Elapsed time (s)", "经过时间（秒）"],
  train: ["Training", "训练"],
  validation: ["Validation", "验证"],
  loss: ["Objective loss", "目标损失"],
  trainLoss: ["Training loss", "训练损失"],
  valLoss: ["Validation loss", "验证损失"],
  grad_norm: ["Gradient norm", "梯度范数"],
  lr: ["Learning rate", "学习率"],
  dataloading_s: ["Data loading (s)", "数据加载（秒）"],
  update_s: ["Update time (s)", "更新耗时（秒）"],
  gpu_mem_gb: ["Peak GPU memory (GiB)", "峰值显存（GiB）"],
  grad_clipped: ["Gradient clipping", "梯度裁剪"],
  skipped_updates: ["Skipped updates", "跳过更新次数"],
  loss_l1: ["Action L1 loss", "动作 L1 损失"],
  loss_kl: ["KL divergence", "KL 散度"],
  loss_kl_weighted: ["Weighted KL loss", "加权 KL 损失"],
  loss_action: ["Action loss", "动作损失"],
  loss_video: ["Video loss", "视频损失"],
  latestStep: ["Latest step", "最新步数"],
  stepAt: ["Step", "步数"],
  budget: ["Budget", "训练预算"],
  unavailable: ["No record", "暂无记录"],
  noData: ["No points in this range", "当前区间没有记录点"],
  sampled: ["Extrema retained", "保留极值"],
  raw: ["Raw points", "原始点"],
  points: ["points", "点"],
  selectHint: ["Drag to zoom. Use ← and → to inspect points.", "拖动选择区间，使用左右方向键查看记录点。"],
  lower: ["From", "起点"],
  upper: ["To", "终点"],
  apply: ["Apply", "应用"],
  invalidRange: ["Enter an increasing, finite range.", "请输入有限且递增的区间。"],
  noneYet: ["Waiting for metric records…", "等待指标写入…"],
  error: ["Could not load data", "无法读取数据"],
  recordIssues: ["Malformed records", "格式异常记录"],
  nonfinite: ["Missing or non-finite scalar values", "缺失或非有限标量值"],
  configKey: ["Parameter", "参数"],
  rawTrain: ["Training · JSONL", "训练 · JSONL"],
  rawVal: ["Validation · JSONL", "验证 · JSONL"],
  logEmpty: ["Waiting for log records…", "等待日志记录…"],
  dataset: ["Evaluation dataset", "评估数据集"],
  datasets: ["Evaluation recordings", "评估录制"],
  selectDataset: ["Select a local dataset", "选择本地数据集"],
  folderHint: [
    "Add a directory outside the workspace at launch with --datasets-dir.",
    "工作区外的数据目录可在启动时用 --datasets-dir 添加。",
  ],
  add: ["Add dataset", "添加数据集"],
  noDataset: ["No datasets found in the configured directories.", "指定目录中暂无数据集。"],
  offlineFile: ["Import an offline evaluation report", "导入离线评估报告"],
  fileHint: [
    "Choose the JSON report produced by offline evaluation.",
    "选择框架离线评估输出的 JSON 报告。",
  ],
  reportBad: [
    "Expected a JSON report with finite, nonnegative mae_by_action values (up to 4 MiB).",
    "请选择包含有限非负 mae_by_action 数值的 JSON 报告（最大 4 MiB）。",
  ],
  remove: ["Remove from view", "移出视图"],
  episodes: ["Episodes", "回合"],
  frames: ["Frames", "帧数"],
  fps: ["FPS", "帧率"],
  frame: ["Frame index", "帧索引"],
  episode: ["Episode", "回合"],
  actionField: ["Action channel", "动作通道"],
  trajectory: ["Recorded trajectory", "录制轨迹"],
  action: ["Recorded action", "记录动作"],
  "observation.state": ["Observed state", "观测状态"],
  recordedUnit: ["Unit unspecified", "单位未声明"],
  task: ["Task", "任务"],
  manifest_match: ["Checkpoint manifest matches", "Checkpoint 清单一致"],
  manifest_different: ["Checkpoint manifest differs", "Checkpoint 清单不同"],
  manifest_unverified: ["Checkpoint association unverified", "Checkpoint 关联未核实"],
  offline: ["Offline action error", "离线动作误差"],
  finalReport: ["Run directory report", "运行目录评估报告"],
  validSteps: ["Valid action steps", "有效动作步"],
  channel: ["Action channel", "动作通道"],
  evalEmpty: ["Add a recording or offline report", "添加评估录制或离线报告"],
  evalHint: [
    "Inspect recorded actions and checkpoint provenance alongside this training run.",
    "可结合当前训练任务查看录制动作、checkpoint 来源和离线误差。",
  ],
  imported: ["Imported report", "导入报告"],
  records: ["Records", "记录数"],
  loading: ["Loading…", "读取中…"],
  params: ["Parameters", "参数"],
  scale: ["Scale", "倍率"],
  denominator: ["Denominator", "分母"],
  reduction: ["Reduction", "聚合"],
  maxMetrics: ["Select 1–32 metrics.", "请选择 1 至 32 项指标。"],
  scope: ["Current attempt", "本次训练"],
  weights: ["Weights", "权重"],
  unnamed: ["Run", "任务"],
  source: ["Source", "来源"],
  noConfig: ["No saved parameters match.", "没有匹配的已保存参数。"],
};
const help = {
  loss_l1: [
    "Absolute action reconstruction error in the policy’s training representation.",
    "在策略训练表示下计算的动作绝对重建误差。",
  ],
  loss_kl: [
    "KL divergence between latent distributions before applying the loss weight.",
    "潜变量分布之间的 KL 散度，按原始尺度记录。",
  ],
  loss_kl_weighted: [
    "KL term after applying its configured objective weight.",
    "KL 项乘以配置权重后对目标损失的贡献。",
  ],
  loss_action: [
    "Action prediction term emitted by the policy. OpenWAM includes its action weight.",
    "策略输出的动作预测损失项。OpenWAM 的记录已含动作损失权重。",
  ],
  loss_video: [
    "Video prediction term emitted by the policy. OpenWAM includes its video weight.",
    "策略输出的视频预测损失项。OpenWAM 的记录已含视频损失权重。",
  ],
  policy: ["Policy implementation used in this attempt.", "本次训练使用的策略实现。"],
  dataset: ["Dataset root resolved by the trainer.", "训练器解析的数据集根目录。"],
  train_episodes: ["Episode indices included in the training split.", "纳入训练集的回合索引。"],
  val_episodes: ["Held-out episode indices used for validation.", "用于验证的留出回合索引。"],
  num_epochs: ["Requested training passes through the sampler.", "配置的采样器训练轮数。"],
  trainable_dtype: ["Storage dtype of trainable parameters.", "可训练参数的存储精度。"],
  world_size: ["Total number of participating training processes.", "实际参与分布式训练的进程总数。"],
  num_workers: [
    "Data loader worker processes per training process.",
    "每个训练进程使用的数据加载工作进程数。",
  ],
  eval_batch_size: ["Maximum examples per validation batch.", "每个验证批次的最大样本数。"],
  normalization: [
    "Source of normalization statistics for model inputs and outputs.",
    "模型输入、输出归一化统计量的来源。",
  ],
  "scheduler.warmup_steps": [
    "Optimizer steps allocated to learning-rate warmup.",
    "用于学习率预热的优化器更新步数。",
  ],
  "optimizer.type": ["Optimizer used to update trainable parameters.", "更新可训练参数的优化器类型。"],
  step: [
    "Number of successful global optimizer updates. Resumed runs retain checkpoint step numbers.",
    "已成功执行的全局优化器更新次数。续训沿用 checkpoint 步数。",
  ],
  samples: [
    "Cumulative samples consumed across ranks, including distributed padding and partial batches.",
    "所有进程累计消耗的样本数，包含分布式补齐和尾批次。",
  ],
  elapsed_s: [
    "Monotonic seconds since this training loop started, including validation and checkpoint pauses.",
    "从本次训练循环开始的单调时钟秒数，包含验证和保存 checkpoint 的时间。",
  ],
  loss: [
    "The objective recorded by the policy. Training and validation may include different terms.",
    "策略记录的优化目标。训练与验证可能使用不同的损失项。",
  ],
  trainLoss: [
    "Objective of the latest successful global optimizer update, with the policy’s loss weights.",
    "最近一次成功更新的全局目标损失，包含策略设定的损失权重。",
  ],
  valLoss: [
    "Latest held-out evaluation using policy-defined denominators. ACT validation excludes the KL term.",
    "最近一次留出集评估，分母由策略定义。ACT 验证不含 KL 项。",
  ],
  grad_norm: [
    "Global gradient norm before clipping, after mixed-precision unscaling.",
    "混合精度梯度取消缩放后、裁剪前的全局梯度范数。",
  ],
  lr: [
    "Learning rate used for this update in optimizer parameter group 0, before scheduler advancement.",
    "本次更新中优化器第 0 个参数组实际使用的学习率，记录于调度器推进前。",
  ],
  grad_clipped: [
    "1 when the pre-clip norm exceeded the configured positive threshold; otherwise 0.",
    "裁剪前范数超过配置的正阈值时记 1，其他情况记 0。",
  ],
  skipped_updates: [
    "Cumulative AMP-skipped optimizer attempts in this loop as of the recorded successful step.",
    "截至该成功更新，本次训练循环因 AMP 溢出而跳过的优化器更新总数。",
  ],
  gpu_mem_gb: [
    "Peak tensor memory allocated on the logging CUDA rank during this update, in GiB.",
    "本次更新期间日志进程所在 CUDA 设备的峰值张量分配量，单位 GiB。",
  ],
  dataloading_s: [
    "Seconds spent fetching microbatches on the logging rank for this update.",
    "日志进程为本次更新读取微批次所用的秒数。",
  ],
  update_s: [
    "Elapsed update time on the logging rank after fetching data, including retries on these batches.",
    "日志进程取得数据后的更新耗时，包含对本组批次的重试，单位为秒。",
  ],
  batch_size: ["Maximum examples per device and microbatch.", "每个设备上单个微批次的最大样本数。"],
  gradient_accumulation_steps: [
    "Number of microbatches accumulated before an optimizer update.",
    "单次优化器更新前累计的微批次数。",
  ],
  "optimizer.lr": [
    "Configured base learning rate before scheduler scaling.",
    "应用调度器缩放前配置的基础学习率。",
  ],
  "optimizer.grad_clip_norm": [
    "Maximum global gradient norm; nonpositive disables clipping.",
    "全局梯度范数阈值，非正值表示关闭裁剪。",
  ],
  "optimizer.weight_decay": [
    "Weight decay coefficient used by the configured optimizer.",
    "当前优化器使用的权重衰减系数。",
  ],
  seed: ["Seed configured for training random-number generators.", "训练随机数生成器的配置种子。"],
  mixed_precision: [
    "Autocast precision used for training computations.",
    "训练计算采用的自动混合精度模式。",
  ],
  steps: ["Target number of successful optimizer updates.", "目标成功优化器更新次数。"],
  eval_steps: ["Successful optimizer updates between validation passes.", "两次验证之间的成功更新步数。"],
  save_freq: ["Optimizer steps between checkpoints.", "保存 checkpoint 的优化器步数间隔。"],
  log_freq: ["Interval for console summaries, in optimizer steps.", "控制台汇总的步数间隔。"],
};
export const $ = (id) => document.getElementById(id);
const requested =
  new URL(location.href).searchParams.get("lang") || localStorage.getItem("am:lang") || navigator.language;
export let lang = requested.startsWith("zh") ? "zh" : "en";
export function toggleLang() {
  lang = lang === "zh" ? "en" : "zh";
  localStorage.setItem("am:lang", lang);
}
export function t(key) {
  return words[key]?.[lang === "zh" ? 1 : 0] ?? key;
}
export function fmt(v, compact = false) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a && (a < 0.001 || a >= 1e8)) return v.toExponential(2);
  return v.toLocaleString(lang, { maximumSignificantDigits: compact ? 4 : 6 });
}
export function explain(key, spec, config = false) {
  const i = lang === "zh" ? 1 : 0;
  let text = help[key]?.[i] || "";
  if (key.startsWith("lr/"))
    text = [
      `Learning rate used by parameter group ${key.slice(3)} for this update.`,
      `本次更新中参数组 ${key.slice(3)} 的实际学习率。`,
    ][i];
  if (spec) {
    const parts = [
      spec.source,
      spec.reduction && t("reduction") + " " + spec.reduction,
      spec.denominator && t("denominator") + " " + spec.denominator,
      spec.scale != null && t("scale") + " " + spec.scale,
    ];
    return [text, parts.filter(Boolean).join(" · ")].filter(Boolean).join("\n");
  }
  return (
    text ||
    (config
      ? ["", ""]
      : [
          "Scalar from the original metric record. The producing policy defines its scale.",
          "原始指标记录中的标量，数值尺度由输出它的策略定义。",
        ])[i]
  );
}
export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}
export function opt(value, text) {
  return new Option(text, value);
}
export function btn(text, action, cls = "secondary") {
  const node = el("button", cls, text);
  node.type = "button";
  node.onclick = action;
  return node;
}
export function hint(text) {
  const node = el("button", "hint", "?");
  node.type = "button";
  node.dataset.help = text;
  node.setAttribute("aria-label", text);
  return node;
}
export function translate() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-t]").forEach((n) => (n.textContent = t(n.dataset.t)));
  document.querySelectorAll("[data-ph]").forEach((n) => (n.placeholder = t(n.dataset.ph)));
  document
    .querySelectorAll("[data-label]")
    .forEach((n) => n.setAttribute("aria-label", t(n.dataset.label)));
}
export async function api(path, params = new URLSearchParams(), signal) {
  const response = await fetch("./api/v1/" + path + "?" + new URLSearchParams(params), {
    signal,
    cache: "no-cache",
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
export function modal(title) {
  $("dialog-title").textContent = title;
  $("dialog-body").replaceChildren();
  if (!$("dialog").open) $("dialog").showModal();
  return $("dialog-body");
}
export function showError(error) {
  if (error?.name === "AbortError") return;
  $("feedback").textContent = t("error") + " · " + error.message;
  $("feedback").hidden = false;
}
function showHelp(event) {
  const target = event.target.closest?.("[data-help]");
  if (!target) return;
  const box = $("help"),
    rect = target.getBoundingClientRect();
  box.textContent = target.dataset.help;
  box.hidden = false;
  box.style.left = Math.max(8, Math.min(rect.left, innerWidth - box.offsetWidth - 8)) + "px";
  box.style.top =
    (rect.bottom + box.offsetHeight + 8 < innerHeight
      ? rect.bottom + 5
      : Math.max(6, rect.top - box.offsetHeight - 5)) + "px";
}
function hideHelp() {
  $("help").hidden = true;
}
for (const event of ["pointerover", "focusin"]) document.addEventListener(event, showHelp);
for (const event of ["pointerout", "focusout", "scroll"]) document.addEventListener(event, hideHelp, true);
$("close-dialog").onclick = () => $("dialog").close();
$("dialog").onclick = (e) => {
  if (e.target === $("dialog") && !e.target.closest("#dialog-body")) {
    const r = $("dialog").getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
      $("dialog").close();
  }
};
