// Shared bilingual dictionary, professional metric guidance, theme manager and DOM helpers.
const words = {
  setting: ["Value", "配置值"],
  toggleSeries: ["Show or hide series", "显示或隐藏此曲线"],
  compareHint: ["Select a run to inspect. Tick others to compare.", "点击查看任务，勾选叠加对比。"],
  compareLimit: ["Compare up to six runs at a time.", "最多同时对比六次训练。"],
  lastEvent: ["Last record", "最近记录"],
  fileUpdated: ["Updated", "更新于"],
  runs: ["Training runs", "训练任务"],
  searchRuns: ["Find a run", "搜索训练任务"],
  filterCharts: ["Filter charts", "筛选曲线"],
  monitor: ["Training", "训练监控"],
  addEval: ["Add evaluation data", "导入评估数据"],
  noRuns: ["Waiting for a training run", "等待训练任务"],
  noRunsHint: [
    "New runs appear here automatically. You can also open an existing output directory.",
    "新训练会自动出现在这里，也可以打开已有的训练输出目录。",
  ],
  views: ["Run views", "运行视图"],
  metrics: ["Metrics", "指标"],
  diagnostics: ["Diagnostics", "诊断"],
  compareTab: ["Compare", "对比"],
  compareTitle: ["Experiment comparison (HParams)", "多实验超参与结果对比"],
  evaluation: ["Evaluation", "评估"],
  evaluationData: ["Evaluation data", "评估数据"],
  config: ["Configuration", "配置"],
  shortConfig: ["Config", "配置"],
  logs: ["Logs", "日志"],
  axis: ["X axis", "横轴"],
  range: ["Range", "区间"],
  chooseMetrics: ["Select metrics", "选择指标"],
  display: ["Chart settings", "图表设置"],
  smoothing: ["EMA smoothing", "EMA 平滑"],
  showRaw: ["Raw", "原始线"],
  showEvents: ["Events", "事件标线"],
  stitchResumes: ["Stitch resumes", "串联续训"],
  chartWidth: ["Min chart width", "最小图宽"],
  resetRange: ["Full history", "完整历史"],
  savedConfig: ["Run configuration", "训练配置"],
  filterConfig: ["Filter parameters", "筛选参数"],
  diffOnly: ["Differences only", "仅看差异项"],
  logStream: ["Live log", "实时日志"],
  logSource: ["Log source", "日志来源"],
  close: ["Close", "关闭"],
  follow: ["Live", "实时"],
  idleLive: ["Idle", "待机"],
  paused: ["Paused", "已暂停"],
  pauseHint: ["Pause auto-refresh", "暂停自动刷新"],
  resumeHint: ["Resume auto-refresh", "恢复自动刷新"],
  toggleTheme: ["Switch light/dark theme", "切换浅色／深色主题"],
  compare: ["Compare", "对比"],
  primary: ["Primary run", "主任务"],
  initial: ["Initial", "初始训练"],
  resume: ["Resume", "续训"],
  resumesCount: ["attempts", "次记录"],
  statusRunning: ["Running", "训练中"],
  statusCompleted: ["Done", "已完成"],
  statusIdle: ["Idle", "已停止"],
  noMatches: ["No matching runs", "没有匹配的任务"],
  step: ["Optimizer step", "优化器步数"],
  samples: ["Samples processed", "已处理样本"],
  elapsed_s: ["Elapsed time (s)", "经过时间（秒）"],
  train: ["Training", "训练"],
  validation: ["Validation", "验证"],
  loss: ["Objective loss", "目标损失"],
  trainLoss: ["Train loss (100-step mean)", "训练损失（近100步均值）"],
  valLoss: ["Validation loss", "验证损失"],
  gradAndLr: ["Grad norm & LR", "梯度范数与学习率"],
  gapOrSpeed: ["Gen. gap / Speed", "泛化差距／速度"],
  grad_norm: ["Gradient norm", "梯度范数"],
  lr: ["Learning rate", "学习率"],
  dataloading_s: ["Data loading (s)", "数据加载（秒）"],
  update_s: ["Update time (s)", "更新耗时（秒）"],
  step_s: ["Total step time (s)", "单步总耗时（秒）"],
  samples_per_s: ["Throughput (samples/s)", "吞吐量（样本/秒）"],
  gpu_mem_gb: ["Peak GPU memory (GiB)", "峰值显存（GiB）"],
  grad_clipped: ["Gradient clipped (0/1)", "梯度裁剪标志"],
  skipped_updates: ["Skipped updates", "跳过更新次数"],
  nonfinite_count: ["Non-finite count", "非有限值计数"],
  "derived/step_s": ["Step time (s)", "单步耗时（秒）"],
  "derived/data_wait_ratio": ["Data wait share", "数据等待占比"],
  "derived/samples_per_s": ["Throughput (samples/s)", "吞吐量（样本/秒）"],
  "derived/clip_fraction": ["Rolling clip rate (50 steps)", "滑动梯度裁剪率（50步）"],
  "derived/generalization_gap": ["Generalization gap (Val − Train)", "泛化差距（验证 − 训练）"],
  loss_l1: ["Action L1 loss", "动作 L1 损失"],
  loss_kl: ["KL divergence", "KL 散度"],
  loss_kl_weighted: ["Weighted KL loss", "加权 KL 损失"],
  loss_action: ["Action loss", "动作损失"],
  loss_video: ["Video loss", "视频损失"],
  stepAt: ["Step", "步数"],
  unavailable: ["No record", "暂无记录"],
  noData: ["No points in this range", "当前区间没有记录点"],
  sampled: ["Extrema", "保留极值"],
  raw: ["Raw", "原始点"],
  points: ["points", "点"],
  selectHint: [
    "Drag to zoom, double-click to reset, ←/→ to inspect.",
    "拖动放大，双击复位，左右方向键逐点查看。",
  ],
  lower: ["From", "起点"],
  upper: ["To", "终点"],
  apply: ["Apply", "应用"],
  invalidRange: [
    "Enter valid bounds with start ≤ end.",
    "请输入有效的起止值，起点不能大于终点。",
  ],
  noneYet: ["Waiting for metric records…", "等待指标写入…"],
  error: ["Could not load data", "无法读取数据"],
  recordIssues: ["Malformed records", "格式异常记录"],
  nonfinite: ["Non-finite scalar values", "非有限标量值"],
  configKey: ["Parameter", "参数"],
  rawTrain: ["Training · JSONL", "训练 · JSONL"],
  rawVal: ["Validation · JSONL", "验证 · JSONL"],
  logEmpty: ["Waiting for log records…", "等待日志记录…"],
  dataset: ["Evaluation dataset", "评估数据集"],
  datasets: ["Evaluation recordings", "评估录制"],
  selectDataset: ["Select a local dataset", "选择本地数据集"],
  folderHint: [
    "Add directories outside workspace at launch with --datasets-dir.",
    "工作区外的数据目录可在启动时用 --datasets-dir 添加。",
  ],
  add: ["Add dataset", "添加数据集"],
  noDataset: ["No datasets found in configured directories.", "指定目录中暂无数据集。"],
  offlineFile: ["Import offline / closed-loop evaluation JSON", "导入离线／闭环评估 JSON"],
  fileHint: [
    "Accepts offline evaluation reports (mae_by_action) or closed-loop rollout reports (success_rate).",
    "支持框架离线评估报告（mae_by_action）或闭环成功率报告（success_rate）。",
  ],
  reportBad: [
    "Expected a valid offline or closed-loop JSON report (up to 4 MiB).",
    "请选择有效的离线评估或闭环评估 JSON 报告（最大 4 MiB）。",
  ],
  remove: ["Remove", "移出"],
  episodes: ["Episodes", "回合"],
  frames: ["Frames", "帧数"],
  fps: ["FPS", "帧率"],
  frame: ["Frame index", "帧索引"],
  episode: ["Episode", "回合"],
  actionField: ["Action channel", "动作通道"],
  trajMode: ["Analysis", "分析视图"],
  mode_position: ["Position (Action vs State)", "位置（动作 vs 状态）"],
  mode_residual: ["Residual (Action − State)", "残差（动作 − 状态）"],
  mode_velocity: ["Velocity (dq/dt)", "速度（dq/dt）"],
  mode_acceleration: ["Acceleration (d²q/dt²)", "加速度（d²q/dt²）"],
  mode_jerk: ["Jerk (d³q/dt³)", "加加速度 Jerk（d³q/dt³）"],
  residual: ["Residual (Action − State)", "残差（动作 − 状态）"],
  trajectory: ["Trajectory", "轨迹"],
  action: ["Recorded action", "记录动作"],
  "observation.state": ["Observed state", "观测状态"],
  recordedUnit: ["Unit unspecified", "单位未声明"],
  task: ["Task", "任务"],
  manifest_match: ["Checkpoint manifest matches", "Checkpoint 清单一致"],
  manifest_different: ["Checkpoint manifest differs", "Checkpoint 清单不同"],
  manifest_unverified: ["Checkpoint association unverified", "Checkpoint 关联未核实"],
  offline: ["Offline & closed-loop evaluation", "离线与闭环评估报告"],
  closedLoop: ["Closed-loop rollout reports", "闭环执行评估报告"],
  successRate: ["Success rate", "闭环成功率"],
  failureModes: ["Failure modes", "失败类型分布"],
  finalReport: ["Run directory report", "运行目录评估报告"],
  validSteps: ["Valid action steps", "有效动作步"],
  channel: ["Action channel", "动作通道"],
  evalEmpty: ["Add a recording or evaluation report", "添加评估录制或评估报告"],
  evalHint: [
    "Inspect action trajectories, residuals, velocity/acceleration/jerk smoothness, and offline or closed-loop metrics.",
    "结合当前训练查看动作轨迹、残差分布、速度/加速度/Jerk 平滑度以及离线和闭环评估。",
  ],
  imported: ["Imported report", "导入报告"],
  records: ["Records", "记录数"],
  loading: ["Loading…", "读取中…"],
  scale: ["Scale", "倍率"],
  denominator: ["Denominator", "分母"],
  reduction: ["Reduction", "聚合"],
  maxMetrics: ["Select at least one metric.", "请至少选择一项指标。"],
  findMetric: ["Find a metric or field name", "搜索指标或字段名"],
  policyMetrics: ["Policy & Objective Losses", "策略与目标损失"],
  generalizationGroup: ["Generalization & Validation", "泛化与验证"],
  optimizer: ["Optimizer & Stability", "优化器与训练稳定性"],
  systemGroup: ["Throughput & System", "吞吐量与系统耗时"],
  trainingMetrics: ["Other Training Metrics", "其他训练指标"],
  probe: ["Module Update Probes", "模块参数更新探针"],
  paired: ["Matched evaluation", "同口径评估"],
  diagTitle: ["Training diagnostics", "训练诊断工作台"],
  diagScope: ["Scope", "分析范围"],
  scope_tail: ["Recent window", "最近窗口"],
  scope_range: ["Synced chart range", "同步曲线区间"],
  trainMetric: ["Focus metric", "诊断指标"],
  componentChanges: ["Loss composition & share", "多损失构成与占比"],
  lossShareTitle: ["Objective loss breakdown (latest window)", "目标损失构成占比（近期窗口）"],
  sharePct: ["Share", "占比"],
  targetMax: ["Target ≤", "目标上限 ≤"],
  optional: ["Optional", "选填"],
  currentWindow: ["Current slice mean", "当前区间均值"],
  previousWindow: ["Previous slice mean", "前一区间均值"],
  stageEvolution: ["Macro stage evolution (Early / Mid / Late)", "全过程三阶段演进（前／中／后期）"],
  stage_early: ["Early (0–33%)", "前期（0–33%）"],
  stage_mid: ["Middle (33–66%)", "中期（33–66%）"],
  stage_late: ["Late (66–100%)", "后期（66–100%）"],
  currentMean: ["Current", "当前均值"],
  earlierMean: ["Previous", "前窗均值"],
  change: ["Change", "变化量"],
  observations: ["records", "条记录"],
  noEarlierWindow: ["No earlier records", "尚无更早记录"],
  noObservations: ["Waiting for training records", "等待训练记录"],
  noComponents: ["No component loss values in this slice.", "此区间暂无分项损失。"],
  noValidation: ["No validation loss recorded.", "尚无验证损失记录。"],
  lowest: ["Lowest observed", "全程最低"],
  sincePreviousVal: ["vs prev eval", "较上次验证"],
  approxGapNote: ["Approx. Val − nearby 50-step train mean", "近似泛化差距（验证 − 邻近50步训练均值）"],
  meanRose: ["Increased", "均值上升"],
  meanFell: ["Decreased", "均值下降"],
  meanUnchanged: ["Unchanged", "均值持平"],
  variation: ["Distribution & tail spread", "分布与尾部波动"],
  histCount: ["Bar height counts raw records per value bin.", "柱高表示各数值区间内的原始记录数。"],
  median: ["Median (P50)", "中位数（P50）"],
  percentile90: ["90th percentile", "第 90 百分位"],
  percentile95: ["95th percentile", "第 95 百分位"],
  optimizerTiming: ["Gradients, LR & throughput", "梯度、学习率与吞吐耗时"],
  gradP95: ["Grad norm (P95)", "梯度范数（P95）"],
  dataWait: ["Data wait share", "数据等待占比"],
  threshold: ["clip C =", "阈值 C ="],
  sampleUnit: ["samples", "个样本"],
  weights: ["Weights", "权重"],
  unnamed: ["Run", "任务"],
  source: ["Source", "来源"],
  window: ["Window", "统计窗口"],
  mean: ["Mean", "均值"],
  sd: ["Std. dev.", "标准差"],
  min: ["Min", "最小值"],
  max: ["Max", "最大值"],
  slope: ["Slope / step", "每步斜率"],
  peak: ["Peak step", "峰值步数"],
  distribution: ["Distribution", "分布"],
  metric: ["Metric", "指标"],
  recordsUnit: ["records", "条"],
  byPolicy: ["Policy defaults", "恢复策略默认"],
  expand: ["Expand chart", "放大曲线"],
  exportPng: ["PNG", "导出 PNG"],
  exportCsv: ["CSV", "导出 CSV"],
  showRuns: ["Open sidebar", "展开侧栏"],
  hideRuns: ["Close sidebar", "收起侧栏"],
  previous: ["Earlier", "前移"],
  next: ["Later", "后移"],
  zoomIn: ["Zoom in", "放大"],
  zoomOut: ["Zoom out", "缩小"],
  undo: ["Undo", "撤销"],
  yScale: ["Y scale", "纵轴尺度"],
  scale_fit: ["Auto fit", "自适应"],
  scale_robust: ["Ignore outliers (P1–P98)", "去离群点（P1–P98）"],
  scale_log: ["Log scale", "对数轴 (Log)"],
  scale_zero: ["Include zero", "包含零点"],
  rawLine: ["Raw", "单步"],
  emaLine: ["EMA", "平滑"],
  firstHit: ["First at target", "首次达标"],
  noHit: ["Target not reached", "尚未达标"],
  latestVal: ["Val evals", "验证次数"],
  clipping: ["Clipped updates", "裁剪比例"],
  estimatedClip: ["from norm", "由范数推算"],
  samplesPerS: ["Samples/s", "样本/秒"],
  weight_norm: ["Weight norm", "参数范数"],
  update_norm: ["Update norm", "更新范数"],
  relative_update: ["Update / weight", "更新／参数比"],
  gradients: ["Grad tensors", "有梯度张量"],
  module: ["Module", "模块"],
  rates: ["Optimizer groups", "优化器参数组"],
  first: ["First", "起点"],
  last: ["Latest", "最新值"],
  train_eval_loss: ["Fixed train-subset loss", "固定训练子集误差"],
  generalization_gap: ["Val − train subset", "同口径泛化差距"],
  train_eval_samples: ["Train subset size", "训练子集样本数"],
  rawTotals: ["Latest component totals", "最近分项累计量"],
  numerator: ["Numerator", "分子"],
  count: ["Count", "计数"],
  phase: ["Phase", "阶段"],
  console: ["Training console", "训练控制台"],
  train_episodes: ["Training episodes", "训练回合"],
  val_episodes: ["Validation episodes", "验证回合"],
  importCheck: ["Import resource report", "导入资源报告"],
  baseModel: ["Base model", "基座模型"],
  modelWeights: ["Model weights", "模型权重"],
  initialWeights: ["Initialization weights", "初始化权重"],
  tokenizer: ["Tokenizer", "分词器"],
  processor: ["Processor", "预处理器"],
  revision: ["Revision", "版本"],
  path: ["Path", "路径"],
  headerHash: ["Checkpoint header SHA-256", "权重文件头 SHA-256"],
  requestedPath: ["Requested path", "配置路径"],
  resolvedFiles: ["Resolved files", "解析到的文件"],
  reportNote: ["Resource note", "资源说明"],
  allParameters: ["Total parameters", "总参数量"],
  frozenParams: ["Frozen parameters", "冻结参数"],
  python: ["Python", "Python 版本"],
  git_commit: ["Git commit", "代码提交"],
  // Compare tab
  showAllAttempts: ["Show all resume attempts", "展开全部续训尝试"],
  seedBand: ["Multi-seed Mean ± Std", "多种子均值±标准差带"],
  scatterTitle: ["Hyperparameter vs Outcome", "超参数与结果散点图"],
  parallelTitle: ["Parallel Coordinates (HParams)", "超参数平行坐标图"],
  xParam: ["X parameter", "横轴超参"],
  yOutcome: ["Y outcome", "纵轴指标"],
  colRun: ["Run", "实验"],
  colPolicy: ["Policy", "算法"],
  colStatus: ["Status", "状态"],
  colProgress: ["Progress", "进度"],
  colBestVal: ["Best Val", "最佳验证"],
  colTrainMean: ["Train Mean", "训练均值"],
  colGap: ["Gen Gap", "泛化差距"],
  colLr: ["LR", "学习率"],
  colBatch: ["Eff Batch", "有效Batch"],
  colWeightDecay: ["Weight decay", "权重衰减"],
  colWarmup: ["Warmup", "预热步"],
  colClip: ["Clip C", "裁剪阈值"],
  colSeed: ["Seed", "种子"],
  colSpeed: ["Samples/s", "吞吐"],
  colMae: ["Offline MAE", "离线 MAE"],
  seedBandHint: ["Select 2+ runs with comparable steps to aggregate Mean ± 1 SD across seeds.", "在左侧勾选 2 个及以上可比实验即可聚合多种子均值 ± 1 标准差带。"],
  // Events
  ev_warmup_end: ["Warmup end", "预热结束"],
  ev_resume: ["Resume", "续训起点"],
  ev_validation_best: ["Best Val", "最佳验证"],
  ev_checkpoint: ["Ckpt", "保存点"],
  ev_amp_overflow: ["AMP skip", "AMP跳步"],
  ev_nonfinite: ["NaN/Inf", "非有限值"],
  ev_offline_eval: ["Eval MAE", "离线评估"],
  jumpToStep: ["Inspect", "查看曲线"],
  // Policy loss names
  loss_diffusion: ["Diffusion loss", "扩散损失"],
  loss_flow: ["Flow matching loss", "流匹配损失"],
  loss_flow_matching: ["Flow matching loss", "流匹配损失"],
  flow_loss: ["Flow matching loss", "流匹配损失"],
  fm_loss: ["Flow matching loss", "流匹配损失"],
  ce_loss: ["Token cross-entropy", "词元交叉熵"],
  action_loss: ["Action loss", "动作损失"],
  video_loss: ["Video loss", "视频损失"],
  loss_video_weighted: ["Weighted video loss", "加权视频损失"],
  loss_action_weighted: ["Weighted action loss", "加权动作损失"],
  loss_classification: ["Classification loss", "分类损失"],
  loss_classification_weighted: ["Weighted classification loss", "加权分类损失"],
  prediction_loss: ["Latent prediction loss", "潜表示预测损失"],
  sigreg_loss: ["SIGReg regularizer", "SIGReg 正则项"],
  latent_loss: ["Latent loss", "潜表示损失"],
  wm_loss: ["World model loss", "世界模型损失"],
  loss_text: ["Text token loss", "文本词元损失"],
  loss_fast: ["FAST action-token loss", "FAST 动作词元损失"],
  loss_vlm: ["Vision-language token loss", "视觉语言词元损失"],
  loss_mse: ["Flow MSE", "流预测均方误差"],
  loss_freq: ["Frequency loss", "频域损失"],
  loss_score: ["Candidate score loss", "候选评分损失"],
  joints_loss: ["Joint prediction loss", "关节预测损失"],
  seq_wise_loss: ["Router load-balancing loss", "路由负载均衡损失"],
  router_z_loss: ["Router Z-loss", "路由 Z 损失"],
  depth_loss: ["Depth alignment loss", "深度对齐损失"],
  future_depth_loss: ["Future depth loss", "未来深度损失"],
  future_video_loss: ["Future video loss", "未来视频损失"],
  current_video_loss: ["Current video loss", "当前视频损失"],
  losses_after_forward: ["Loss before episode masking", "回合掩码前损失"],
  losses_after_in_ep_bound: ["Loss after episode masking", "回合掩码后损失"],
  losses_after_rm_padding: ["Loss after action trimming", "裁去填充动作维后损失"],
  active_action_dims: ["Valid action coordinates", "有效动作坐标数"],
  loss_vq_reconstruction: ["Quantizer reconstruction L1", "量化器重建 L1"],
  loss_vq_commitment: ["Quantizer commitment loss", "量化承诺损失"],
  loss_code_classification: ["Code classification loss", "编码分类损失"],
  loss_primary_code: ["Primary code focal loss", "一级编码焦点损失"],
  loss_secondary_code: ["Secondary code focal loss", "二级编码焦点损失"],
  loss_offset: ["Offset L1 loss", "偏移动作 L1 损失"],
  equal_primary_code_rate: ["Primary code match rate", "一级编码匹配率"],
  equal_secondary_code_rate: ["Secondary code match rate", "二级编码匹配率"],
  mean_batch_distinct_codes: ["Distinct codes / microbatch", "每微批次不同编码数"],
  mean_batch_distinct_combinations: ["Distinct code pairs / microbatch", "每微批次不同编码组合数"],
  mean_batch_max_action_error: ["Peak action error mean", "微批次最大动作误差均值"],
  vq_action_error: ["Decoded action MAE", "解码动作 MAE"],
  offset_action_error: ["Corrected action MAE", "偏移修正动作 MAE"],
  action_mse_error: ["Action MSE", "动作均方误差"],
  "policy/loss": ["Policy-reported loss", "策略返回损失"],
  l1_loss: ["Action L1 loss", "动作 L1 损失"],
  kld_loss: ["KL divergence", "KL 散度"],
  kld_loss_weighted: ["Weighted KL loss", "加权 KL 损失"],
  recorded: ["Trainer report", "训练器报告"],
  supplied: ["Imported report", "导入报告"],
  badCheck: [
    "Choose a resource or launch preflight JSON for this run (up to 4 MiB).",
    "请选择当前任务的资源或启动预检 JSON（最大 4 MiB）。",
  ],
  effective_batch_size: ["Effective batch", "有效批大小"],
  training_samples: ["Training samples", "训练样本"],
  validation_samples: ["Validation samples", "验证样本"],
  excluded_samples: ["Excluded samples", "已排除样本"],
  reportMetric: ["Error measure", "误差指标"],
  bias_by_action: ["Signed bias", "有符号偏差"],
  rmse_by_action: ["RMSE", "均方根误差"],
  mae_by_action: ["MAE", "平均绝对误差"],
  p50_by_action: ["P50 abs error", "P50 绝对误差"],
  p90_by_action: ["P90 abs error", "P90 绝对误差"],
  p95_by_action: ["P95 abs error", "P95 绝对误差"],
  horizon: ["Prediction offset", "预测偏移"],
  all: ["All", "全部"],
  noConfig: ["No saved parameters match.", "没有匹配的已保存参数。"],
  bias: ["Signed bias", "有符号偏差"],
  mae: ["MAE", "平均绝对误差"],
  rmse: ["RMSE", "均方根误差"],
  rms: ["RMS", "均方根"],
  maxAbs: ["Peak |val|", "绝对峰值"],
  p50: ["P50", "P50"],
  p90: ["P90", "P90"],
  p95: ["P95", "P95"],
};

// Concise, domain-expert explanations (Definition · What to watch / Pair with)
const help = {
  step: [
    "Successful global optimizer updates. Resumed runs keep checkpoint step numbering.\nCompare efficiency against Samples and Elapsed time too, not steps alone.",
    "已执行的全局优化器更新步数；续训沿用 checkpoint 步数。\n比较训练效率时，应同时对照“已处理样本”和“经过时间”。",
  ],
  samples: [
    "Cumulative samples processed across all ranks (including distributed padding).\nUse as X-axis when comparing runs with different effective batch sizes.",
    "所有进程累计处理的样本总数（含分布式补齐）。\n对比不同有效批大小的实验时，应切到此横轴衡量样本效率。",
  ],
  elapsed_s: [
    "Monotonic wall-clock seconds since training loop started (includes eval & save pauses).\nBest X-axis for judging real time-to-quality.",
    "训练循环启动后的单调真实耗时（含验证和保存暂停）。\n衡量“达到相同质量所需时间”的首选横轴。",
  ],
  loss: [
    "Total weighted training objective optimized by the policy.\nWatch: pair with Validation loss and sub-loss shares; one dominant term (e.g. video) can mask action improvement.",
    "策略优化的加权总目标损失。\n观察要点：需结合验证损失和各子损失占比一起看；主导项（如视频项）可能掩盖动作项的真实改善。",
  ],
  trainLoss: [
    "Mean objective loss over the latest 100 optimizer steps (with latest raw step shown beside it).\nAvoid comparing a single noisy step directly against a full validation mean.",
    "最近 100 次更新的训练损失均值（右侧附最新单步值）。\n避免用单步随机噪点直接与全验证集均值做过拟合判断。",
  ],
  valLoss: [
    "Held-out validation objective using the policy's evaluation denominators.\nWatch: lowest observed step vs latest; repeated rises while train drops signal overfitting.",
    "留出验证集上的同口径损失。\n观察要点：对照全程最低值与最新值；若训练持续降而验证连续升，说明出现过拟合。",
  ],
  grad_norm: [
    "Global L2 gradient norm before clipping, after AMP unscaling.\nNot 'smaller is always better' — watch spikes relative to the clip threshold C and pair with LR & clip rate.",
    "混合精度取消缩放后、裁剪前的全局梯度 L2 范数。\n并非越大或越小越好：重点看相对裁剪阈值 C 的尖峰频率，并配合学习率与裁剪率判断稳定性。",
  ],
  lr: [
    "Actual learning rate applied to parameter group 0 before scheduler step.\nVerify warmup slope, decay schedule, and pair sudden loss jumps with LR transitions.",
    "本次更新第 0 参数组实际使用的学习率（调度器步进前）。\n用于核对预热、衰减曲线，并与损失突变做事件关联。",
  ],
  grad_clipped: [
    "Per-step indicator (1 if pre-clip grad norm exceeded threshold C, else 0).\nPrefer the rolling clip rate curve for judging persistent gradient saturation.",
    "单步裁剪标志（裁剪前范数超过阈值 C 记 1，否则 0）。\n建议看“滑动裁剪率”曲线判断是否存在持续饱和。",
  ],
  skipped_updates: [
    "Cumulative optimizer updates skipped due to AMP gradient overflow.\nFrequent skips waste compute and stall effective learning.",
    "因混合精度梯度溢出而跳过的累计更新次数。\n频繁跳步会拖慢有效学习并提示数值不稳定。",
  ],
  nonfinite_count: [
    "Explicit count of non-finite (NaN/Inf) gradient or metric events per step.\nDo not rely solely on gaps in smoothed curves to detect numeric failure.",
    "显式记录的非有限值（NaN/Inf）异常次数。\n不要仅凭曲线断开来猜测数值崩溃。",
  ],
  "derived/clip_fraction": [
    "Rolling 50-step fraction of updates hitting grad_clip_norm.\nWatch: >25% sustained clipping suggests high LR, noisy batches, or tight clip threshold.",
    "近 50 步触发梯度裁剪的比例。\n观察要点：若长期高于 25%，需检查学习率偏大、异常样本或裁剪阈值偏紧。",
  ],
  "derived/generalization_gap": [
    "Validation loss minus training loss (exact paired train-subset eval if enabled, else nearby 50-step train mean).\nGap widening alone is fine if val still falls; gap widening + val rising = overfitting.",
    "验证损失减训练损失（开启同口径采样时为固定训练子集差值，否则取验证步邻近 50 步训练均值）。\n差距扩大且验证仍降属正常拟合加深；差距扩大且验证上升则是过拟合信号。",
  ],
  "derived/samples_per_s": [
    "Effective samples processed per second (effective batch / total step time).\nFaster steps do not guarantee faster time-to-quality — pair with Val vs Elapsed time.",
    "每秒处理的有效样本数（有效批大小 ÷ 单步总耗时）。\n单步更快不等于更快达标，需结合“验证损失-真实耗时”判断。",
  ],
  "derived/step_s": [
    "Total wall time per optimizer step (dataloading_s + update_s).\nWatch for upward drift indicating memory fragmentation or I/O throttling.",
    "每次优化器更新的总耗时（数据加载 + 前反向与更新）。\n用于发现训练中途掉速或 I/O 抖动。",
  ],
  "derived/data_wait_ratio": [
    "Fraction of step time spent waiting for microbatches: dataloading_s / (dataloading_s + update_s).\nSustained >35% indicates an input pipeline bottleneck (workers / video decode / storage).",
    "单步中等待数据加载的耗时占比。\n若长期超过 35%，说明存在数据读取或视频解码瓶颈，GPU 在空等。",
  ],
  dataloading_s: [
    "Seconds spent fetching microbatches on the logging rank.\nPair with update_s to distinguish compute-bound vs I/O-bound steps.",
    "日志进程读取微批次的等待秒数。\n与更新耗时对照，区分计算瓶颈还是数据瓶颈。",
  ],
  update_s: [
    "Forward + backward + optimizer step time after batches are fetched (includes AMP retries).",
    "取得数据后的前向、反向与参数更新耗时（含溢出重试）。",
  ],
  gpu_mem_gb: [
    "Peak CUDA tensor memory allocated on the logging rank during this step (GiB).",
    "本步日志进程所在 GPU 的峰值张量显存分配量（GiB）。",
  ],
  loss_flow: [
    "Flow matching objective between predicted velocity field and sampled noise-to-action target.\nPair with validation flow loss and offline action MAE.",
    "采样噪声水平下，预测流场与目标动作流之间的误差。\n需结合验证流损失与离线动作 MAE 共同判断。",
  ],
  loss_diffusion: [
    "Denoising prediction error at sampled diffusion timesteps.",
    "采样去噪时间步上的扩散目标预测误差。",
  ],
  loss_l1: [
    "Mean absolute action reconstruction error in the policy's normalized training representation.",
    "策略归一化表示下的动作 L1 绝对重建误差。",
  ],
  loss_kl: [
    "Unweighted KL divergence between CVAE latent posterior and prior.\nWatch for KL collapse (near 0) or runaway growth.",
    "原始尺度的潜变量 KL 散度（乘权重前）。\n关注是否坍缩趋近 0 或异常发散。",
  ],
  loss_kl_weighted: [
    "KL divergence multiplied by its configured objective weight (actual contribution to total loss).",
    "乘以配置权重后的 KL 项，代表其对总目标损失的实际数值贡献。",
  ],
  loss_action: [
    "Action prediction loss emitted by the policy.\nNote: in multi-modal models (e.g. OpenWAM), action loss may be much smaller in magnitude than video loss — always inspect separately.",
    "策略输出的动作预测损失项。\n注意：多模态模型（如 OpenWAM）中动作项量级常远小于视频项，务必单独观察其收敛趋势。",
  ],
  loss_video: [
    "Video / world-model prediction loss emitted by the policy.\nCan dominate total loss numerically; check action loss independently to judge control learning.",
    "策略输出的视频／世界模型预测损失项。\n数值上常主导总损失，判断控制能力需单独查看动作损失和评估误差。",
  ],
  loss_fast: [
    "Cross-entropy loss over FAST discrete action tokens.",
    "FAST 离散动作词元的交叉熵损失。",
  ],
  loss_vlm: [
    "Vision-language token modeling loss.",
    "视觉-语言词元建模损失。",
  ],
  wm_loss: [
    "World-model latent prediction loss.",
    "世界模型潜空间预测损失。",
  ],
  sigreg_loss: [
    "SIGReg representation regularization term preventing latent collapse.",
    "防止潜表示坍缩的 SIGReg 正则化损失。",
  ],
  smoothing: [
    "Debiased Exponential Moving Average applied to raw records across all metrics.\n0 = off; higher = smoother trend. Quantitative diagnostics always use raw unsampled records.",
    "基于原始记录计算的去偏指数移动平均（支持全部标量曲线）。\n0 为关闭，越大越平滑；定量诊断始终基于原始记录，不对平滑图求导。",
  ],
  yScale: [
    "Auto fit: full range · Ignore outliers: P1–P99 raw bounds so spikes don't crush convergence · Log: multi-decade ranges (LR, action loss) · Include zero: true magnitude baseline.",
    "自适应：全范围 · 去离群点：按原始 P1–P99 定界避免尖峰压扁趋势 · 对数轴：适合跨数量级指标 · 包含零点：看真实绝对量级。",
  ],
  relative_update: [
    "Actual optimizer step magnitude relative to weights: R_l = ||Δθ_l||₂ / ||θ_l||₂.\nCS231n rough order ~1e-3 (not a hard rule); watch for stalled updates (<1e-6) or zero-norm denominator distortion.",
    "模块实际更新幅度与权重范数之比 R_l = ||Δθ_l||₂ / ||θ_l||₂。\nCS231n 粗参考量级约 1e-3（非硬标准）；关注长期过小（<1e-6）或参数范数趋零导致的放大误导。",
  ],
  clipping: [
    "Fraction of optimizer updates in this slice whose pre-clip norm exceeded grad_clip_norm.",
    "当前区间内裁剪前梯度范数超过阈值的更新比例。",
  ],
  median: [
    "50th percentile of raw values in the slice — robust against occasional batch spikes.",
    "当前区间原始值的中位数（P50），不受偶发极端 batch 尖峰干扰。",
  ],
  percentile95: [
    "95th percentile of raw values — reveals heavy-tail spikes hidden by the mean.",
    "第 95 百分位，用于暴露被均值掩盖的尾部尖峰和困难样本。",
  ],
  slope: [
    "Linear least-squares trend per optimizer step over raw records in the slice.\nNote: curve slope dL/dt is NOT parameter gradient ∇L or Hessian curvature.",
    "当前区间原始记录按步数线性拟合的斜率。\n注意：曲线斜率 dL/dt 既不是参数梯度 ∇L，二次差分也不是 Hessian 曲率。",
  ],
  generalization_gap: [
    "Same-mode evaluation gap G(t) = L_val(t) − L_train_eval(t) using identical weights & eval mode.",
    "同权重、同评估模式下的严谨泛化差距 G(t) = L_val(t) − L_train_eval(t)。",
  ],
  train_eval_loss: [
    "Evaluation-mode loss on a fixed training subset (without dropout/train-only noise) for apples-to-apples comparison with validation loss.",
    "固定训练子集在 eval 模式下的损失（消除 dropout 等口径差异），用于与验证损失严谨对比。",
  ],
  dataWait: [
    "Share of step wall time spent waiting on DataLoader vs GPU compute.",
    "单步总耗时中 DataLoader 等待时间所占比例。",
  ],
  "optimizer.lr": [
    "Configured peak/base learning rate before scheduler scaling.",
    "应用学习率调度前配置的基础／峰值学习率。",
  ],
  "optimizer.grad_clip_norm": [
    "Global L2 gradient clipping threshold C; <= 0 disables clipping. Drawn as a dashed reference line on Gradient norm.",
    "全局 L2 梯度裁剪阈值 C（非正值关闭）。会在梯度范数曲线上绘制水平参考虚线。",
  ],
  "optimizer.weight_decay": [
    "Decoupled weight decay regularization coefficient.",
    "优化器使用的权重衰减正则化系数。",
  ],
  "scheduler.warmup_steps": [
    "Optimizer steps allocated to LR warmup; marked as a vertical event line on charts.",
    "学习率预热步数，会在曲线上标注为竖向事件参考线。",
  ],
  batch_size: [
    "Per-device microbatch size. Effective batch = batch_size × grad_accum × world_size.",
    "单卡单微批次样本数；有效批大小 = batch_size × 梯度累积 × 进程数。",
  ],
  seed: [
    "Random seed for RNG initialization. Same seed does not guarantee bit-exact cross-hardware reproducibility; always record commit & versions too.",
    "随机种子。相同 seed 不保证跨硬件/版本完全复现，需同时留存代码版本和依赖环境。",
  ],
  mixed_precision: [
    "Autocast precision mode (none / bfloat16 / float16).",
    "混合精度训练模式（none / bfloat16 / float16）。",
  ],
  frozenParams: [
    "Parameters with requires_grad=False at startup. Expand to inspect per-module trainable vs frozen breakdown.",
    "启动时 requires_grad=False 的冻结参数总量及占比；展开可逐模块核对冻结范围。",
  ],
};

export const $ = (id) => document.getElementById(id);

const requestedLang =
  new URL(location.href).searchParams.get("lang") || localStorage.getItem("am:lang") || "en";
export let lang = requestedLang.startsWith("zh") ? "zh" : "en";
export function toggleLang() {
  lang = lang === "zh" ? "en" : "zh";
  localStorage.setItem("am:lang", lang);
}

// Theme management (light / dark)
const storedTheme = localStorage.getItem("am:theme");
export let theme =
  storedTheme === "dark" || storedTheme === "light"
    ? storedTheme
    : matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";

const themeListeners = new Set();
export function onThemeChange(fn) {
  themeListeners.add(fn);
  return () => themeListeners.delete(fn);
}
export function applyTheme() {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = theme === "dark" ? "#151a23" : "#ffffff";
  const icon = $("theme-icon");
  if (icon) icon.setAttribute("href", theme === "dark" ? "#icon-sun" : "#icon-moon");
  for (const fn of themeListeners) fn(theme);
}
export function toggleTheme() {
  theme = theme === "dark" ? "light" : "dark";
  localStorage.setItem("am:theme", theme);
  applyTheme();
}

export function t(key) {
  const zh = lang === "zh";
  if (Object.hasOwn(words, key)) return words[key][zh ? 1 : 0];
  let m = /^classification_confusion_dim_(\d+)_(\d+)_(\d+)$/.exec(key);
  if (m)
    return zh
      ? `维度 ${m[1]} · 实际 ${m[2]} → 预测 ${m[3]}`
      : `Dim ${m[1]} · actual ${m[2]} → predicted ${m[3]}`;
  m = /^classification_dim_(\d+)_class_(\d+)_(recall|support)$/.exec(key);
  if (m)
    return zh
      ? `维度 ${m[1]} · 类别 ${m[2]} ${m[3] === "recall" ? "召回率" : "样本数"}`
      : `Dim ${m[1]} · class ${m[2]} ${m[3]}`;
  m = /^classification_dim_(\d+)_macro_recall$/.exec(key);
  if (m) return zh ? `维度 ${m[1]} · 宏平均召回率` : `Dim ${m[1]} · macro recall`;
  m = /^loss_classification_dim_(\d+)$/.exec(key);
  if (m) return t("loss_classification") + " · " + (zh ? "维度 " : "dim ") + m[1];
  if (key.startsWith("loss_l1_")) return t("loss_l1") + " · " + key.slice(8);
  if (key.startsWith("lr/")) return t("lr") + " · " + (zh ? "参数组 " : "group ") + key.slice(3);
  if (key.startsWith("module/")) {
    const [, mod, qty] = key.split("/");
    return `${mod} · ${t(qty)}`;
  }
  return key;
}

export function fmt(v, compact = false) {
  if (!Number.isFinite(v)) return "—";
  if (!compact && Number.isSafeInteger(v)) return v.toLocaleString(lang);
  const a = Math.abs(v);
  if (a && (a < 0.001 || a >= 1e7)) return v.toExponential(2);
  return v.toLocaleString(lang, { maximumSignificantDigits: compact ? 4 : 6 });
}

export function fmtPct(v, compact = true) {
  if (!Number.isFinite(v)) return "—";
  return fmt(v, compact) + "%";
}

export function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return fmt(seconds, true) + "s";
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function explain(key, spec, run) {
  const i = lang === "zh" ? 1 : 0;
  const alias = {
    flow_loss: "loss_flow",
    fm_loss: "loss_flow",
    loss_flow_matching: "loss_flow",
    video_loss: "loss_video",
    action_loss: "loss_action",
    kld_loss: "loss_kl",
    l1_loss: "loss_l1",
    step_s: "derived/step_s",
    samples_per_s: "derived/samples_per_s",
  };
  let text = help[key]?.[i] || help[alias[key]]?.[i] || "";
  if (run?.policy === "openwam" && ["loss_action", "loss_video"].includes(key))
    text += ["\nIncludes the configured objective weight.", "\n已包含该项的配置权重。"][i];
  if (key.endsWith("_macro_recall"))
    text = [
      "Mean recall across classes that have observed targets.",
      "在出现过真实目标的类别上，对各类别召回率取平均。",
    ][i];
  else if (key.endsWith("_recall"))
    text = [
      "Fraction of observed targets of this class that were correctly predicted.",
      "该类别的真实目标中被正确预测的比例。",
    ][i];
  else if (key.endsWith("_support"))
    text = ["Number of observed targets of this class.", "该类别出现的有效目标数。"][i];
  else if (key.startsWith("classification_confusion_"))
    text = [
      "Count for the displayed actual and predicted class pair.",
      "真实类别与预测类别构成的这一组合的出现次数。",
    ][i];

  if (key === "loss" && run?.policy === "openwam") {
    text = [
      "OpenWAM objective = λ_video · L_video + λ_action · L_action.\nVideo loss is often much larger numerically; check Action loss separately.",
      "OpenWAM 目标损失 = λ_video·视频损失 + λ_action·动作损失。\n视频项数值常占主导，判断控制精度务必单独看动作损失。",
    ][i];
    const model = run.config?.model;
    if (Number.isFinite(model?.lambda_video) && Number.isFinite(model?.lambda_action)) {
      text +=
        ["\nWeights (video / action): ", "\n权重（视频／动作）："][i] +
        fmt(model.lambda_video) +
        " / " +
        fmt(model.lambda_action);
    }
  } else if (key === "loss" && run?.policy === "act") {
    text = [
      "ACT objective = Action L1 + β · KL (during CVAE training). Validation reports Action L1.",
      "ACT 目标损失 = 动作 L1 + β·KL（CVAE 训练期）；验证阶段报告动作 L1。",
    ][i];
  } else if (key.startsWith("module/")) {
    const quantity = key.split("/")[2];
    text = help[quantity]?.[i] ||
      {
        update_norm: [
          "L2 norm of post-step minus pre-step trainable parameters.",
          "抽样更新前后可训练参数差的 L2 范数。",
        ],
        weight_norm: [
          "Pre-update L2 norm of sampled trainable parameters. Pair with update_norm so near-zero norms don't mislead.",
          "更新前参数 L2 范数；需与更新范数同看，避免分母趋零造成比值失真。",
        ],
      }[quantity]?.[i] || "";
  } else if (key.startsWith("lr/")) {
    text = [
      `Actual learning rate applied to optimizer parameter group ${key.slice(3)}.`,
      `本次更新中优化器参数组 ${key.slice(3)} 的实际学习率。`,
    ][i];
  }
  if (spec) {
    const parts = [
      spec.source !== key && spec.source,
      spec.reduction &&
        t("reduction") +
          ": " +
          (spec.reduction === "mean" ? t("mean") : spec.reduction === "sum" ? ["sum", "求和"][i] : spec.reduction),
      spec.denominator && t("denominator") + ": " + spec.denominator,
      spec.scale != null && spec.scale !== 1 && t("scale") + " ×" + spec.scale,
    ];
    return [text, parts.filter(Boolean).join(" · ")].filter(Boolean).join("\n");
  }
  return text;
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
export function describe(node, description) {
  if (description) {
    node.dataset.help = description;
    if (!["BUTTON", "INPUT", "SELECT", "A"].includes(node.tagName)) node.tabIndex = 0;
    node.setAttribute("aria-description", description);
  } else {
    delete node.dataset.help;
    node.removeAttribute("aria-description");
  }
}
export function iconButton(id, label, action) {
  const button = btn("", action, "icon-button");
  button.title = label;
  button.setAttribute("aria-label", label);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS(svg.namespaceURI, "use");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  use.setAttribute("href", "#icon-" + id);
  svg.append(use);
  button.append(svg);
  return button;
}
export function table(parent, headings, rows) {
  if (!rows.length) return;
  const wrap = el("div", "table-wrap"),
    grid = el("table"),
    header = grid.createTHead().insertRow();
  for (const key of headings) header.append(el("th", null, t(key)));
  const body = grid.createTBody();
  for (const values of rows) {
    const tr = body.insertRow();
    values.forEach((v) => {
      const td = tr.insertCell();
      if (v instanceof Node) td.append(v);
      else td.textContent = v ?? "—";
    });
  }
  wrap.append(grid);
  parent.append(wrap);
}
export function disclosure(parent, label) {
  const node = el("details", "compact");
  node.append(el("summary", null, label));
  parent.append(node);
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

const etagCache = new Map();
export async function api(path, params = new URLSearchParams(), signal) {
  const qs = new URLSearchParams(params).toString();
  const url = "./api/v1/" + path + (qs ? "?" + qs : "");
  const headers = {};
  const cached = etagCache.get(url);
  if (cached) headers["If-None-Match"] = cached.etag;
  const response = await fetch(url, { signal, cache: "no-cache", headers });
  if (response.status === 304 && cached) return structuredClone(cached.value);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  const etag = response.headers.get("ETag");
  if (etag) {
    if (etagCache.size > 160) etagCache.delete(etagCache.keys().next().value);
    etagCache.set(url, { etag, value });
  }
  return structuredClone(value);
}

export function modal(title) {
  $("dialog").classList.remove("inspector");
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
  (target.closest("dialog") || document.body).append(box);
  const parts = target.dataset.help.split("\n");
  box.replaceChildren();
  parts.forEach((line, idx) => {
    const div = el("div", idx === 0 ? "help-title" : idx === parts.length - 1 && parts.length > 2 ? "help-meta" : "");
    div.textContent = line;
    box.append(div);
  });
  box.hidden = false;
  box.style.left = Math.max(8, Math.min(rect.left, innerWidth - box.offsetWidth - 8)) + "px";
  box.style.top =
    (rect.bottom + box.offsetHeight + 8 < innerHeight
      ? rect.bottom + 6
      : Math.max(6, rect.top - box.offsetHeight - 6)) + "px";
}
function hideHelp() {
  $("help").hidden = true;
}
for (const event of ["pointerover", "focusin"]) document.addEventListener(event, showHelp);
for (const event of ["pointerout", "focusout", "scroll"])
  document.addEventListener(event, hideHelp, true);
$("close-dialog").onclick = () => $("dialog").close();
$("dialog").onclick = (e) => {
  if (e.target === $("dialog") && !e.target.closest("#dialog-body")) {
    const r = $("dialog").getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
      $("dialog").close();
  }
};
applyTheme();
