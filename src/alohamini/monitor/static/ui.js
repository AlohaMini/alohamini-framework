// Shared display strings, metric definitions and DOM helpers.
const words = {
  setting: ["Value", "配置值"],
  toggleSeries: ["Show or hide series", "显示或隐藏此曲线"],
  compareHint: ["Select a run to inspect. Tick others to compare.", "点击查看任务，勾选叠加对比。"],
  compareLimit: ["Compare up to six attempts at a time.", "可同时对比六次训练。"],
  lastEvent: ["Last record", "最近记录"],
  fileUpdated: ["Files updated", "文件更新"],
  runs: ["Training runs", "训练任务"],
  searchRuns: ["Find a run", "搜索训练任务"],
  monitor: ["Training", "训练监控"],
  addEval: ["Add evaluation data", "导入评估数据"],
  noRuns: ["Waiting for a training run", "等待训练任务"],
  noRunsHint: [
    "New runs appear here automatically. You can also open an existing output directory.",
    "新训练会自动出现在这里，也可以打开已有的训练输出目录。",
  ],
  views: ["Run views", "运行视图"],
  metrics: ["Metrics", "指标"],
  evaluation: ["Evaluation", "评估"],
  evaluationData: ["Evaluation data", "评估数据"],
  config: ["Configuration", "配置"],
  logs: ["Logs", "日志"],
  axis: ["X axis", "横轴"],
  range: ["Range", "区间"],
  editRange: ["Enter exact range", "输入精确区间"],
  chooseMetrics: ["Select metrics", "选择指标"],
  display: ["Display", "显示设置"],
  chartWidth: ["Minimum chart width", "最小图宽"],
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
  stepAt: ["Step", "步数"],
  unavailable: ["No record", "暂无记录"],
  noData: ["No points in this range", "当前区间没有记录点"],
  sampled: ["Extrema retained", "保留极值"],
  raw: ["Raw points", "原始点"],
  points: ["points", "点"],
  selectHint: [
    "Drag to zoom. Use ← and → to inspect points.",
    "拖动选择区间，使用左右方向键查看记录点。",
  ],
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
  scale: ["Scale", "倍率"],
  denominator: ["Denominator", "分母"],
  reduction: ["Reduction", "聚合"],
  maxMetrics: ["Select at least one metric.", "请至少选择一项指标。"],
  findMetric: ["Find a metric or field name", "搜索指标或字段名"],
  previousPage: ["Previous charts", "上一组曲线"],
  nextPage: ["Next charts", "下一组曲线"],
  policyMetrics: ["Policy metrics", "策略指标"],
  trainingMetrics: ["Training and system metrics", "训练与系统指标"],
  windowComparison: ["Training window comparison", "训练窗口对比"],
  trainMetric: ["Training metric", "训练指标"],
  componentChanges: ["Loss components", "损失分项变化"],
  targetMax: ["Target ≤", "目标上限 ≤"],
  optional: ["Optional", "选填"],
  currentWindow: ["Current window mean", "本窗口均值"],
  previousWindow: ["Previous window mean", "上一窗口均值"],
  currentMean: ["Current", "本窗均值"],
  earlierMean: ["Previous", "上窗均值"],
  change: ["Change", "变化量"],
  observations: ["valid records", "条有效记录"],
  noEarlierWindow: ["No earlier records", "尚无更早记录"],
  noObservations: ["Waiting for training records", "等待训练记录"],
  noComponents: ["No component values in this window.", "此窗口暂无分项记录。"],
  noValidation: ["No validation loss recorded.", "尚无验证损失记录。"],
  lowest: ["Lowest observed", "全程最低"],
  sincePreviousVal: ["Change since previous evaluation", "较上次验证"],
  meanRose: ["Mean increased", "均值上升"],
  meanFell: ["Mean decreased", "均值下降"],
  meanUnchanged: ["Mean unchanged", "均值持平"],
  variation: ["Variation and distribution", "波动与分布"],
  histCount: ["Bar height counts observations in each value range.", "柱高表示该数值区间内的记录数。"],
  median: ["Median (P50)", "中位数（P50）"],
  percentile95: ["95th percentile", "第 95 百分位"],
  optimizerTiming: ["Gradients, learning rates and timing", "梯度、学习率与训练耗时"],
  gradP95: ["Gradient norm (P95)", "梯度范数（P95）"],
  threshold: ["threshold", "阈值"],
  sampleUnit: ["samples", "个样本"],
  weights: ["Weights", "权重"],
  unnamed: ["Run", "任务"],
  source: ["Source", "来源"],
  diagnostics: ["Diagnostics", "诊断"],
  trend: ["Loss trend", "损失趋势"],
  off: ["Off", "关闭"],
  window: ["Window", "统计窗口"],
  mean: ["Mean", "均值"],
  prevMean: ["Previous mean", "前一窗口均值"],
  sd: ["Std. dev.", "标准差"],
  min: ["Min", "最小值"],
  max: ["Max", "最大值"],
  slope: ["Slope / step", "每步斜率"],
  peak: ["Peak step", "峰值步数"],
  distribution: ["Window distribution", "窗口分布"],
  metric: ["Metric", "指标"],
  recordsUnit: ["records", "条记录"],
  byPolicy: ["Policy defaults", "恢复策略默认"],
  expand: ["Expand chart", "放大曲线"],
  showRuns: ["Show run list", "展开任务列表"],
  hideRuns: ["Hide run list", "收起任务列表"],
  previous: ["Earlier", "前移"],
  next: ["Later", "后移"],
  zoomIn: ["Zoom in", "放大"],
  zoomOut: ["Zoom out", "缩小"],
  undo: ["Undo zoom", "撤销缩放"],
  yScale: ["Y axis", "纵轴"],
  autoScale: ["Fit data", "适应数据"],
  zeroScale: ["Include zero", "包含零点"],
  interval: ["View", "当前范围"],
  rawLine: ["Raw", "单步"],
  target: ["Validation loss ≤", "验证损失 ≤"],
  best: ["Best observed", "已观察到的最低值"],
  firstHit: ["First at target", "首次达到目标"],
  noHit: ["Target not reached", "尚未达到目标"],
  latestVal: ["Validation checkpoints", "验证记录数"],
  efficiency: ["Training performance", "训练性能"],
  clipping: ["Clipped updates", "裁剪更新"],
  estimatedClip: ["inferred from norm", "按梯度范数推算"],
  samplesPerS: ["Samples/s", "样本/秒"],
  components: ["Policy components", "策略分项"],
  weight_norm: ["Weight norm", "参数范数"],
  update_norm: ["Update norm", "更新范数"],
  relative_update: ["Update / weight", "更新／参数比"],
  gradients: ["Connected tensors", "有梯度张量"],
  module: ["Module", "模块"],
  probe: ["Sampled parameter updates", "抽样参数更新"],
  rates: ["Optimizer groups", "优化器参数组"],
  first: ["First", "起点"],
  last: ["Latest", "最近值"],
  train_eval_loss: ["Fixed train-subset loss", "固定训练子集误差"],
  generalization_gap: ["Validation − train subset", "验证减训练子集"],
  train_eval_samples: ["Train subset size", "训练子集样本数"],
  paired: ["Matched evaluation", "同口径评估"],
  rawTotals: ["Latest component totals", "最近分项累计量"],
  numerator: ["Numerator", "分子"],
  count: ["Count", "计数"],
  phase: ["Phase", "阶段"],
  yes: ["Yes", "是"],
  console: ["Training console", "训练控制台"],
  stage: ["Stage", "阶段"],
  train_episodes: ["Training episodes", "训练回合"],
  val_episodes: ["Validation episodes", "验证回合"],
  resources: ["Resource preflight", "资源预检"],
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
  losses_after_rm_padding: ["Loss after action-dimension trimming", "裁去填充动作维后的损失"],
  active_action_dims: ["Valid action coordinates per sample", "每样本有效动作坐标数"],
  loss_vq_reconstruction: ["Quantizer reconstruction L1", "量化器重建 L1"],
  loss_vq_commitment: ["Quantizer commitment loss", "量化承诺损失"],
  loss_code_classification: ["Code classification loss", "编码分类损失"],
  loss_primary_code: ["Primary code focal loss", "一级编码焦点损失"],
  loss_secondary_code: ["Secondary code focal loss", "二级编码焦点损失"],
  loss_offset: ["Offset L1 loss", "偏移动作 L1 损失"],
  equal_primary_code_rate: ["Primary code match rate", "一级编码匹配率"],
  equal_secondary_code_rate: ["Secondary code match rate", "二级编码匹配率"],
  mean_batch_distinct_codes: ["Distinct codes per microbatch", "每微批次不同编码数"],
  mean_batch_distinct_combinations: ["Distinct code pairs per microbatch", "每微批次不同编码组合数"],
  mean_batch_max_action_error: ["Mean of microbatch peak action errors", "微批次最大动作误差的均值"],
  vq_action_error: ["Decoded action MAE", "解码动作平均绝对误差"],
  offset_action_error: ["Corrected action MAE", "偏移修正后动作平均绝对误差"],
  action_mse_error: ["Action MSE", "动作均方误差"],
  "policy/loss": ["Policy-reported loss", "策略返回的损失"],
  l1_loss: ["Action L1 loss", "动作 L1 损失"],
  kld_loss: ["KL divergence", "KL 散度"],
  kld_loss_weighted: ["Weighted KL loss", "加权 KL 损失"],
  recorded: ["Trainer report", "训练器报告"],
  supplied: ["Imported report", "导入的报告"],
  badCheck: [
    "Choose a resource or launch preflight JSON for this run (up to 4 MiB).",
    "请选择当前任务的资源或启动预检 JSON（最大 4 MiB）。",
  ],
  kind: ["Kind", "类型"],
  status: ["Status", "状态"],
  paths: ["Paths and details", "路径与详情"],
  runtime: ["Training setup", "训练设置"],
  flags: ["Parameter trainability at startup", "启动时参数冻结状态"],
  frozen: ["Frozen", "已冻结"],
  trainable: ["Trainable", "可训练"],
  training_samples: ["Training samples", "训练样本"],
  validation_samples: ["Validation samples", "验证样本"],
  effective_batch_size: ["Effective batch", "有效批大小"],
  excluded_samples: ["Excluded samples", "已排除样本"],
  optimizer: ["Optimizer", "优化器"],
  precision: ["Precision", "精度"],
  reportMetric: ["Error measure", "误差指标"],
  bias_by_action: ["Signed bias", "有符号偏差"],
  rmse_by_action: ["RMSE", "均方根误差"],
  mae_by_action: ["MAE", "平均绝对误差"],
  horizon: ["Prediction offset", "预测偏移"],
  all: ["All", "全部"],
  noConfig: ["No saved parameters match.", "没有匹配的已保存参数。"],
};
const help = {
  loss_flow: [
    "Error between the predicted flow and its training target at sampled noise levels.",
    "在采样噪声水平下，预测流与训练目标之间的误差。",
  ],
  loss_diffusion: [
    "Error against the configured diffusion target at sampled denoising steps.",
    "在采样的去噪步上，相对于配置扩散目标的误差。",
  ],
  losses_after_forward: [
    "Mean forward error before masking timesteps at episode boundaries.",
    "按回合边界屏蔽时间步之前的平均预测误差。",
  ],
  losses_after_in_ep_bound: [
    "Mean after zeroing padded timesteps, including the zeros in the denominator.",
    "将回合外时间步置零后的均值，分母仍包含这些位置。",
  ],
  losses_after_rm_padding: [
    "Mean after trimming action dimensions; masked timesteps still enter the denominator.",
    "裁去填充动作维后的均值，分母仍包含已屏蔽的时间步。",
  ],
  loss_vq_reconstruction: [
    "Mean absolute error between an action chunk and its quantized reconstruction.",
    "动作块与量化重建结果之间的平均绝对误差。",
  ],
  loss_vq_commitment: [
    "Sum of the residual quantizer commitment terms before their objective multiplier.",
    "残差量化各级承诺损失之和，按乘入总目标前的尺度记录。",
  ],
  mean_batch_distinct_codes: [
    "Distinct code indices counted per quantizer level and summed within each microbatch, then averaged across microbatches and ranks.",
    "各量化级分别计数不同编码并在微批次内相加，再跨微批次和进程取均值。",
  ],
  mean_batch_distinct_combinations: [
    "Number of different code tuples in each local microbatch, averaged across microbatches and ranks.",
    "每个本地微批次中不同编码组合的数量，再跨微批次和进程取均值。",
  ],
  mean_batch_max_action_error: [
    "Largest absolute action-coordinate error in each local microbatch, averaged across microbatches and ranks.",
    "先取各本地微批次最大的动作坐标绝对误差，再跨微批次和进程取均值。",
  ],
  active_action_dims: [
    "Number of unmasked action coordinates across the predicted timesteps of one sample, averaged over samples.",
    "对每个样本统计整个预测时段的有效动作坐标数，再按样本平均。",
  ],
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
  loss_action: ["Action prediction term emitted by the policy.", "策略返回的动作预测损失项。"],
  loss_video: ["Video prediction term emitted by the policy.", "策略返回的视频预测损失项。"],
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
    "Latest held-out evaluation using the policy's validation objective and denominators.",
    "最近一次留出集评估的损失，使用该策略的验证目标与分母。",
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
  eval_steps: [
    "Successful optimizer updates between validation passes.",
    "两次验证之间的成功更新步数。",
  ],
  save_freq: ["Optimizer steps between checkpoints.", "保存 checkpoint 的优化器步数间隔。"],
  log_freq: ["Interval for console summaries, in optimizer steps.", "控制台汇总的步数间隔。"],
};
export const $ = (id) => document.getElementById(id);
const requested =
  new URL(location.href).searchParams.get("lang") || localStorage.getItem("am:lang") || "en";
export let lang = requested.startsWith("zh") ? "zh" : "en";
export function toggleLang() {
  lang = lang === "zh" ? "en" : "zh";
  localStorage.setItem("am:lang", lang);
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
  return key;
}
export function fmt(v, compact = false) {
  if (!Number.isFinite(v)) return "—";
  if (!compact && Number.isSafeInteger(v)) return v.toLocaleString(lang);
  const a = Math.abs(v);
  if (a && (a < 0.001 || a >= 1e8)) return v.toExponential(2);
  return v.toLocaleString(lang, { maximumSignificantDigits: compact ? 4 : 6 });
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
  };
  let text = help[key]?.[i] || help[alias[key]]?.[i] || "";
  if (run?.policy === "openwam" && ["loss_action", "loss_video"].includes(key))
    text += [" Includes the configured objective weight.", " 已包含该项的配置权重。"][i];
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
      "OpenWAM objective = weighted video loss + weighted action loss.",
      "OpenWAM 目标损失等于加权视频损失与加权动作损失之和。",
    ][i];
    const model = run.config?.model;
    if (Number.isFinite(model?.lambda_video) && Number.isFinite(model?.lambda_action)) {
      text +=
        [" Video / action weights ", " 视频／动作权重 "][i] +
        fmt(model.lambda_video) +
        " / " +
        fmt(model.lambda_action);
    }
  } else if (key === "loss" && run?.policy === "act") {
    text = [
      "ACT uses action L1 and adds weighted KL during VAE training. Validation reports action L1.",
      "ACT 使用动作 L1，启用 VAE 的训练另加加权 KL。验证记录动作 L1。",
    ][i];
  } else if (key.startsWith("module/")) {
    const quantity = key.split("/")[2];
    text =
      {
        relative_update: [
          "L2 norm of the actual optimizer update divided by the pre-update weight norm.",
          "实际优化器更新的 L2 范数除以更新前的参数范数。",
        ],
        update_norm: [
          "L2 norm of the sampled post-step minus pre-step parameters.",
          "抽样更新前后参数差的 L2 范数。",
        ],
        weight_norm: [
          "Pre-update L2 norm of the sampled trainable parameters.",
          "抽样模块可训练参数在更新前的 L2 范数。",
        ],
        grad_norm: [
          "Unscaled module gradient L2 norm before clipping.",
          "抽样模块解除缩放后、裁剪前的梯度 L2 范数。",
        ],
      }[quantity]?.[i] || "";
  }
  if (key.startsWith("lr/"))
    text = [
      `Learning rate used by parameter group ${key.slice(3)} for this update.`,
      `本次更新中参数组 ${key.slice(3)} 的实际学习率。`,
    ][i];
  if (spec) {
    const parts = [
      spec.source !== key && spec.source,
      spec.reduction &&
        t("reduction") +
          " " +
          (spec.reduction === "mean"
            ? t("mean")
            : spec.reduction === "sum"
              ? ["sum", "求和"][i]
              : spec.reduction),
      spec.denominator && t("denominator") + " " + spec.denominator,
      spec.scale != null && spec.scale !== 1 && t("scale") + " " + spec.scale,
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
    node.tabIndex = 0;
    node.setAttribute("aria-description", description);
  } else {
    delete node.dataset.help;
    node.removeAttribute("tabindex");
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
