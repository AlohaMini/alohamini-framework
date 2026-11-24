# 本地训练监控

[English](../training-monitor.md)

```bash
alohamini train monitor --open
```

默认地址为 `http://127.0.0.1:7860/`。训练前、训练中或结束后均可启动，直接读取框架已有的输出文件。关闭查看器后，训练继续运行。

| 参数 | 用途 |
| --- | --- |
| `--runs-dir PATH` | 单次训练输出目录，或包含多个训练目录的父目录。默认使用工作区的 `runs/`。 |
| `--datasets-dir PATH` | 一个评估数据集或包含多个数据集的目录，可重复指定。默认使用工作区的 `datasets/`。 |
| `--logs-dir PATH` | 后台训练的启动配置和同名控制台日志目录。默认使用工作区的 `logs/training/`。 |
| `--port N` | 本机监听端口。传入 `0` 自动选择空闲端口并打印地址。 |
| `--allow-origin ORIGIN` | 允许指定 Studio 浏览器来源读取数据，例如 `http://localhost:5173`。可重复指定。 |

工作区默认位于 `~/Alohamini_workspace`，可通过 `ALOHAMINI_WORKSPACE` 设置。指标与日志查看只需平台包。评估轨迹还需要 PyArrow，训练环境和 PC 环境已包含该依赖。

## 查看与对比

- 点击任务查看最近一次训练和验证记录，勾选其他任务可叠加曲线、并排比较配置。从同一 checkpoint 发起的多次续训分别展示。
- 在“选择指标”中增减面板。拖动曲线或输入区间可查看局部记录，“完整历史”恢复全图。鼠标悬停或使用左右方向键可读数。指标旁的问号说明含义、聚合方式和分母。
- 实线表示训练，虚线表示验证。悬浮读数使用所选坐标的实际记录。面板右上角显示区间内的展示点数及记录数。短区间展示原始点，长区间按指标分别保留首尾、分桶极值及异常缺口。
- “日志”显示原始结构化记录及可用的后台控制台日志，控制台按启动文件名选择。页面只保留最近一段文本，可暂停自动刷新阅读。标签页进入后台会暂停轮询。
- 顶部可切换中英文，也可通过 `?lang=zh` 或 `?lang=en` 指定语言。地址栏包含所选主任务的标识。

训练器在每次成功更新后刷新 `metrics.jsonl`，验证写入 `validation-metrics.jsonl`。续训使用对应的 `*-from-STEP-ATTEMPT.jsonl`、配置和指标定义。新版记录包含累计消耗样本数、本次训练循环的单调计时秒数 `elapsed_s` 及事件发生时的 Unix 时间 `event_time_s`。循环耗时包含中间验证与保存 checkpoint 的停顿。训练记录还包含本次循环累计 AMP 跳步次数、启用裁剪时的裁剪标记，以及多参数组的实际学习率。历史日志使用文件中实际记录的横轴。

## 评估数据

选择“导入评估数据”，添加本地 AlohaMini 原生或 LeRobot v3 数据集。查看器读取元数据，按回合和动作通道读取 Parquet 数值列。轨迹横轴为录制的帧索引，可叠加同名且声明单位兼容的观测通道，缺失单位标为“单位未声明”。若 `meta/evaluation.json` 记录了带编号的 checkpoint 和清单 SHA-256，可与当前训练目录中的清单核对。

“评估”同时展示训练目录当前的 `offline-evaluation.json`。可从文件选择器导入含 `mae_by_action` 的离线评估 JSON（最大 4 MiB），按各动作通道原始尺度查看 MAE。最终报告属于整个输出目录。数据集选择和导入报告保存在当前浏览器的主任务视图中，移出视图会保留源文件。

## Studio 接口

在 `/api/v1` 下使用只读 HTTP GET 接口，返回 JSON。训练与数据集 ID 由列表接口提供。Studio 可通过同源代理访问，代理的上游 Host 使用回环地址。浏览器直连时可在启动查看器时指定允许的来源。

| 接口 | 参数与结果 |
| --- | --- |
| `/runs` | 返回 `runs[]`，含训练尝试的 `id`、名称、续训标识、`version` 和 `metadata_version`。版本变化后再取详情。 |
| `/runs/{id}` | 配置、分项指标定义、控制台日志名、最终离线报告。各阶段含指标名、可用横轴、最新记录与异常计数。`brief=1` 省略配置、定义、报告、日志名。 |
| `/runs/{id}/series` | 可重复传入最多 32 个 `metric`。横轴 `axis=step\|samples\|elapsed_s`，闭区间 `min`、`max`，每阶段每指标点数 `points=16..4096`。默认查询 loss、step、800 点。 |
| `/runs/{id}/records` | 原始结构化记录。`phase=train\|validation`、`limit=1..1000` 和可选 `cursor`，`tail=1` 从最新一页开始。返回 `records`、`cursor`、`more`、`reset` 和 `invalid_lines`。 |
| `/runs/{id}/console` | 传入详情中的日志 `name` 和可选字节游标 `cursor`，返回 UTF-8 文本 `text`、`cursor` 与 `reset`。 |
| `/datasets`、`/datasets/{id}` | 数据集列表及元信息。详情可带 `run={attempt_id}` 核对记录的 checkpoint 清单。 |
| `/datasets/{id}/trajectory` | 传入回合 `episode`、完整动作字段名 `field` 和可选 `points`，返回帧索引曲线、帧数与声明单位。 |

```bash
curl -s http://127.0.0.1:7860/api/v1/runs
curl -s "http://127.0.0.1:7860/api/v1/runs/$RUN_ID/series?metric=loss&metric=grad_norm&points=800"
```

曲线响应的 `train` 和 `validation` 各自包含 `mode`、`cursor`、完整横轴范围 `extent`，以及 `series[指标名] = {count, points: [[x, y], ...]}`。保持指标、区间和点数不变，分别用 `after_train`、`after_validation` 回传游标。`append` 表示追加点并累加记录数，`replace` 表示整体替换该阶段。缩放视图或需要重新分桶时，请求新的全景或区间快照。游标按记录计数，同一步的多条记录均保留。

服务按需索引完整的 JSONL 行，在内存中保留最近查看的六个训练尝试。文件替换、截断或索引淘汰会使旧游标失效。格式错误行单独计数，缺失或非有限标量显示为缺口。定量分析可分页读取 `records` 或直接读取原始 JSONL。
