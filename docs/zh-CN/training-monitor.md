# 本地训练监控

[English](../training-monitor.md)

在已安装 AlohaMini 的环境中启动。默认地址为 `http://127.0.0.1:7860/`。查看指标与日志只需平台包，读取评估轨迹还需 PyArrow，已包含在训练环境中。

```bash
conda activate alohamini
alohamini train monitor --open
alohamini train monitor --runs-dir /path/to/runs --datasets-dir /path/to/datasets
```

工作区默认为 `~/Alohamini_workspace`，可用 `ALOHAMINI_WORKSPACE` 修改。

| 启动参数 | 用途 |
| --- | --- |
| `--runs-dir PATH` | 单次训练目录或其父目录，默认使用工作区的 `runs/`。每次续训单独展示。 |
| `--datasets-dir PATH` | 原生 AlohaMini 或 LeRobot v3 评估数据集或其父目录。可重复，默认使用工作区的 `datasets/`。 |
| `--logs-dir PATH` | 后台启动配置和控制台日志目录，默认使用工作区的 `logs/training/`。 |
| `--port N`、`--open` | 指定回环端口，`0` 自动选择。用 `--open` 打开浏览器。 |
| `--allow-origin ORIGIN` | 允许指定 Studio 浏览器来源读取 API，可重复指定。 |

## 查看训练

- 首次打开使用英文。右上角切换中英文，也可指定 `?lang=zh` 或 `?lang=en`。桌面侧栏可收起，勾选其他任务可叠加对比。
- **指标**按策略已记录的损失项和参数组选择曲线。悬停指标名查看定义。浅线为单步值，实线趋势为完整记录窗口的算术均值，虚线为验证。点击图例显隐曲线。长区间按指标保留首尾、分桶极值及缺口。
- **放大曲线**可独立拖选区间、前移或后移、撤销、恢复全程，以及切换自适应或包含零点的纵轴。放大后重新读取所选区间。总览横轴可选择已记录的步数、样本数或经过时间。
- **诊断**显示最近窗口的分布、P50/P95、波动、每步斜率、分项前后窗口均值、学习率、裁剪和耗时。输入验证损失阈值可查看首次达标的实测记录。旧日志按裁剪前范数和明确阈值推算的裁剪比例会注明来源。
- 展开顶部的**训练设置**可查看启动时的冻结模块、参数量、优化器参数组及预检。支持当前 `train check` 报告和历史 `RUN.preflight.json`，也可在此导入文件。旧控制台若仅有参数总数与可训练数，则展示汇总。**配置**可按参数名或值搜索，**日志**可切换原始 JSONL 和控制台。
- **评估**支持导入本地录制或离线报告（JSON，最大 4 MiB）。录制按回合和通道查看原始帧轨迹；报告按实际字段展示各通道 MAE、偏差、RMSE 及预测偏移的 MAE。数据来源与 checkpoint 清单匹配状态随数据展示。导入选择保存在当前浏览器中。

## 可选训练诊断

| 训练参数 | 采集内容 |
| --- | --- |
| `--diagnostics.modules='["model.action_head"]' --diagnostics.update_every=500` | 示例为 ACT 动作头。仅在指定更新步复制这些模块的可训练参数，记录裁剪前梯度、实际更新范数及更新/参数比。支持单设备和 DDP。 |
| `--diagnostics.train_eval_samples=64` | 在已有周期验证时，用相同权重及评估模式额外评估固定训练子集，记录 `train_eval_loss` 和 `generalization_gap`。 |

两项默认关闭。冻结快照在启动时采集一次。离线动作误差随已有预测聚合。新训练每次成功更新写入计数、样本数、单调耗时与事件时间。历史记录的缺失字段显示为空。关闭查看器后训练继续运行。

## 文件和 Studio API

训练目录包含 `train.json`、`metrics.jsonl`、`validation-metrics.jsonl`、`metrics-schema.json`、`metrics-resources.json` 和目录级 `offline-evaluation.json`。续训使用对应的 `from-STEP-ATTEMPT` 指标、预检和 `resume-STEP-ATTEMPT.json` 配置。初始任务也可读取该目录中的 `resources.json` 和日志目录中的同名启动预检。归档布局 `runs/RUN` 和相邻 `logs/RUN.preflight.json` 可自动识别。

只读 HTTP GET 接口前缀为 `/api/v1`。Studio 可通过同源代理访问，代理上游使用回环 Host，或用 `--allow-origin` 指定浏览器来源。

| 路径 | 参数与返回值 |
| --- | --- |
| `/runs`、`/runs/{id}` | 列表含 `id`、`version`、`metadata_version`。详情含阶段、配置、`schema.runtime`、`preflight`、`launch_preflight`、控制台计数及离线报告。`brief=1` 只取动态摘要。 |
| `/runs/{id}/series` | 最多 32 个 `metric`，`axis=step\|samples\|elapsed_s`，闭区间 `min/max`，每条曲线 `points=16..4096`（默认 800）。`mean_window=0..4096`，默认关闭。 |
| `/runs/{id}/diagnostics` | `metric`、`window=2..4096`、可选验证目标 `target`。返回基于原始训练记录的窗口统计、优化器诊断与验证记录。 |
| `/runs/{id}/records`、`/runs/{id}/console` | 原始记录使用 `phase=train\|validation`、`limit=1..1000`、`cursor`、可选 `tail=1`。控制台使用日志名 `name` 和字节游标。 |
| `/datasets`、`/datasets/{id}`、`/datasets/{id}/trajectory` | 数据集列表、元信息（可传 `run` 核对清单）、指定 `episode`、`field`、`points` 的帧索引曲线。 |

曲线按 `train` 和 `validation` 返回 `extent`、`cursor`、`series` 和可选 `means`。各项含 `count` 与 `points: [[x,y], ...]`。保持指标、坐标、区间、分辨率和窗口不变，以 `after_train`、`after_validation` 回传游标。`append` 追加点与计数，`replace` 替换该阶段。原始记录可分页读取，异常行计数独立返回。
