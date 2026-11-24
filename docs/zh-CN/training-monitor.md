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

## 云端训练，本地查看

训练机上启动监控，指定实际训练输出目录。使用自定义控制台目录时加 `--logs-dir /path/to/logs`。

```bash
alohamini train monitor --runs-dir /path/to/runs --port 7860
```

本机打开 SSH 隧道，再访问 `http://127.0.0.1:7861/`。将 `user@training-host` 换为可登录训练机的 SSH 地址。查看期间保持两个命令运行。本机只需 SSH 和浏览器。

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:7861:127.0.0.1:7860 user@training-host
```

## 查看训练

- 首次打开使用英文。顶部标签切换页面。可收起的侧栏显示任务、算法和当前任务的更新时间，勾选其他任务可叠加对比。语言与自动刷新位于右上角，窄屏时位于侧栏底部。链接可指定 `?lang=zh` 或 `?lang=en`。
- **指标**按任务展示总损失、全部已记录的策略分项与学习率，曲线每页最多八张。用“选择指标”搜索字段并组合曲线。图表按可用宽度自动换行。“显示设置”可调整横轴、趋势窗口和桌面最小图宽（默认 640 px）。浅线为单步值，趋势线为完整记录窗口的算术均值，虚线为验证。点击图例显隐，悬停或聚焦指标名查看定义。
- 拖动**区间**的两个手柄可调整所有总览曲线的起止位置，点击区间数值可精确输入。“完整历史”随新增记录延伸。**放大曲线**提供独立的区间滑块、拖选放大、平移、撤销和纵轴设置。横轴可选择已记录的步数、样本数或经过时间。
- **诊断**对比前后两个训练窗口的均值、有效记录数和分项变化。验证区显示最近与全程最低的验证损失，输入目标上限可查首次达标步数。展开查看分布、梯度裁剪、学习率、耗时和已采样的模块更新。
- **配置**可搜索分词器、基座、初始化权重、冻结参数和优化器设置。展开资源查看记录的路径、版本或哈希，也可导入当前任务的预检 JSON。**日志**可切换实时训练控制台与原始 JSONL。
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
| `/runs`、`/runs/{id}` | 列表含 `id`、`version`、`metadata_version`。详情含各阶段已记录的 `policy_metrics`、配置、`schema.runtime`、`preflight`、`launch_preflight`、控制台计数及离线报告。`brief=1` 只取动态摘要。 |
| `/runs/{id}/series` | 最多 32 个 `metric`，`axis=step\|samples\|elapsed_s`，闭区间 `min/max`，每条曲线 `points=16..4096`（默认 800）。`mean_window=0..4096`，默认关闭。 |
| `/runs/{id}/diagnostics` | `metric`、`window=2..4096`、可选验证目标 `target`。返回基于原始训练记录的窗口统计、优化器诊断与验证记录。 |
| `/runs/{id}/records`、`/runs/{id}/console` | 原始记录使用 `phase=train\|validation`、`limit=1..1000`、`cursor`、可选 `tail=1`。控制台使用日志名 `name` 和字节游标。 |
| `/datasets`、`/datasets/{id}`、`/datasets/{id}/trajectory` | 数据集列表、元信息（可传 `run` 核对清单）、指定 `episode`、`field`、`points` 的帧索引曲线。 |

曲线按 `train` 和 `validation` 返回 `extent`、`cursor`、`series` 和可选 `means`。各项含 `count` 与 `points: [[x,y], ...]`。保持指标、坐标、区间、分辨率和窗口不变，以 `after_train`、`after_validation` 回传游标。`append` 追加点与计数，`replace` 替换该阶段。原始记录可分页读取，异常行计数独立返回。
