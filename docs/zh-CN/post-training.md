# 策略后训练

[English](../post-training.md) | **简体中文**

在 PC 上激活 `alohamini` 环境，准备好已训练的原生策略 checkpoint，并启动机器人 Host。采集期间策略权重固定，每轮训练结束后加载新 checkpoint 再部署。

| 目标 | 采集入口 | 下一步 |
|---|---|---|
| 用人工纠错改进现有策略 | `alohamini collect dagger` | 合并原始示范与纠错片段，从原 checkpoint 监督微调 |
| 用成功／失败奖励训练 π0.5 残差策略 | `alohamini collect rlt` | 用 RLT 评分回合离线训练 Actor/Critic |
| 记录普通策略的完整交互与评分 | `alohamini collect scored` | 生成含执行回执和奖励的 v3 数据集 |

## DAgger 纠错与微调

连接已标定主臂，在 X11 键盘会话运行：

```bash
alohamini collect dagger --host HOST --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint --dataset task_corrections --task "TASK"
```

Space 暂停并保持从臂。用 `--dagger.handover_mode` 选择接管方式。

| 模式 | 暂停后开始纠错 |
|---|---|
| `aligned`（默认） | 按 Tab，手动对齐各关节和夹爪。关节稳定后逐个保持；全部到位后扶稳主臂，再按 Tab 开始纠错。 |
| `relative` | 按一次 Tab，以此时主从臂各自的位置为起点。移动主臂，位移映射为从臂的绝对关节目标。主臂保持无力矩。 |

纠错期间底盘停止、升降保持。按 Tab 结束并保存该段纠错，按 Space 交还策略，从当前观测重新推理。Esc 退出。
对齐模式的主臂保持输出上限默认为 `0.2`（`--dagger.hold_output_limit`），关节容差默认为 3 度（`--dagger.alignment_tolerance_deg`）。保持期间需扶稳主臂；故障、反馈中断或对齐超时会释放保持。
慢速同步推理在当前计算返回后响应暂停；需要异步推理时添加 `--inference.type=async`。DAgger 接管要求手臂维度未固定。

合并示范与纠错，开始新一轮微调：

```bash
alohamini dataset edit --operation.type merge \
  --operation.roots '["~/Alohamini_workspace/datasets/task_demo", "~/Alohamini_workspace/datasets/task_corrections"]' \
  --output ~/Alohamini_workspace/datasets/task_merged

alohamini train \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_merged \
  --output_dir ~/Alohamini_workspace/runs/act_dagger_01 \
  --policy.device cuda --steps 10000 --batch_size 8 --background
```

两份数据须有相同字段、相机、标定和 FPS；按合并后的回合编号指定独立验证集。
该命令从原权重开始新一轮微调。后台日志位于 `~/Alohamini_workspace/logs/training/act_dagger_01.log`。
下一轮运行 `alohamini collect dagger` 时，用 `--policy.path` 指定新的 checkpoint。

## RLT 离线强化学习

在同一工作目录创建 `run.json`。填入已训练的原生 π0.5 关节坐标 checkpoint、示范数据集、任务指令和机器人 Host 地址。Token 训练至少需要两个有效示范回合，训练和验证按回合划分。

```json
{
  "checkpoint": "/path/to/pi05/pretrained_model",
  "dataset": "/path/to/demonstrations",
  "task": "Place the block in the tray",
  "host": "HOST",
  "output_dir": "token-run",
  "chunk_size": 10
}
```

在该目录依次运行四条命令，完成 token 训练、基座采集、离线 Actor/Critic 训练和残差策略部署。

```bash
alohamini train rlt token --config run.json --steps 5000 --batch_size 4

alohamini collect rlt --config run.json --output-dir round-1 --policy.n_action_steps 5

alohamini train rlt ac --config run.json --dataset round-1/lerobot_v3 \
  --token-checkpoint token-run/token --output-dir ac-run --steps 5000

alohamini collect rlt --config run.json --output-dir round-2 --policy.n_action_steps 5 \
  --token-checkpoint token-run/token --actor-checkpoint ac-run/actor
```

- **第一轮采集**：按 H 相对接管或交还。按 C 标记残差阶段；全程未标记时，AC 使用整个回合中有执行回执的有效动作。按 → 结束，场景复位后输入 0（失败）或 1（成功）并回车，评分回合自动写入 `round-1/lerobot_v3/`。保留成功与失败回合。
- **第二轮部署**：按 C 开启或退出 Actor 修正，阶段外由基座执行。示例每次预测 10 步，最多执行 5 步后重新规划。两轮使用相同的基座、动作坐标和动作块长度。

续训、自动阶段切换（gate）、补评分及导出恢复见 [RLT 使用说明](../../src/alohamini/policies/rlt/README.md)。

## 策略评分回合

```bash
alohamini collect scored --host HOST --robot-model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --task "拿起物体" --output-dir ~/Alohamini_workspace/datasets/scored
```

Space 暂停或继续。加 `--intervention` 可用 H 进行主臂相对接管和交还。单臂接管用 `--intervention left` 或 `--intervention right`。任务结束按 →，提前截断按 T。场景复位就绪后输入 0 或 1 并回车，开始下一回合。Q 退出，未完成回合保存在 `pending/`。

已评分回合自动导出到 `lerobot_v3/`。每步记录控制来源、请求目标和 Host 回执，`experience.applied_mask` 标明有回执的动作维度。终点观测的前一步用 `next.done` 标记回合结束，`next.terminated` 和 `next.truncated` 区分任务结束和提前截断。追加采集用 `collect scored --resume`，重试导出用 `dataset export DIR --format lerobot-v3 --output DIR/lerobot_v3`。

标准动作训练只读取全部动作维度均有回执的窗口。仅回报手臂目标的 Host 会将底盘和升降标记为未确认。已封存回合可用 `alohamini collect score --output-dir DIR --episode ID --reward 0` 补交评分，成功填 `1`。
