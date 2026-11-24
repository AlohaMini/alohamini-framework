# 5090 上的 π0.5 人工示教与 RL token 训练

在 5090 的前台终端运行下列命令。配置目录里有两份文件，`collect.json` 指向主臂串口和标定，`token.json` 指向离线示教数据和 π0.5 checkpoint。

```bash
conda activate alohamini_rlt
cd ~/Alohamini_software_RLT
CFG="$HOME/Alohamini_workspace/configs/rlt_pi05_20261118"
```


## 已评分回合的 LeRobot v3 数据

采集器会把已评分的回合追加到 `$OUT/lerobot_v3`。原来的 `rlt collect ... --output-dir "$OUT" --resume` 命令可以继续使用。视频在独立的后台进程编码，退出采集时会等已提交的回合写完。未评分回合保留在 `pending/`，补交评分后也会导出。手动补导或恢复中断的导出可运行：

```bash
python -m alohamini rlt export-v3 --output-dir "$OUT"
```

- `videos/observation.images.<相机名>/chunk-000/file-000.mp4` 可以用普通播放器查看。每路沿用采集器的 H.264、YUV420p 和 CRF 18。导出时先按 Host JPEG 约定还原 RGB。读取 MP4 时直接解码为 RGB。
- 每个回合的末帧是复位前的终点观测。Parquet 的 `rlt.action_valid` 标明可用于 AC 的动作，`rlt.source` 的 1、2、3 分别表示 VLA、RLT、人工。未生效、HOLD 和终点占位动作均有明确标记。手动评分保存在回合元数据中，最后一个有效动作同时带有 `next.reward` 和 `next.done`。
- 迁移时复制整个 `lerobot_v3`，包含 `meta/rlt`。要按本项目的 AC 规则读取单个回合，向 `alohamini.rlt.replay.materialize_episode(policy, directory)` 传入 `lerobot_v3/meta/rlt/episode_000000`。读取器会用视频及保存的 Host 状态重建窗口起点的完整 π0.5 参考动作。回执、接管来源和真实时间戳随数据集保存。

## 边推理边采集

确认你自己的 Host 已就绪，左主臂接在 5090，场地和物体已摆好。采集配置沿用 30 Hz、15 步动作块和前视、胸部、左腕三路模型输入，固定右臂和升降轴。启动时会先恢复固定轴目标，就绪后自动开始第一回合。

```bash
PI05_CKPT="$HOME/Alohamini_workspace/runs/pi05_module_to_shape_12_flow_full_3cam_90train6val_b32_span035_20261101/checkpoints/010000/pretrained_model"
TASK="First, pick up the green module and place it into the circular slot on the right. Next, pick up the orange module and place it into the pentagonal slot in the middle. Finally, pick up the purple module and place it into the square slot on the left."
OUT="$HOME/Alohamini_workspace/runs/pi05_rlt_relative_demo"
python -m alohamini rlt collect \
  --host 192.168.50.2 \
  --robot_model alohamini2pro \
  --policy.path "$PI05_CKPT" \
  --task "$TASK" \
  --device cuda --fps 30 --policy.n_action_steps 15 \
  --fixed-dimensions arm_right lift_axis \
  --left-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E112729-if00 \
  --leader-id am_leader_bi \
  --calibration-dir "$HOME/Alohamini_workspace/calibration/rlt_legacy_20261118/leader" \
  --output-dir "$OUT" --resume
```

换任务时，修改 `TASK` 和 `OUT`，按需更换 `PI05_CKPT`。使用固定轴时，checkpoint 里要有对应的固定目标。相机顺序来自 checkpoint。

也可以沿用 5090 的主臂串口、标定等默认值，直接在命令里覆盖任务参数：

```bash
python -m alohamini rlt collect --config "$CFG/collect.json" \
  --policy.path "$PI05_CKPT" --task "$TASK" --fixed-dimensions arm_right lift_axis \
  --output-dir "$OUT" --resume
```

| 时机 | 操作 |
| --- | --- |
| π0.5 执行中，需要示教 | 按 `h`，从臂从当前位置跟随左主臂的相对位移。再次按 `h` 交还 π0.5。每回合均可接管零次或多次。 |
| 回合结束 | 按 **→**。程序收尾记录、保存终点，并让左臂和夹爪回到本回合起始姿态。 |
| 左臂回位、物体摆好、手已离开后 | 输入 `1`（成功）或 `0`（失败），按回车。程序提交本回合并自动开始下一回合。 |
| 暂停、继续或退出 | 空格暂停或继续，`q` 退出。若在回位结束前已评分，依提示在复位后再按回车确认。 |

采集 π0.5 示教时，可按 `c` 标注阶段开始，再按一次标注阶段结束。整回合都不按 `c` 时，新版将有 Host 应用回执的 VLA 和 HUMAN 动作纳入整回合训练范围。每步观测、三路图像、动作来源和回执都随回合保存，便于事后按画面复查。

同一个 `OUT` 配合 `--resume` 可以续采。已评分的回合在 `$OUT/episodes/<回合ID>/`，未提交的回合在 `$OUT/pending/`。进行中退出的回合保留为未完成。对已结束且终点和命令回执完整的回合，可在退出采集后补交评分：

```bash
read -rp '待补评分的回合 ID: ' EPISODE
python -m alohamini rlt collect --output-dir "$OUT" \
  --score-episode "$EPISODE" --reward 1
```

## 离线训练编码器与解码器

训练读取 `token.json` 中的 `module_to_shape_14` 示教集，冻结原 π0.5 并按回合划分训练和验证集。使用新输出目录运行一次训练：

```bash
python -m alohamini rlt train-token --config "$CFG/token.json" \
  --output-dir "$HOME/Alohamini_workspace/runs/rlt_pi05_token_$(date +%Y%m%d_%H%M%S)"
```

`token.pt` 是验证重建损失最低时的权重，`token-last.pt` 是最后一次保存的权重，训练和验证曲线在 `token-metrics.jsonl`。π0.5 示教采集本身可以在重新训练 token 之前运行。

## 用人工评分训练 Actor–Critic

在 5090 的 `alohamini_rlt` 环境中，从已评分的采集目录训练。以下例子每 15 步取一个窗口，与该配置的动作块长度相同。

```bash
RUNS="$HOME/Alohamini_workspace/runs"
CFG="$HOME/Alohamini_workspace/configs/rlt_pi05_20261118"
TOKEN="$RUNS/rlt_module_to_shape_14_token_20261110_b16_s20000/token.pt"
DATA="$RUNS/pi05_rlt_relative_demo"
AC_OUT="$RUNS/rlt_ac_$(date +%Y%m%d_%H%M%S)"
python -m alohamini rlt train-ac --config "$CFG/collect.json" \
  --dataset "$DATA" --token-checkpoint "$TOKEN" --output-dir "$AC_OUT" \
  --ac-steps 5000 --ac-batch-size 128 --replay-stride 15
```

也可以将 `--dataset` 指向完整导出的 `lerobot_v3`。训练只读取已评分且可训练的回合，动作取自 Host 的应用回执。标过 `c` 的回合沿用关键段，没有标段的新格式回合使用全部有效动作。暂停、被替代的命令和被拒绝的命令不作为学习动作。按整回合分训练和验证，同一结果有多个回合时会留出验证回合。

人工 `0/1` 在最后一个有效动作上形成终止奖励。双 Critic 通过带时间折扣的目标学习，Actor 根据 Critic 的评分修正 π0.5 的左臂六关节和夹爪动作。人工接管时用实际执行的动作作模仿目标，自主段用冻结的 π0.5 参考动作作正则。单坐标修正上限由 `ac_residual_limit` 控制，默认是模型归一化单位的 0.25。训练指标写到 `ac-metrics.jsonl`，`ac.pt` 中保存网络、Token 指纹和训练/验证回合 ID。验证损失衡量离线拟合，抓取成功率需要单独实测。

需要用训练后的 Actor 采集时，在新的采集目录启动。沿用普通采集的启动和人工接管流程。

```bash
python -m alohamini rlt collect --config "$CFG/collect.json" \
  --output-dir "${AC_OUT}_collect" --token-checkpoint "$TOKEN" --actor-checkpoint "$AC_OUT/ac.pt"
```

采集器核对基础模型、任务、动作坐标、时间尺度及 Token 指纹。加载 Actor 后，每个新回合先由 π0.5 控制，记录为 `VLA`。到达需要修正的放置阶段时按 `c`，自动动作改由 Actor 修正并记录为 `RLT`，同时保存原 π0.5 参考和 Actor 版本。再按 `c` 恢复 π0.5。同一回合可多次切换，例如分别在橙色和紫色对槽时开启。切换会清空旧动作，等待新观测的策略结果时保持当前位置。`h` 可随时接管，再按 `h` 回到当前阶段选择的自动策略。离线标注用于训练，现场阶段边界由操作者按 `c` 指定。

## 训练自动阶段分类器（Gate）

按带帧边界的放置区间标注文件（`rlt-placement-intervals.json`），复用冻结的 RL Token 和归一化左臂本体状态训练可重复进入/退出的阶段分类器：

```bash
GATE_OUT="$RUNS/rlt_gate_$(date +%Y%m%d_%H%M%S)"
INTERVALS="$DATA/rlt-placement-intervals.json"
python -m alohamini rlt train-gate --config "$CFG/collect.json" \
  --dataset "$DATA/lerobot_v3" --intervals "$INTERVALS" \
  --token-checkpoint "$TOKEN" --validation-fraction 0.24 --gate-eval-stride 20 \
  --output-dir "$GATE_OUT"
```

训练按完整回合划分为训练组、选权组和最终留出组（因此标注文件中至少需要 3 条涵盖各放置阶段的已评分回合），分别训练 `state_only`、`token_only` 与 `token_state` 三个探针，并在 `gate-validation.json` 中报告不同投票步距下的 Precision、Recall、误入与漏检。`gate.pt` 保存联合分类器及配套 Token 指纹。

`rlt-placement-intervals.json` 的格式示例（帧号为回合内左闭右开区间 `[start_frame, end_frame_exclusive)`）：

```json
{
  "version": 1,
  "fps": 30,
  "episodes": [
    {
      "episode_id": "<32位十六进制回合ID>",
      "segments": [
        {"name": "orange_placement", "start_frame": 310, "end_frame_exclusive": 420, "reward": 1},
        {"name": "purple_placement", "start_frame": 620, "end_frame_exclusive": 740, "reward": 1}
      ]
    }
  ]
}
```

实机加载分类器自动切入/退出对槽阶段（默认连续 2 次新观测概率 `>= 0.75` 进入、连续 2 次 `<= 0.35` 退出）：

```bash
python -m alohamini rlt collect --config "$CFG/collect.json" \
  --token-checkpoint "$TOKEN" --gate-checkpoint "$GATE_OUT/gate.pt" \
  --actor-checkpoint "$AC_OUT/ac.pt" --output-dir "${AC_OUT}_gated"
```

只加载 `--token-checkpoint` 与 `--gate-checkpoint`（不传 `--actor-checkpoint`）时，仅记录分类器概率，实际控制动作仍全部来自 π0.5，便于先做影子验证。

## 夹爪张大自动退出 RLT 兜底机制

加载 Actor 或阶段分类器时，默认启用基于左从臂夹爪实测开度的退出兜底规则：进入对槽阶段后，程序先确认夹爪实测开度 `<= 18%`（已夹持），随后连续 3 份不同的新 Host 反馈达到 `>= 24%`（已松爪），即立刻清除剩余 Actor 动作、退出本段 RLT 并切回 π0.5 重新规划；加载分类器时，需等夹爪重新回到夹持范围后才允许再次自动进入，防止松爪后反复误触发。

```bash
python -m alohamini rlt collect --config "$CFG/collect.json" \
  --token-checkpoint "$TOKEN" --actor-checkpoint "$AC_OUT/ac.pt" \
  --gripper-exit-closed-pct 18 --gripper-exit-open-pct 24 \
  --gripper-exit-confirmations 3 --output-dir "${AC_OUT}_release"
```

纯 π0.5 示教采集如需启用该规则可显式加 `--gripper-exit`；如需关闭兜底、完全手动按 `c` 控制可加 `--no-gripper-exit`。每次自动退出的实测开度、反馈序号与控制代次都会写入 `episode.json` 的 `gripper_exits`。

## 标定

5090 本地的映射在 `~/Alohamini_workspace/calibration/rlt_legacy_20261118/mapping.json`。采集使用该目录下与 USB 序列号匹配的左主臂文件。主臂为被动读取，接管时锁存主从臂各自起点。配置的 `calibration_mode` 为 `strict`，执行前会将实际 Host 的标定与模型元数据比对；若失败，应确认接入的是采集训练数据时使用的设备和标定。
