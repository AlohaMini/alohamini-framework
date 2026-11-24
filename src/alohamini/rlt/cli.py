"""RL token training and relative human-in-the-loop collection."""

import argparse
from dataclasses import fields
from pathlib import Path

from alohamini.rlt.config import RLTConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alohamini rlt")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="用 π0.5 连续采集可人工接管的回合")
    collect.add_argument("--config", help="可选的采集默认配置。命令行显式参数优先")
    collect.add_argument(
        "--policy.path", "--checkpoint", dest="checkpoint", help="基座 checkpoint 目录"
    )
    collect.add_argument("--task", help="策略任务提示词")
    collect.add_argument("--host", help="Host 地址")
    collect.add_argument("--robot_model", "--robot-model", dest="robot_model", help="机器人型号")
    collect.add_argument("--device", help="推理设备，例如 cuda")
    collect.add_argument("--fps", type=float, help="控制频率，须与 checkpoint 一致")
    collect.add_argument(
        "--chunk-size", dest="chunk_size", type=int, help="每个动作块的步数"
    )
    collect.add_argument(
        "--fixed-dimensions", nargs="+", metavar="DIM",
        help="使用 checkpoint 保存的固定目标，例如 arm_right lift_axis",
    )
    collect.add_argument("--policy.n_action_steps", dest="execution_steps", type=int)
    collect.add_argument("--learned-action-names", dest="learned_action_names", nargs="+")
    collect.add_argument("--left-port", help="左主臂串口")
    collect.add_argument("--leader-id", help="左主臂标定 ID")
    collect.add_argument("--calibration-dir", help="主臂标定文件目录")
    collect.add_argument("--output-dir", help="回合保存位置")
    collect.add_argument("--token-checkpoint", help="AC 权重配套的冻结 Token 编码器")
    collect.add_argument("--actor-checkpoint", help="按 c 开启对槽阶段时使用的离线 Actor 权重")
    collect.add_argument("--gate-checkpoint", help="阶段分类器 gate，按观测自动进入和退出")
    collect.add_argument("--gate-enter-threshold", type=float)
    collect.add_argument("--gate-exit-threshold", type=float)
    collect.add_argument("--gate-confirmations", type=int)
    collect.add_argument(
        "--gripper-exit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="根据左从臂夹爪实际开度退出对槽阶段；加载 Actor 或阶段分类器时默认启用",
    )
    collect.add_argument("--gripper-exit-open-pct", type=float, help="松爪退出开度，默认 24%%")
    collect.add_argument(
        "--gripper-exit-closed-pct", type=float, help="先确认夹持的开度上限，默认 18%%"
    )
    collect.add_argument("--gripper-exit-confirmations", type=int, help="连续反馈数，默认 3")
    collect.add_argument("--resume", action="store_true", help="检查采集配置并继续追加回合")
    collect.add_argument("--score-episode", help="补交已结束回合的奖励，填写 pending 下的回合 ID")
    collect.add_argument("--reward", type=int, choices=(0, 1))
    train = commands.add_parser("train-token", help="用离线示教训练 RL token 的编码器与解码器")
    train.add_argument("--config", required=True, help="训练配置 JSON")
    train.add_argument("--output-dir", help="Token 权重和验证结果保存位置")
    train.add_argument("--baseline-checkpoint", help="在同一验证集上评估的已有 token 权重")
    gate = commands.add_parser("train-gate", help="按完整回合留出，训练可重复进入的阶段分类器")
    gate.add_argument("--config", required=True)
    gate.add_argument("--intervals", required=True, help="带帧边界的放置阶段标注 JSON")
    gate.add_argument("--dataset", help="采集目录或导出的 LeRobot v3 目录")
    gate.add_argument("--token-checkpoint")
    gate.add_argument("--output-dir", required=True)
    gate.add_argument("--gate-steps", type=int)
    gate.add_argument("--gate-batch-size", type=int)
    gate.add_argument("--gate-sample-stride", type=int)
    gate.add_argument("--gate-eval-stride", type=int, help="附加评估投票步距，可计入推理和回执耗时")
    gate.add_argument("--validation-fraction", type=float, help="按完整回合划分的验证比例")
    gate.add_argument("--gate-enter-threshold", type=float)
    gate.add_argument("--gate-exit-threshold", type=float)
    gate.add_argument("--gate-confirmations", type=int)
    ac = commands.add_parser("train-ac", help="用已评分回合离线训练残差 Actor 和双 Critic")
    ac.add_argument("--config", required=True, help="对应采集任务的配置 JSON")
    ac.add_argument("--dataset", help="原始 RLT 采集目录或独立的 lerobot_v3 目录")
    ac.add_argument("--token-checkpoint", help="冻结的 Token 权重 token")
    ac.add_argument("--output-dir", required=True, help="AC 训练输出目录")
    ac.add_argument("--ac-steps", type=int, help="Critic 更新次数")
    ac.add_argument("--ac-batch-size", type=int, help="每次更新的动作窗口数")
    ac.add_argument("--replay-stride", type=int, help="同段内相邻训练窗口的起点间隔")
    ac.add_argument("--resume", action="store_true", help="从最后一个完整 AC checkpoint 续训")
    for stage, command in (("token", train), ("gate", gate), ("ac", ac)):
        command.add_argument("--steps", dest=stage + "_steps", type=int)
        command.add_argument("--batch_size", dest=stage + "_batch_size", type=int)
        command.add_argument("--device")
        command.add_argument("--lr", type=float)
        command.add_argument("--seed", type=int)
        command.add_argument("--mixed_precision", choices=("none", "bfloat16", "float16"))
    ac.add_argument("--save_freq", type=int)
    export = commands.add_parser("export-v3", help="将已评分的回合补存为 LeRobot v3")
    export.add_argument("--output-dir", required=True, help="原始 RLT 采集目录")
    args = parser.parse_args(argv)
    if args.command == "export-v3":
        from alohamini.rlt.v3 import export_v3

        info = export_v3(args.output_dir)
        count = info["total_episodes"] if info else 0
        print(f"LeRobot v3 共 {count} 个已评分回合，目录 {Path(args.output_dir).expanduser() / 'lerobot_v3'}")
        return 0
    if args.command == "collect" and (args.score_episode is not None or args.reward is not None):
        from alohamini.rlt.storage import RunLock, commit_episode

        episode = args.score_episode or ""
        if (
            args.reward is None
            or len(episode) != 32
            or any(c not in "0123456789abcdef" for c in episode)
        ):
            parser.error("补交评分需要 --score-episode <32位小写十六进制ID> 和 --reward 0 或 1")
        if args.output_dir:
            root = Path(args.output_dir).expanduser().resolve()
        elif args.config:
            root = Path(RLTConfig.read(args.config).output_dir)
        else:
            parser.error("补交评分需要用 --output-dir 或 --config 指定回合目录")
        with RunLock(root / "control.lock"):
            target = commit_episode(root / "pending" / episode, args.reward)
        print(f"已提交回合 {target}")
        from alohamini.rlt.v3 import export_v3

        export_v3(root)
        return 0

    overrides = {
        field.name: vars(args)[field.name]
        for field in fields(RLTConfig)
        if vars(args).get(field.name) is not None
    }
    if not args.config and not {"checkpoint", "task", "output_dir"} <= overrides.keys():
        parser.error("直接采集需要 --policy.path、--task 和 --output-dir，也可用 --config 提供默认值")
    stage = args.command.removeprefix("train-") if args.command.startswith("train-") else None
    config = RLTConfig.read(args.config, stage=stage, **overrides)
    if args.command == "train-token":
        from alohamini.rlt.training import train_token

        train_token(config, baseline_checkpoint=args.baseline_checkpoint)
    elif args.command == "train-gate":
        from alohamini.rlt.gate_training import train_gate

        train_gate(config, intervals=args.intervals)
    elif args.command == "train-ac":
        from alohamini.rlt.ac_training import train_ac

        train_ac(config, resume=args.resume)
    else:
        from alohamini.rlt.runtime import run

        run(config, resume=args.resume)
    return 0
