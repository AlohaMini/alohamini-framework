"""RLT supervised and actor-critic training stages."""

import argparse

from alohamini.rl.rlt.config import RLTConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alohamini train rlt")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("token", help="用离线示教训练 RL token 的编码器与解码器")
    train.add_argument("--config", required=True, help="训练配置 JSON")
    train.add_argument("--output-dir", help="Token 权重和验证结果保存位置")
    train.add_argument("--baseline-checkpoint", help="在同一验证集上评估的已有 token 权重")
    gate = commands.add_parser("gate", help="按完整回合留出，训练可重复进入的阶段分类器")
    gate.add_argument("--config", required=True)
    gate.add_argument("--intervals", required=True, help="带帧边界的阶段标注 JSON")
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
    ac = commands.add_parser("ac", help="用已评分回合离线训练残差 Actor 和双 Critic")
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
        command.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
        command.add_argument("--seed", type=int)
        command.add_argument("--mixed_precision", choices=("none", "bfloat16", "float16"))
    for command in (train, ac, gate):
        command.add_argument(
            "--learned-action-names", nargs="+", help="选择动作坐标名称或 arm_left、arm_right 组"
        )
    for command in (ac, train):
        command.add_argument("--save_freq", type=int)
    args = parser.parse_args(argv)
    config = RLTConfig.read(args.config, stage=args.command, **RLTConfig.overrides(vars(args)))
    if args.command == "token":
        from alohamini.rl.rlt.token_training import train_token

        train_token(config, baseline_checkpoint=args.baseline_checkpoint)
    elif args.command == "gate":
        from alohamini.rl.rlt.gate_training import train_gate

        train_gate(config, intervals=args.intervals)
    elif args.command == "ac":
        from alohamini.rl.rlt.ac_training import train_ac

        train_ac(config, resume=args.resume)
    return 0
