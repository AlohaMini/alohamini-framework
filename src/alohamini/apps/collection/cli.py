"""Collect policy interactions and score sealed episodes."""

import argparse
import json
import re
from pathlib import Path

from alohamini.apps.collection.native import CollectionConfig


def _add_rlt_arguments(collect):
    collect.add_argument("--config", help="可选的采集默认配置。命令行显式参数优先")
    collect.add_argument(
        "--policy.path", "--checkpoint", dest="checkpoint", help="基座 checkpoint 目录"
    )
    collect.add_argument("--task", help="策略任务提示词")
    collect.add_argument("--host", help="Host 地址")
    collect.add_argument("--robot_model", "--robot-model", dest="robot_model", help="机器人型号")
    collect.add_argument("--device", help="推理设备，例如 cuda")
    collect.add_argument("--fps", type=float, help="控制频率，须与 checkpoint 一致")
    collect.add_argument("--chunk-size", dest="chunk_size", type=int, help="每个动作块的步数")
    collect.add_argument(
        "--fixed-dimensions",
        nargs="+",
        metavar="DIM",
        help="使用 checkpoint 保存的固定目标，例如 arm_right lift_axis",
    )
    collect.add_argument("--policy.n_action_steps", dest="execution_steps", type=int)
    collect.add_argument(
        "--max-joint-velocity-deg-s", type=float, help="人工接管的关节速度上限，默认 60 度/秒"
    )
    collect.add_argument(
        "--return-to-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="评分前将受控关节送回起点",
    )
    collect.add_argument("--left-port", help="左主臂串口")
    collect.add_argument("--right-port", help="右主臂串口")
    collect.add_argument("--leader-id", help="主臂标定 ID")
    collect.add_argument("--calibration-dir", help="主臂标定文件目录")
    collect.add_argument("--output-dir", help="回合保存位置")
    collect.add_argument("--token-checkpoint", help="AC 权重配套的冻结 Token 编码器")
    collect.add_argument("--actor-checkpoint", help="按 c 开启残差控制阶段时使用的离线 Actor 权重")
    collect.add_argument("--gate-checkpoint", help="阶段分类器 gate，按观测自动进入和退出")
    collect.add_argument("--gate-enter-threshold", type=float)
    collect.add_argument("--gate-exit-threshold", type=float)
    collect.add_argument("--gate-confirmations", type=int)
    collect.add_argument(
        "--gripper-exit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按夹爪开度退出残差控制阶段，默认关闭",
    )
    collect.add_argument("--gripper-exit-field", help="松爪退出使用的夹爪位置字段")
    collect.add_argument("--gripper-exit-open-pct", type=float, help="松爪退出开度，默认 24%%")
    collect.add_argument(
        "--gripper-exit-closed-pct", type=float, help="先确认夹持的开度上限，默认 18%%"
    )
    collect.add_argument("--gripper-exit-confirmations", type=int, help="连续反馈数，默认 3")
    collect.add_argument("--resume", action="store_true", help="检查采集配置并继续追加回合")
    collect.add_argument("--learned-action-names", nargs="+", help="选择动作坐标名称或手臂组")


def main(argv):
    parser = argparse.ArgumentParser(prog="alohamini collect")
    from alohamini.apps.collection.dagger import add_dagger_arguments, pop_dagger_options
    from alohamini.apps.policy_cli import add_policy_arguments, run_policy

    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("scored", help="用原生策略采集完整评分回合")
    run.add_argument("--policy.path", dest="checkpoint", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--host", default=CollectionConfig.host)
    run.add_argument("--robot-model", "--robot_model", default=CollectionConfig.robot_model)
    run.add_argument("--device", default=CollectionConfig.device)
    run.add_argument(
        "--calibration-mode",
        choices=("strict", "normalized", "legacy_normalized"),
        default="strict",
    )
    run.add_argument("--goal-images", type=json.loads, help="目标图像的相机名到本地路径映射")
    run.add_argument(
        "--intervention",
        nargs="?",
        const="both",
        choices=("left", "right", "both"),
        help="启用相对接管，可指定左臂或右臂",
    )
    run.add_argument("--left-port", default=CollectionConfig.left_port)
    run.add_argument("--right-port", default=CollectionConfig.right_port)
    run.add_argument("--leader-id")
    run.add_argument("--calibration-dir")
    run.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="给已封存的回合补交评分")
    score.add_argument("--episode", required=True)
    score.add_argument("--reward", type=int, choices=(0, 1), required=True)
    for cmd in (run, score):
        cmd.add_argument("--output-dir", required=True)
    dagger = commands.add_parser("dagger", help="策略执行期间按需接管并保存人工纠错片段")
    add_policy_arguments(dagger, dataset_required=True)
    add_dagger_arguments(dagger)
    rlt = commands.add_parser("rlt", help="采集包含基座参考动作和阶段标签的评分回合")
    _add_rlt_arguments(rlt)
    parsed = parser.parse_args(argv)
    if parsed.command == "rlt":
        from alohamini.apps.collection.rlt import run as collect_rlt
        from alohamini.rl.rlt.config import RLTConfig

        overrides = RLTConfig.overrides(vars(parsed))
        if not parsed.config and not {"checkpoint", "task", "output_dir"} <= overrides.keys():
            rlt.error("采集需要 --policy.path、--task 和 --output-dir，或 --config")
        collect_rlt(RLTConfig.read(parsed.config, **overrides), resume=parsed.resume)
        return 0
    args = vars(parsed)
    if parsed.command == "dagger":
        config = pop_dagger_options(args)
        return run_policy(args, dagger=config)
    command = args.pop("command")
    root = Path(args["output_dir"]).expanduser().resolve()
    if command == "scored":
        from alohamini.apps.collection.native import worker
        from alohamini.apps.collection.scored_session import run as collect

        resume = args.pop("resume")
        args["output_dir"] = str(root)
        collect(CollectionConfig(**args), worker=worker, resume=resume)
    else:
        from alohamini.datasets.experience.storage import RunLock, commit_episode
        from alohamini.datasets.experience.v3 import export_v3
        from alohamini.datasets.v3_extensions import collection_layout

        if not re.fullmatch("[0-9a-f]{32}", args["episode"]):
            parser.error("--episode 需要 32 位十六进制回合 ID")
        with RunLock(root / "control.lock"):
            layout = collection_layout(root)
            commit_episode(
                root / "pending" / args["episode"], args["reward"], validate=layout.validate
            )
        export_v3(root, layout=layout)
    return 0
