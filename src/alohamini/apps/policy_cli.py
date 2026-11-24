"""Policy loading and execution options shared by evaluation and correction collection."""

import argparse
import json
import logging

from alohamini.model import robot_models


def add_policy_arguments(parser, *, dataset_required=False):
    from alohamini.cli import _ensemble_coefficient

    parser.add_argument("--host", "--robot.remote_ip", dest="host", required=True)
    parser.add_argument(
        "--robot_model",
        "--robot.robot_model",
        dest="robot_model",
        required=True,
        choices=robot_models(),
    )
    policy_source = parser.add_mutually_exclusive_group(required=True)
    policy_source.add_argument(
        "--policy",
        dest="policy_factory",
        help="可导入的 module:factory，工厂函数返回策略对象",
    )
    policy_source.add_argument(
        "--policy.path", dest="checkpoint", help="本地 AlohaMini 策略 checkpoint"
    )
    parser.add_argument("--device", help="checkpoint 推理设备，默认 cuda")
    parser.add_argument(
        "--language-replan-interval-s",
        dest="language_replan_interval_s",
        type=float,
        default=argparse.SUPPRESS,
        help="支持语言规划的策略按观测时间周期更新子任务",
    )
    parser.add_argument(
        "--goal-images",
        dest="goal_images",
        type=json.loads,
        default=argparse.SUPPRESS,
        help='目标图像映射，例如 {"forward":"/path/goal.png"}；仅用于支持视觉目标的策略',
    )
    from alohamini.trajectory_execution import add_inference_arguments

    add_inference_arguments(parser)
    parser.add_argument(
        "--policy.time_scale", dest="time_scale", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--policy.n_action_steps",
        dest="n_action_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="执行多少个动作后重新预测，省略时沿用 checkpoint",
    )
    parser.add_argument(
        "--policy.temporal_ensemble_coeff",
        dest="temporal_ensemble_coeff",
        type=_ensemble_coefficient,
        default=argparse.SUPPRESS,
        help="ACT 融合系数；none 关闭，0 等权融合，省略时沿用 checkpoint",
    )
    parser.add_argument(
        "--fixed-dimensions",
        nargs="+",
        default=argparse.SUPPRESS,
        help="固定维度或 arm_left/arm_right/lift_axis/base；新增维度使用训练动作均值",
    )
    parser.add_argument(
        "--fixed-current",
        nargs="+",
        default=argparse.SUPPRESS,
        help=(
            "固定所选维度在推理启动时的实测位置，无需训练数据集；"
            "可选 arm_left/arm_right/lift_axis/base"
        ),
    )
    parser.add_argument(
        "--fixed-dataset",
        default=argparse.SUPPRESS,
        help="旧 checkpoint 计算固定均值所用的原训练数据集",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--calibration-mode",
        choices=("strict", "normalized", "legacy_normalized"),
        default=argparse.SUPPRESS,
        help="strict 要求训练标定一致；normalized 迁移已知标定；legacy_normalized 声明旧数据为"
        "关节[-100,100]/夹爪[0,100]、底盘m/s及deg/s、升降mm，并绑定当前Host标定",
    )
    parser.add_argument("--episode_time", dest="episode_time_s", type=float, default=60)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--reset_time", dest="reset_time_s", type=float, default=10)
    parser.add_argument(
        "--dataset", dest="dataset_name", required=dataset_required, help="本地数据集名称"
    )
    parser.add_argument("--task", default="robot task")


def run_policy(options, *, dagger=None):
    from alohamini.apps.evaluation import evaluate

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    options.pop("command")
    if dagger is not None:
        options["dagger"] = dagger
    checkpoint = options.pop("checkpoint")
    device = options.pop("device")
    from alohamini.trajectory_execution import pop_inference_options

    execution = pop_inference_options(options)
    overrides = {
        key: options.pop(key)
        for key in (
            "n_action_steps",
            "temporal_ensemble_coeff",
            "fixed_dimensions",
            "fixed_dataset",
            "fixed_current",
            "calibration_mode",
            "goal_images",
            "time_scale",
            "language_replan_interval_s",
        )
        if key in options
    }
    if checkpoint is not None:
        from pathlib import Path

        if not (Path(checkpoint).expanduser() / "policy.json").is_file():
            raise ValueError(
                "Expected an AlohaMini checkpoint; use the LeRobot fork for LeRobot checkpoints"
            )

        def load_checkpoint():
            from alohamini.learning.policy import NativePolicy

            policy = NativePolicy(
                checkpoint,
                device=device or "cuda",
                task=options["task"],
                **(
                    {"robot_model": options["robot_model"]}
                    if overrides.get("calibration_mode") == "legacy_normalized"
                    else {}
                ),
                **overrides,
            )
            from alohamini.learning.trajectory_execution import prepare_execution

            return prepare_execution(policy, execution, control_fps=options["fps"])

        options["policy_factory"] = load_checkpoint
    elif device is not None or overrides or execution != type(execution)():
        raise ValueError("Checkpoint options require --policy.path, not --policy")
    evaluate(**options)
    return 0
