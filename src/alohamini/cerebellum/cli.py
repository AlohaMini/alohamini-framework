"""Operator CLI for the Linux navigation workstation. No ROS import in the CLI process."""
import argparse
import ipaddress
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime
from uuid import uuid4


def parser():
    root = argparse.ArgumentParser(prog="alohamini navigate", description="LightNav 自由导航和底盘桥管理")
    actions = root.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("run", "启动、静检、等待 GO 并导航，退出时关闭会话"),
        ("follow", "持续跟随，模型停车后等待新航点，Ctrl-C 结束"),
        ("start", "启动服务并禁用底盘桥命令"),
        ("drive", "使用当前服务执行导航"),
        ("preflight", "只读检查当前服务和传感器"),
        ("status", "显示 ROS 节点、速度发布者和 GPU 使用情况"),
        ("estop", "锁存急停并禁用桥"),
        ("stop", "禁用桥并清理服务"),
        ("restart", "安全停止旧服务并重新启动，桥保持禁用"),
    ):
        sub = actions.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--ros-domain", type=int, choices=range(1, 233), metavar="1..232", default=181)
        if name in ("start", "restart", "run", "follow"):
            sub.add_argument("--instruction", required=name in ("run", "follow"), help="LightNav 自然语言指令")
            if name != "follow":
                arrival = sub.add_mutually_exclusive_group()
                arrival.add_argument("--color", choices=("purple", "blue", "green"), help="明确指定近场目标颜色")
                arrival.add_argument("--no-color-arrival", action="store_true", help="仅按导航指令运行，关闭颜色停靠")
            sub.add_argument("--pi-host", type=ipaddress.IPv4Address, default="192.168.2.2", help="树莓派 IPv4 地址")
        if name in ("start", "restart", "run", "follow", "drive", "preflight"):
            sub.add_argument("--config", type=Path, help="低速导航配置 JSON 的路径")
            if name != "follow":
                sub.add_argument("--chair", action="store_true", help="启用绕椅专项验收条件")
            sub.add_argument("--dry-run", action="store_true", help="静态观察，底盘桥保持禁用")
        if name in ("preflight", "drive", "run", "follow"):
            sub.add_argument("--output-dir", type=Path, help="新建的结果目录；run 下包含 preflight/ 和 drive/")
    return root


def launcher():
    # Editable checkouts use their own scripts. A wheel can operate a deployed checkout.
    candidates = (Path(__file__).resolve().parents[3], Path.cwd(),
                  Path.home() / "workspace/jjy/validation/m3_4-20261125")
    for checkout in candidates:
        path = checkout / "examples/navigation/m3_navigation.sh"
        if path.is_file():
            return path
    raise FileNotFoundError("请在包含 examples/navigation/m3_navigation.sh 的 5090 部署目录运行")


class Session:
    def __init__(self, script, env):
        self.script = script
        self.env = env
        self.child = None
        self.detached = False
        self.interrupted = 0

    def interrupt(self, signum, _frame):
        self.interrupted = signum
        self.forward_signal()

    def forward_signal(self):
        if self.child is not None and self.child.poll() is None:
            try:
                if self.detached:
                    os.killpg(self.child.pid, self.interrupted)
                else:
                    self.child.send_signal(self.interrupted)
            except ProcessLookupError:
                pass

    def call(self, *args, cleanup=False):
        if self.interrupted and not cleanup:
            raise InterruptedError()
        # SSH authentication may need /dev/tty at startup. The running monitor
        # and its shell receive group signals even for TERM sent only to this CLI.
        self.detached = args[0] in ("drive", "preflight")
        self.child = subprocess.Popen(["bash", str(self.script), *map(str, args)],
                                      env=self.env, cwd=self.script.parents[2],
                                      start_new_session=self.detached)
        try:
            if self.interrupted and not cleanup:
                self.forward_signal()
            rc = self.child.wait()
        finally:
            self.child = None
        if self.interrupted and not cleanup:
            raise InterruptedError()
        if rc:
            raise subprocess.CalledProcessError(rc, args)


def arrival_color(args):
    if args.no_color_arrival:
        if args.chair:
            raise ValueError("--chair requires a metric target; remove --no-color-arrival or --chair")
        print("[arrival] disabled by --no-color-arrival", flush=True)
        return None
    if args.color:
        print(f"[arrival] enabled color={args.color} from --color", flush=True)
        return args.color
    instruction = args.instruction or ""
    colors = [c for c, zh in (("purple", "紫色"), ("blue", "蓝色"), ("green", "绿色"))
              if zh in instruction or re.search(rf"\b{c}\b", instruction, re.IGNORECASE | re.ASCII)]
    if len(colors) == 1:
        print(f"[arrival] auto-enabled color={colors[0]} from instruction", flush=True)
        return colors[0]
    if args.chair:
        if colors:
            raise ValueError("Ambiguous chair target; select --color explicitly")
        print("[arrival] enabled color=purple from --chair preset", flush=True)
        return "purple"
    reason = "multiple supported colors; select --color" if colors else "no supported color in instruction"
    print(f"[arrival] disabled: {reason}", flush=True)
    return None


def execute(args, script):
    env = dict(os.environ, ROS_DOMAIN_ID=str(args.ros_domain), ROS_LOCALHOST_ONLY="1")
    for key in ("M3_STOP_TOKEN", "M3_SESSION_TOKEN"):
        env.pop(key, None)
    follow = args.action == "follow"
    if hasattr(args, "instruction"):
        color = None if follow else arrival_color(args)
        env.pop("M3_TARGET_ID", None)
        env.update(M3_INSTRUCTION=args.instruction or "", M3_COLOR=color or "", M3_PI_HOST=str(args.pi_host))
        env["M3_DRY_RUN"] = "true" if args.dry_run else "false"
    elif getattr(args, "dry_run", False):
        env["M3_DRY_RUN"] = "true"
    if getattr(args, "config", None):
        env["M3_CONFIG"] = str(args.config.expanduser().resolve())
    flags = ["--chair"] if getattr(args, "chair", False) else []
    if follow:
        from .state import load_config
        config = args.config or script.parents[2] / "evidence/m4-follow-shared.json"
        env["M3_CONFIG"] = str(config.expanduser().resolve())
        if not load_config(env["M3_CONFIG"])["control"]["continuous_follow"]:
            raise ValueError("follow requires control.continuous_follow=true")
        flags = ["--follow"]
    session = Session(script, env)
    output = getattr(args, "output_dir", None)
    if args.action in ("run", "follow", "drive", "preflight"):
        output = (output.expanduser().resolve() if output else script.parents[2] / "trials" /
                  (args.action + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    old = {s: signal.signal(s, session.interrupt) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        if args.action in ("run", "follow"):
            output.mkdir(parents=True, exist_ok=False)
            print(f"导航记录 {output}", flush=True)
            env["M3_SESSION_TOKEN"] = uuid4().hex
            try:
                session.call("start", *flags)
                session.call("preflight", output / "preflight", *flags)
                session.call("drive", output / "drive", *flags)
            finally:
                # Only release a bridge belonging to this invocation. A rejected
                # second invocation must never disable the first operator's bridge.
                env["M3_STOP_TOKEN"] = env["M3_SESSION_TOKEN"]
                for s in old:
                    signal.signal(s, signal.SIG_IGN)
                session.call("stop", cleanup=True)
        elif args.action in ("drive", "preflight"):
            session.call(args.action, output, *flags)
        else:
            session.call(args.action, *flags)
    except InterruptedError:
        return 128 + session.interrupted
    except subprocess.CalledProcessError as exc:
        return exc.returncode if exc.returncode > 0 else 128-exc.returncode
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if not sys.platform.startswith("linux"):
        print("请在 Linux ROS 2 工作站上运行导航服务。", file=sys.stderr)
        return 2
    try:
        return execute(args, launcher())
    except (OSError, ValueError) as exc:
        print(f"alohamini navigate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
