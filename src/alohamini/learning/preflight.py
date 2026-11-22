"""供桌面端在启动训练前读取算法、设备和本地模型文件。"""

import json
from pathlib import Path

from alohamini.policies.registry import ALGORITHMS, algorithm


_LABELS = {
    "act": "ACT",
    "am_act": "AM-ACT",
    "pi05": "π0.5",
    "pi0": "π0",
    "pi0_fast": "π0-FAST",
    "smolvla": "SmolVLA",
    "diffusion": "Diffusion Policy",
}
_PI = {"pi0", "pi05", "pi0_fast"}
# 这些适配器从 --policy.path 读取本地基础权重。
_REQUIRED_BASE = _PI | {
    "acdit", "cosmos3_nano", "cosmos3_edge", "dm05", "eo1", "evo1", "fastwam", "fineart_vla",
    "g05", "lingbot_va", "lingbot_vla_v2", "longwam", "smolvla", "spirit",
    "wall_oss_05", "xiaomi_robotics1", "xvla",
}
_SHARED_WEIGHTS = _REQUIRED_BASE - {
    "acdit", "fastwam", "lingbot_va", "lingbot_vla_v2", "smolvla", "spirit", "xiaomi_robotics1",
}
_NATIVE_ONLY = {
    "acdit", "act", "am_act", "bspline", "cdp", "diffusion", "dit_policy",
    "multi_task_dit", "vqbet",
}


def training_catalog():
    return [
        {"id": name, "label": _LABELS.get(name, name.replace("_", " ").upper())}
        for name in sorted(ALGORITHMS, key=lambda name: (name != "act", name))
    ]


def _check_safetensors(files):
    from safetensors import safe_open

    for filename, expected in files:
        with safe_open(filename, framework="numpy") as stream:
            keys = stream.keys()
            if not keys or (expected is not None and set(keys) != expected):
                raise ValueError(f"权重索引与文件内容不一致：{filename}")


def inspect_training(policy, device, pretrained_path, assets):
    from safetensors import SafetensorError

    from alohamini.learning.checkpoint import _asset_fields, model_directory, resolve_pretrained
    from alohamini.learning.execution import validate_local_workers
    from alohamini.learning.weights import _files
    from alohamini.paths import WorkspacePaths

    component = algorithm(policy)
    fields = _asset_fields(component)
    issues = []

    def issue(code, detail=""):
        issues.append({"code": code, "detail": str(detail)})

    try:
        validate_local_workers(1, device)
    except (ValueError, RuntimeError) as error:
        issue("device", error)

    model = {name: assets[name] for name in fields if name in assets}
    settings = {"policy": policy, "device": device, "model": model}
    root = model_directory(pretrained_path) if pretrained_path else None
    native = root is not None and (root / "policy.json").is_file()
    if root is not None:
        settings["pretrained_path"] = str(root)
        try:
            if not root.exists():
                raise FileNotFoundError(root)
            if native:
                settings = resolve_pretrained(settings)
                _check_safetensors([(root / "model.safetensors", None)])
            elif policy in _NATIVE_ONLY:
                issue("native_checkpoint", root)
            elif policy in _SHARED_WEIGHTS:
                _check_safetensors(_files(root))
            elif policy == "lingbot_va":
                _check_safetensors(
                    _files(root / "transformer", "diffusion_pytorch_model.safetensors")
                )
            elif policy in ("fastwam", "lingbot_vla_v2", "smolvla", "spirit"):
                weights = (
                    root if policy == "spirit" and root.is_file() else root / "model.safetensors"
                )
                _check_safetensors([(weights, None)])
                if policy == "smolvla":
                    config = json.loads((root / "config.json").read_text())
                    if config.get("type") != "smolvla" or config.get("use_peft", False):
                        raise ValueError("请提供完整的 SmolVLA 基础模型")
            elif policy in ("lewm", "xiaomi_robotics1"):
                filename = "weights.pt" if policy == "lewm" else "model_states.pt"
                weights = root / filename if root.is_dir() or policy == "lewm" else root
                if not weights.is_file():
                    raise FileNotFoundError(weights)
        except (OSError, ValueError, KeyError, TypeError, SafetensorError) as error:
            issue("weights", error)
    elif policy in _REQUIRED_BASE:
        issue("missing_weights")

    if policy in _PI:
        from alohamini.policies.pi05.tokenizer import PaligemmaTokenizer

        tokenizer = settings["model"].get("tokenizer_path")
        if not tokenizer:
            issue("missing_tokenizer")
        else:
            try:
                PaligemmaTokenizer(tokenizer)
            except (OSError, RuntimeError, ValueError) as error:
                issue("tokenizer", error)
        if policy == "pi0_fast" or settings["model"].get("training_recipe") == "ki":
            from alohamini.policies.pi05.fast_tokenizer import fast_asset_files

            fast_tokenizer = settings["model"].get("fast_tokenizer_path")
            if not fast_tokenizer:
                issue("missing_fast_tokenizer")
            else:
                try:
                    fast_asset_files(fast_tokenizer)
                except OSError as error:
                    issue("fast_tokenizer", error)

    if not issues:
        try:
            options = component.options(component.apply_preset(settings), device)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            issue("options", error)

        if not issues and not native:
            try:
                if policy == "gr00t_n17" and options.get("initialize_from_base", True):
                    _check_safetensors(_files(root or WorkspacePaths().pretrained / policy))
                if policy == "hyvla05":
                    source = root or Path(options["assets_path"])
                    _check_safetensors([(source / "model.safetensors", None)])
                if policy == "openwam":
                    source = root or WorkspacePaths().pretrained / policy
                    files = list(source.glob("checkpoint_step_*.safetensors"))
                    if len(files) != 1:
                        raise ValueError(f"仅能有一个 checkpoint_step_*.safetensors：{source}")
                    _check_safetensors([(files[0], None)])
                if policy == "vla_jepa":
                    if options.get("base_weights_mode", "backbones") == "policy":
                        if root is None:
                            raise ValueError("请填写 VLA-JEPA 策略权重的本地路径")
                        _check_safetensors(_files(root))
                    if options.get("base_weights_mode", "backbones") == "backbones":
                        directory = Path(options["assets_path"])
                        _check_safetensors(_files(directory / "qwen"))
                        if options.get("enable_world_model", True):
                            _check_safetensors(_files(directory / "vjepa2"))
            except (OSError, ValueError, SafetensorError) as error:
                issue("weights", error)

    return {"assets": list(fields), "issues": issues}
