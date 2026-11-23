"""给 Studio 提供训练算法目录和只读预检。"""

from alohamini.policies.registry import ALGORITHMS, algorithm

_DISPLAY_NAMES = {
    "act": "ACT",
    "am_act": "AM-ACT",
    "pi0": "π0",
    "pi05": "π0.5",
    "pi0_fast": "π0-FAST",
    "smolvla": "SmolVLA",
    "diffusion": "Diffusion Policy",
}


def training_catalog():
    return [
        {"id": name, "label": _DISPLAY_NAMES.get(name, name.replace("_", " ").upper())}
        for name in sorted(ALGORITHMS)
    ]


def inspect_training(policy, device, pretrained_path, assets):
    from safetensors import SafetensorError, safe_open

    from alohamini.learning.checkpoint import _asset_fields, model_directory, resolve_pretrained
    from alohamini.learning.execution import validate_local_workers

    component = algorithm(policy)
    fields = _asset_fields(component)
    issues = []
    resource_errors = (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError)

    def issue(code, detail=""):
        issues.append({"code": code, "detail": str(detail)})

    try:
        validate_local_workers(1, device)
    except (ValueError, RuntimeError) as error:
        issue("device", error)

    settings = {
        "policy": policy,
        "device": device,
        "model": {name: assets[name] for name in fields if name in assets},
    }
    if pretrained_path:
        settings["pretrained_path"] = pretrained_path
        try:
            root = model_directory(pretrained_path)
            if not root.exists():
                raise FileNotFoundError(root)
            if (root / "policy.json").is_file():
                settings = resolve_pretrained(settings)
                with safe_open(root / "model.safetensors", framework="numpy"):
                    pass
            elif check_weights := getattr(component, "inspect_base_weights", None):
                check_weights(root)
            elif root.is_file() and root.suffix == ".safetensors":
                with safe_open(root, framework="numpy"):
                    pass
        except resource_errors as error:
            issue("weights", error)

    options = settings["model"]
    if not any(item["code"] == "weights" for item in issues):
        try:
            settings = component.apply_preset(settings)
            options = settings.get("model", {})
            options = component.options(settings, device)
        except (*resource_errors, ImportError) as error:
            # 只归类训练入口已经报出的 CLI 参数缺项。
            detail = str(error)
            if not pretrained_path and "--policy.path" in detail:
                issue("missing_weights", detail)
            elif detail == "--policy.path requires an AlohaMini checkpoint model directory":
                issue("native_checkpoint", detail)
            else:
                issue("options", detail)

    if tokenizers := getattr(component, "training_tokenizers", None):
        for code, path, check in tokenizers(options):
            if not path:
                issue(f"missing_{code}")
                continue
            try:
                check(path)
            except resource_errors as error:
                issue(code, error)

    return {"assets": list(fields), "issues": issues}
