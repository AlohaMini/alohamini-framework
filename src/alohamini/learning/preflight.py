"""供桌面端在启动训练前读取算法、设备和本地模型文件。"""

from alohamini.policies.registry import algorithm
from alohamini.policies.registry import training_catalog as training_catalog


def inspect_training(policy, device, pretrained_path, assets):
    from safetensors import SafetensorError

    from alohamini.learning.checkpoint import _asset_fields, model_directory, resolve_pretrained
    from alohamini.learning.execution import validate_local_workers
    from alohamini.learning.weights import (
        DEFAULT_WEIGHTS_FILENAME,
        MissingTrainingWeights,
        TrainingWeightSource,
    )

    component = algorithm(policy)
    fields = _asset_fields(component)
    weight_sources = getattr(component, "training_weight_sources", None)
    tokenizer_sources = getattr(component, "training_tokenizers", None)
    issues = []
    resource_errors = (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError)

    def issue(code, detail=""):
        issues.append({"code": code, "detail": str(detail)})

    try:
        validate_local_workers(1, device)
    except (ValueError, RuntimeError) as error:
        issue("device", error)
    device_issues = len(issues)

    settings = {
        "policy": policy,
        "device": device,
        "model": {name: assets[name] for name in fields if name in assets},
    }
    native = False
    if pretrained_path:
        try:
            root = model_directory(pretrained_path)
            settings["pretrained_path"] = str(root)
            if not root.exists():
                raise FileNotFoundError(root)
            native = (root / "policy.json").is_file()
            if native:
                settings = resolve_pretrained(settings)
                TrainingWeightSource(root / DEFAULT_WEIGHTS_FILENAME).inspect()
            if not native and weight_sources is None:
                issue("native_checkpoint", root)
        except resource_errors as error:
            issue("weights", error)

    options = settings["model"]
    option_error = None
    if len(issues) == device_issues:
        try:
            settings = component.apply_preset(settings)
            options = settings.get("model", {})
            options = component.options(settings, device)
        except (*resource_errors, ImportError) as error:
            option_error = error

    if not native and weight_sources is not None and len(issues) == device_issues:
        try:
            for source in weight_sources(settings, options):
                source.inspect()
        except MissingTrainingWeights as error:
            issue("missing_weights", error)
        except resource_errors as error:
            issue("weights", error)

    if tokenizer_sources is not None:
        for code, path, check in tokenizer_sources(options):
            if not path:
                issue(f"missing_{code}")
                continue
            try:
                check(path)
            except resource_errors as error:
                issue(code, error)

    if option_error is not None and len(issues) == device_issues:
        code = "missing_weights" if isinstance(option_error, MissingTrainingWeights) else "options"
        issue(code, option_error)
    return {"assets": list(fields), "issues": issues}
