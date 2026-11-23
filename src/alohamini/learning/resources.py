"""Training-owned resource resolution and its serializable inspection interface."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from alohamini.learning.resource_checks import ResourceCheck, ResourceChecks
from alohamini.learning.weights import ResolvedWeights, inspect_weights, resolve_weights

_RESOLUTION_KEY = "_resolved_weight_sources"


def local_file(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("A local resource file must be specified")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Resource file is empty: {path}")
    return path


def local_tokenizer(value):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(value).expanduser().resolve()), local_files_only=True, trust_remote_code=False
    )


def local_processor(value, **kwargs):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(Path(value).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=False,
        **kwargs,
    )


def file_check(name, path, *, kind="file"):
    return ResourceCheck(name, kind, path, local_file, validation="nonempty_file")


def json_check(name, path):
    return ResourceCheck(
        name,
        "config",
        path,
        lambda value: json.loads(local_file(value).read_text()),
        validation="json_parse",
    )


def weight_check(
    name, path, *, filename="model.safetensors", single=False, directory_only=False, select=None
):
    def resolve(value):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("A local policy weight path is required")
        if directory_only and Path(value).expanduser().is_file():
            raise ValueError("This policy requires a checkpoint directory for --policy.path")
        return resolve_weights(select(value) if select else value, filename)

    def validate(selected):
        if single:
            selected.single_file()
        return inspect_weights(selected)

    return ResourceCheck(name, "weights", path, validate, resolve, validation="safetensors_headers")


def tokenizer_check(name, path):
    return ResourceCheck(name, "tokenizer", path, local_tokenizer, validation="tokenizer_load")


def processor_check(name, path, **kwargs):
    return ResourceCheck(
        name,
        "processor",
        path,
        lambda value: local_processor(value, **kwargs),
        validation="processor_load",
    )


def torchvision_backbone_checks(options, config_class):
    """Report the same TorchVision enum and cache location used by model construction."""
    weights = options.get("pretrained_backbone_weights", config_class.pretrained_backbone_weights)
    if weights is None:
        return
    from urllib.parse import urlparse

    from torch.hub import get_dir
    from torchvision.models import get_model_weights

    backbone = options.get("vision_backbone", config_class.vision_backbone)
    selected = get_model_weights(backbone).verify(weights)
    cached = Path(get_dir()) / "checkpoints" / Path(urlparse(selected.url).path).name
    if cached.is_file():
        yield file_check("vision_backbone", cached, kind="weights")
    else:
        yield ResourceCheck(
            "vision_backbone",
            "weights",
            cached,
            status="unverified",
            reason=f"Not cached; the TorchVision loader may download {selected.url}",
            validation="nonempty_file",
        )


class ResourceAlgorithm:
    """Adapters declare checks once for both training and external inspection."""

    allow_scratch = False
    tokenizer_fields = ()
    processor_fields = ()
    processor_kwargs = {}
    single_weight_file = False
    weights_directory_only = False

    def training_weight_sources(self, settings, options):
        path = settings.get("pretrained_path")
        if not path and self.allow_scratch:
            return {}
        return {
            "policy": weight_check(
                "policy",
                path,
                single=self.single_weight_file,
                directory_only=self.weights_directory_only,
            )
        }

    def training_asset_checks(self, settings, options):
        return [
            *(tokenizer_check(name, options.get(name)) for name in self.tokenizer_fields),
            *(
                processor_check(name, options.get(name), **self.processor_kwargs)
                for name in self.processor_fields
            ),
        ]


def training_weights(component, settings, model=None, *, name="policy"):
    """Use the files selected at preparation, including direct adapter calls."""
    prepared = getattr(component, "_prepared_weight_sources", None)
    if prepared is not None and settings.get("pretrained_path") == component._prepared_weight_path:
        return prepared[name]
    options = vars(model.config) if model is not None else settings.get("model", {})
    check = component.training_weight_sources(settings, options)[name]
    return ResourceChecks(settings.get("policy"), strict=True).run(check)


def _json_value(value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if is_dataclass(item):
            return asdict(item)
        raise TypeError(f"Training configuration is not JSON serializable: {type(item).__name__}")

    return json.loads(json.dumps(value, default=encode, allow_nan=False))


def _selection_signature(settings):
    selected = {key: settings.get(key) for key in ("policy", "pretrained_path", "model")}
    data = json.dumps(_json_value(selected), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _saved_sources(plan, settings):
    if not plan or plan.get("signature") != _selection_signature(settings):
        return None
    if plan.get("schema_version") != 1 or not isinstance(plan.get("sources"), dict):
        raise ValueError("Unsupported resolved training weights")
    result = {}
    for name, value in plan["sources"].items():
        if value.get("type") == "safetensors":
            result[name] = ResolvedWeights.from_dict(value["value"])
        elif value.get("type") == "file":
            result[name] = Path(value["value"])
        else:
            raise ValueError("Unsupported resolved weight source type")
    return result


def _source_plan(settings, sources):
    return {
        "schema_version": 1,
        "signature": _selection_signature(settings),
        "sources": {
            name: {"type": "safetensors", "value": value.to_dict()}
            if isinstance(value, ResolvedWeights)
            else {"type": "file", "value": str(value)}
            for name, value in sources.items()
        },
    }


@dataclass
class PreparedTraining:
    settings: dict
    component: object
    options: dict | None
    report: dict


def _blocked_resources(checks, cfg, component):
    checks.run(
        ResourceCheck(
            "policy",
            "weights",
            cfg.get("pretrained_path"),
            status="blocked",
            reason="Training options could not be resolved",
        )
    )
    for name in getattr(component, "asset_fields", ()):
        checks.run(
            ResourceCheck(
                name,
                "asset",
                cfg.get("model", {}).get(name),
                status="blocked",
                reason="Training options could not be resolved",
            )
        )


def _prepare_training(settings, *, strict):
    from alohamini.learning.checkpoint import model_directory, read_checkpoint, resolve_pretrained
    from alohamini.policies.registry import algorithm

    checks = ResourceChecks(settings.get("policy", "act"), strict=strict)
    cfg, component, options = deepcopy(settings), None, None
    plan = cfg.pop(_RESOLUTION_KEY, None)
    try:
        cfg = resolve_pretrained(cfg)
        if cfg.get("resume"):
            if not cfg.get("_checkpoint"):
                raise ValueError("Resume requires a saved training checkpoint")
            root, manifest = read_checkpoint(cfg["_checkpoint"])
            if cfg.get("policy", manifest["kind"]) != manifest["kind"]:
                raise ValueError("Resume checkpoint policy type differs")
            cfg["policy"] = manifest["kind"]
            component = algorithm(cfg["policy"])
            cfg = component.apply_preset(cfg)
            options = component.checkpoint_options(
                root, deepcopy(manifest["config"]), cfg.get("device", "cuda"), None, "checkpoint"
            )
        else:
            if cfg.get("pretrained_path"):
                cfg["pretrained_path"] = str(Path(cfg["pretrained_path"]).expanduser().resolve())
            component = algorithm(cfg.get("policy", "act"))
            cfg.setdefault("policy", "act")
            cfg.setdefault("model", {})
            cfg = component.apply_preset(cfg)
            options = component.options(cfg, cfg.get("device", "cuda"))
    except Exception as error:
        checks.fail("configuration", error)
        _blocked_resources(checks, cfg, component)
        return PreparedTraining(cfg, component, None, checks.report())

    checks.policy = cfg.get("policy", "act")
    sources = {}
    try:
        if cfg.get("resume"):
            checkpoint = Path(cfg["_checkpoint"]).expanduser().resolve()
            checks.run(file_check("training_state", checkpoint / "training_state/state.pt"))
            requests = {"policy": weight_check("policy", root)}
            saved = None
        else:
            source = model_directory(cfg["pretrained_path"]) if cfg.get("pretrained_path") else None
            native = source is not None and (source / "policy.json").is_file()
            requests = (
                {"policy": weight_check("policy", source)}
                if native
                else component.training_weight_sources(cfg, options)
            )
            saved = _saved_sources(plan, cfg)
        for name, request in requests.items():
            if saved is not None:
                if name not in saved:
                    raise ValueError(f"Resolved training weights are missing {name}")
                request = request.resolved(saved[name])
            result = checks.run(request)
            if result is not None:
                sources[name] = result
        if not requests:
            checks.run(
                ResourceCheck(
                    "policy",
                    "weights",
                    None,
                    status="not_required",
                    reason="No initialization weights are selected for this training mode",
                )
            )
    except Exception as error:
        checks.fail("weights", error)
    try:
        for request in component.training_asset_checks(cfg, options):
            checks.run(request)
    except Exception as error:
        checks.fail("assets", error)

    if not cfg.get("resume"):
        for field in getattr(component, "asset_fields", ()):
            if options.get(field):
                cfg.setdefault("model", {})[field] = str(options[field])
        selected = sources.get("policy")
        if isinstance(selected, ResolvedWeights) and getattr(
            component, "pin_policy_weights", False
        ):
            cfg["pretrained_path"] = str(selected.path)
        component._prepared_weight_sources = sources
        component._prepared_weight_path = cfg.get("pretrained_path")
        if checks.report()["ok"]:
            cfg[_RESOLUTION_KEY] = _source_plan(cfg, sources)
    report = checks.report(_json_value(cfg))
    return PreparedTraining(cfg, component, None if cfg.get("resume") else options, report)


def prepare_training(settings):
    """Resolve and validate the exact inputs consumed by this training start."""
    return _prepare_training(settings, strict=True)


def check_training_resources(settings):
    """Return JSON-compatible resource results without starting training or allocating a model."""
    if not isinstance(settings, dict):
        checks = ResourceChecks(None, strict=False)
        checks.fail("configuration", TypeError("Training settings must be a JSON object"))
        return checks.report()
    try:
        return _prepare_training(settings, strict=False).report
    except Exception as error:
        checks = ResourceChecks(settings.get("policy"), strict=False)
        checks.fail("configuration", error)
        return checks.report()
