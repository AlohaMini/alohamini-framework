"""Map public training names to a selected policy's actual options."""

import math
from dataclasses import fields

from alohamini.policies.registry import algorithm

MODEL_PARAMETERS = {
    "chunk_size": "chunk_size",
    "backbone": "vision_backbone",
    "nheads": "n_heads",
    "enc_layers": "n_encoder_layers",
    "dec_layers": "n_decoder_layers",
    "kl_weight": "kl_weight",
}


def _validate(name, value):
    if name in ("lr", "lr_backbone", "kl_weight"):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and nonnegative")
    elif name == "backbone":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("--backbone requires a model name or path")
    elif type(value) is not int or value < 1:
        raise ValueError(f"--{name} must be a positive integer")


def _chunk_length(component, options, future):
    """Convert predicted action rows to the policy's native window units."""
    history_offset = getattr(component, "chunk_history_offset", None)
    if history_offset is not None:
        history = options.get("n_obs_steps", component.config_class.n_obs_steps)
        if type(history) is not int or history < 1:
            raise ValueError("n_obs_steps must be a positive integer")
        return future + history + history_offset
    unit = getattr(component, "actions_per_chunk_unit", None)
    if unit:
        per_unit = options.get(unit, getattr(component.config_class, unit))
        if type(per_unit) is not int or per_unit < 1 or future % per_unit:
            raise ValueError(f"--chunk_size must be divisible by {unit}")
        return future // per_unit
    return future


def resolve_common_parameters(cfg, kind, *, base_model=None, cli_keys=()):
    """Resolve explicit overrides before applying presets or inheriting native model weights."""
    names = [key for key in (*MODEL_PARAMETERS, "lr", "lr_backbone") if key in cfg]
    if not names:
        return cfg
    component = algorithm(kind)
    model = cfg.setdefault("model", {})
    options = {**(base_model or {}), **model}
    native_fields = {f.name for f in fields(component.config_class)}
    aliases = getattr(component, "training_parameter_fields", {})
    mappings = {**MODEL_PARAMETERS, **(aliases(options) if callable(aliases) else aliases)}
    assigned = {}
    # Apply command-line aliases after file aliases, including tied native fields.
    for name in sorted(names, key=lambda key: key in cli_keys):
        setting = cfg.pop(name)
        _validate(name, setting)
        if name in ("lr", "lr_backbone"):
            target = cfg.setdefault("optimizer", {})
            field = name
            flag = f"optimizer.{field}"
            if name == "lr_backbone" and not getattr(component, "visual_backbone_modules", ()):
                raise ValueError(f"{kind} does not declare a visual backbone for --lr_backbone")
        else:
            path = mappings[name].split(".")
            if path[0] not in native_fields:
                raise ValueError(f"{kind} does not expose --{name}; see the policy configuration")
            field = path[-1]
            flag = f"policy.{path[0]}"
            target = model
            for part in path[:-1]:
                target = target.setdefault(part, {})
            if name == "chunk_size":
                setting = _chunk_length(component, options, setting)
        # Some models pair encoder and decoder blocks at each depth.
        target_key = (flag, field)
        if target_key in assigned:
            previous_name, previous_setting = assigned[target_key]
            if previous_setting != setting and (previous_name in cli_keys) == (name in cli_keys):
                raise ValueError(f"{kind} ties the requested layer counts to --{flag}")
        if field in target and target[field] != setting:
            if flag in cli_keys and name not in cli_keys:
                continue
            if name not in cli_keys or flag in cli_keys:
                raise ValueError(f"Conflicting --{name} and --{flag} in {kind} configuration")
        target[field] = setting
        assigned[target_key] = (name, setting)
    return cfg
