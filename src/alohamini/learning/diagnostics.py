"""Optional measurements at optimizer and evaluation boundaries."""

import math
from collections import defaultdict

import torch
from torch.utils.data import Subset


def trainability(model):
    """Count unique parameters by flag and collapse uniformly frozen/trainable branches."""
    counts = defaultdict(lambda: [0, 0])
    children = defaultdict(dict)
    for name, param in model.named_parameters():
        path = ""
        index = int(param.requires_grad)
        size = param.numel()
        counts[path][index] += size
        for part in name.split("."):
            child = f"{path}.{part}" if path else part
            children[path][child] = None
            counts[child][index] += size
            path = child
    modules = []

    def visit(path):
        frozen, trainable = counts[path]
        if path and not (frozen and trainable):
            modules.append({"name": path, "frozen": frozen, "trainable": trainable})
        else:
            for child in children[path]:
                visit(child)

    visit("")
    frozen, trainable = counts[""]
    return {"trainable": trainable, "frozen": frozen, "modules": modules}


def diagnostic_options(cfg):
    options = cfg.get("diagnostics", {})
    if not isinstance(options, dict):
        raise ValueError("diagnostics must be an object")
    allowed = {"update_every", "modules", "train_eval_samples"}
    if options.keys() - allowed:
        raise ValueError(f"Unknown diagnostics options: {sorted(options.keys() - allowed)}")
    every = options.get("update_every", 0)
    count = options.get("train_eval_samples", 0)
    if any(type(v) is not int or v < 0 for v in (every, count)):
        raise ValueError("Diagnostic intervals and sample counts must be nonnegative integers")
    modules = options.get("modules", [])
    if (
        not isinstance(modules, list)
        or not all(isinstance(s, str) and s and "/" not in s for s in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("diagnostics.modules must contain distinct named module paths")
    if bool(every) != bool(modules):
        raise ValueError("Set both diagnostics.update_every and diagnostics.modules")
    if every and cfg.get("distributed_backend") == "fsdp2":
        raise ValueError("Full-module optimizer probes support single-device and replicated DDP")
    if count and not cfg.get("eval_steps"):
        raise ValueError("diagnostics.train_eval_samples requires periodic validation")
    return every, modules, count


class FixedTrainSubset(Subset):
    """Fixed bin midpoints in training sample order, independent of sampler or training RNG."""

    def __init__(self, dataset, count):
        count = min(count, len(dataset))
        indices = [(2 * i + 1) * len(dataset) // (2 * count) for i in range(count)]
        super().__init__(dataset, indices)

    def action_metadata(self, i):
        return self.dataset.action_metadata(self.indices[i])


class ModuleUpdates:
    def __init__(self, model, names):
        self.groups = {}
        modules = dict(model.named_modules())
        for name in names:
            if name not in modules:
                raise ValueError(f"Unknown diagnostic module: {name}")
            params = [p for p in modules[name].parameters() if p.requires_grad]
            if not params:
                raise ValueError(f"Diagnostic module has no trainable parameters: {name}")
            self.groups[name] = params

        # Overlapping named modules share one saved copy of each parameter.
        self.parameters = tuple(dict.fromkeys(p for params in self.groups.values() for p in params))

    @torch.no_grad()
    def capture(self, scale):
        before = {p: p.detach().clone() for p in self.parameters}
        grads = {}
        for name, params in self.groups.items():
            present = [p.grad for p in params if p.grad is not None]
            grads[name] = (_norm(present) / scale if present else None, len(present))
        return before, grads

    @torch.no_grad()
    def measure(self, capture):
        before, grads = capture
        packed, metrics = {}, {}
        for name, params in self.groups.items():
            prefix = f"module/{name}/"
            grad, connected = grads[name]
            packed[prefix + "weight_norm"] = _norm(before[p] for p in params)
            # Subtract in at least FP32 to retain representable storage changes.
            packed[prefix + "update_norm"] = _norm(p.to(_dtype(p)) - before[p] for p in params)
            if grad is not None:
                packed[prefix + "grad_norm"] = grad
            else:
                metrics[prefix + "grad_norm"] = None
            metrics[prefix + "grad_tensors"] = connected
            metrics[prefix + "trainable_tensors"] = len(params)
        # A single device-to-host transfer for all selected modules.
        for key, value in zip(packed, torch.stack(list(packed.values())).tolist(), strict=True):
            metrics[key] = value if math.isfinite(value) else None
        for name in self.groups:
            prefix = f"module/{name}/"
            weight, delta = metrics[prefix + "weight_norm"], metrics[prefix + "update_norm"]
            ratio = delta / weight if weight and delta is not None else math.nan
            metrics[prefix + "relative_update"] = ratio if math.isfinite(ratio) else None
        return metrics


def _dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _norm(tensors):
    norms = [torch.linalg.vector_norm(t, dtype=_dtype(t)) for t in tensors]
    return torch.linalg.vector_norm(torch.stack(norms))
