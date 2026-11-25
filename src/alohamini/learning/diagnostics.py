"""Optional measurements at optimizer and evaluation boundaries."""

import json
import logging
import math
import platform
import subprocess
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from torch.utils.data import Subset


def environment_fingerprint(cwd=None):
    """Capture lightweight runtime and git provenance without failing offline runs."""
    info = {
        "python": platform.python_version(),
        "git_commit": None,
        "git_dirty": None,
    }
    repo = Path(cwd) if cwd else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if commit.returncode == 0 and commit.stdout.strip():
            info["git_commit"] = commit.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            if dirty.returncode == 0:
                info["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


class BoundedJsonlWriter:
    """Bounded-memory background JSONL writer with snapshot serialization and flush barriers."""

    def __init__(self, path, *, max_records=4096, max_bytes=16 * 1024 * 1024):
        self.path = Path(path)
        self.max_records = max_records
        self.max_bytes = max_bytes
        self._stream = self.path.open("xb")
        self._cond = threading.Condition()
        self._queue = deque()
        self._queued_bytes = 0
        self._inflight = 0
        self._closed = False
        self._written = 0
        self._write_errors = 0
        self._last_error = None
        self._last_flush_s = 0.0
        self._thread = threading.Thread(
            target=self._run, name=f"jsonl-{self.path.name}", daemon=True
        )
        self._thread.start()

    def write(self, record):
        """Serialize on the caller thread to isolate mutable objects and reject NaN/Inf."""
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        size = len(payload)
        with self._cond:
            if self._closed:
                raise ValueError("Writer is closed")
            # Apply backpressure when the background writer falls behind memory budget.
            while (
                not self._closed
                and self._last_error is None
                and (
                    len(self._queue) >= self.max_records
                    or (self._queued_bytes + size > self.max_bytes and self._queue)
                )
            ):
                self._cond.wait(timeout=0.05)
            if self._last_error is not None:
                raise OSError(f"Failed writing {self.path.name}") from self._last_error
            self._queue.append(payload)
            self._queued_bytes += size
            self._cond.notify_all()

    def flush(self):
        with self._cond:
            while (self._queue or self._inflight) and self._last_error is None and not self._closed:
                self._cond.wait(timeout=0.05)
            if self._last_error is not None:
                raise OSError(f"Failed flushing {self.path.name}") from self._last_error

    def stats(self):
        with self._cond:
            return {
                "queued": len(self._queue),
                "queued_bytes": self._queued_bytes,
                "written": self._written,
                "write_errors": self._write_errors,
                "last_flush_s": self._last_flush_s,
            }

    def _run(self):
        logger = logging.getLogger(__name__)
        while True:
            batch = []
            batch_bytes = 0
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue and self._closed:
                    break
                while self._queue:
                    item = self._queue.popleft()
                    batch.append(item)
                    batch_bytes += len(item)
                    if batch_bytes >= 256 * 1024:
                        break
                self._queued_bytes -= batch_bytes
                self._inflight += len(batch)
                self._cond.notify_all()
            started = time.perf_counter()
            try:
                self._stream.write(b"".join(batch))
                self._stream.flush()
            except OSError as exc:
                logger.warning("Failed writing training log %s: %s", self.path.name, exc)
                with self._cond:
                    self._inflight -= len(batch)
                    self._write_errors += 1
                    self._last_error = exc
                    self._queue.clear()
                    self._queued_bytes = 0
                    self._cond.notify_all()
                break
            else:
                elapsed = time.perf_counter() - started
                with self._cond:
                    self._inflight -= len(batch)
                    self._written += len(batch)
                    self._last_flush_s = elapsed
                    self._cond.notify_all()

    def close(self):
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=10.0)
        try:
            self._stream.flush()
        finally:
            self._stream.close()
        if self._last_error is not None:
            raise OSError(f"Failed closing {self.path.name}") from self._last_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def gradient_alignment(losses, parameters):
    """Low-frequency diagnostic: per-term gradient norms and pairwise cosine similarity."""
    params = [p for p in parameters if p.requires_grad]
    names = list(losses)
    if len(names) < 2 or not params:
        return {}
    vectors = {}
    metrics = {}
    for index, name in enumerate(names):
        grads = torch.autograd.grad(
            losses[name],
            params,
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )
        present = [g.detach().float().reshape(-1) for g in grads if g is not None]
        if not present:
            metrics[f"grad_term/{name}_norm"] = 0.0
            continue
        flat = torch.cat(present)
        norm = torch.linalg.vector_norm(flat).item()
        metrics[f"grad_term/{name}_norm"] = norm if math.isfinite(norm) else None
        if math.isfinite(norm) and norm > 0:
            vectors[name] = (flat, norm)
    for i, a in enumerate(names):
        if a not in vectors:
            continue
        va, na = vectors[a]
        for b in names[i + 1 :]:
            if b not in vectors:
                continue
            vb, nb = vectors[b]
            if va.numel() == vb.numel():
                cos = (torch.dot(va, vb).item()) / (na * nb)
                metrics[f"grad_cos/{a}__{b}"] = cos if math.isfinite(cos) else None
    return metrics


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
