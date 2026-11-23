"""Serializable results for the resource checks performed by training adapters."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ResourceCheck:
    name: str
    kind: str
    path: str | Path | None
    validate: Callable | None = None
    resolve: Callable | None = None
    status: str = "ready"
    reason: str | None = None
    depends_on: tuple = ()
    validation: str = "adapter"

    def __post_init__(self):
        if self.status not in {"ready", "blocked", "deferred", "unverified", "not_required"}:
            raise ValueError(f"Unsupported resource check status: {self.status}")
        if self.status == "ready" and self.validate is None:
            raise ValueError(f"Resource check has no validator: {self.name}")

    def resolved(self, value):
        return replace(self, resolve=lambda _: value)


def error_details(error):
    return {"type": type(error).__name__, "message": str(error)}


def resource_paths(value):
    from alohamini.learning.weights import ResolvedWeights

    if isinstance(value, ResolvedWeights):
        return [str(path) for path, _ in value.files]
    if isinstance(value, (str, Path)) and str(value).strip():
        return [str(Path(value).expanduser().resolve())]
    return []


class ResourceChecks:
    def __init__(self, policy, *, strict):
        self.policy = policy if isinstance(policy, str) else None
        self.strict = strict
        self.resources = []
        self.errors = []
        self.complete = True

    def run(self, check):
        paths = (
            [str(check.path)]
            if check.status == "unverified" and check.path
            else resource_paths(check.path)
        )
        row = {
            "name": check.name,
            "kind": check.kind,
            "requested_path": paths[0] if paths else None,
            "paths": paths,
            "status": check.status,
            "available": None,
            "reason": check.reason,
            "error": None,
            "depends_on": list(check.depends_on),
            "validation": check.validation,
        }
        self.resources.append(row)
        blocked = [
            name
            for name in check.depends_on
            if not any(
                item["name"] == name and item["available"] is True for item in self.resources
            )
        ]
        if blocked:
            row.update(status="blocked", reason=f"Requires ready resources: {', '.join(blocked)}")
            self.complete = False
            if self.strict:
                raise ValueError(f"Resource {check.name}: {row['reason']}")
            return None
        if check.status != "ready":
            if check.status in {"unverified", "deferred", "blocked"}:
                self.complete = False
            return None
        try:
            selected = check.resolve(check.path) if check.resolve else check.path
            row["paths"] = resource_paths(selected)
            value = check.validate(selected)
            row["available"] = True
            return value
        except Exception as error:
            row.update(status="failed", available=False, error=error_details(error))
            if self.strict:
                raise
            return None

    def fail(self, stage, error):
        self.complete = False
        self.errors.append({"stage": stage, **error_details(error)})
        if self.strict:
            raise error

    def report(self, settings=None):
        failed = bool(self.errors) or any(row["status"] == "failed" for row in self.resources)
        return {
            "schema_version": 1,
            "policy": self.policy,
            "ok": not failed,
            "complete": self.complete,
            "resources": deepcopy(self.resources),
            "errors": deepcopy(self.errors),
            "resolved_settings": deepcopy(settings) if not failed else None,
        }
