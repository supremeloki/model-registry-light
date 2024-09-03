from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class RegistryError(Exception):
    pass


class ModelNotFoundError(RegistryError):
    def __init__(self, name: str, version: str | None = None) -> None:
        target = f"{name}@{version}" if version else name
        super().__init__(f"model not found: {target!r}")


class InvalidModelNameError(RegistryError):
    pass


class VersionConflictError(RegistryError):
    pass


def _validate_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise InvalidModelNameError(
            f"name must match [a-z0-9][a-z0-9._-]{{1,63}}, got {name!r}"
        )


@dataclass(frozen=True)
class ModelVersion:
    model: str
    version: str
    created_at: str
    artifact_path: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: frozenset[str] = frozenset()
    deprecated: bool = False

    @property
    def is_current(self) -> bool:
        return not self.deprecated


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    versions: tuple[ModelVersion, ...] = ()

    def latest(self) -> ModelVersion | None:
        return self.versions[-1] if self.versions else None


class ModelRegistry:
    def __init__(self, storage_path: Path | None = None) -> None:
        self._models: dict[str, list[ModelVersion]] = {}
        self._storage_path = storage_path
        if storage_path and storage_path.exists():
            self._load(storage_path)

    def register(
        self,
        name: str,
        version: str,
        artifact_path: str,
        params: dict[str, Any] | None = None,
        tags: frozenset[str] = frozenset(),
    ) -> ModelVersion:
        _validate_name(name)
        history = self._models.setdefault(name, [])
        if any(v.version == version for v in history):
            raise VersionConflictError(f"{name}@{version} already registered")
        entry = ModelVersion(
            model=name,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            artifact_path=artifact_path,
            params=dict(params or {}),
            tags=tags,
        )
        history.append(entry)
        self._flush()
        return entry

    def get(self, name: str, version: str | None = None) -> ModelVersion:
        history = self._models.get(name)
        if not history:
            raise ModelNotFoundError(name)
        if version is None or version == "latest":
            return history[-1]
        for candidate in history:
            if candidate.version == version:
                return candidate
        raise ModelNotFoundError(name, version)

    def list_models(self) -> tuple[RegisteredModel, ...]:
        return tuple(
            RegisteredModel(name=name, versions=tuple(history))
            for name, history in sorted(self._models.items())
        )

    def find_by_tag(self, tag: str) -> tuple[ModelVersion, ...]:
        matches = [
            version
            for history in self._models.values()
            for version in history
            if tag in version.tags
        ]
        return tuple(sorted(matches, key=lambda v: (v.model, v.created_at)))

    def deprecate(self, name: str, version: str) -> ModelVersion:
        target = self.get(name, version)
        updated = replace(target, deprecated=True)
        history = self._models[name]
        index = next(i for i, v in enumerate(history) if v.version == version)
        history[index] = updated
        self._flush()
        return updated

    def delete_version(self, name: str, version: str) -> bool:
        history = self._models.get(name)
        if not history:
            raise ModelNotFoundError(name)
        remaining = [v for v in history if v.version != version]
        if len(remaining) == len(history):
            raise ModelNotFoundError(name, version)
        if remaining:
            self._models[name] = remaining
        else:
            del self._models[name]
        self._flush()
        return True

    def _flush(self) -> None:
        if not self._storage_path:
            return
        payload = {
            name: [entry.__dict__ | {"tags": sorted(entry.tags)} for entry in history]
            for name, history in self._models.items()
        }
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"corrupt registry file: {exc}") from exc
        for name, entries in payload.items():
            self._models[name] = [
                ModelVersion(**{**entry, "tags": frozenset(entry["tags"])})
                for entry in entries
            ]
