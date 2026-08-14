#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of compute-flow update."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brick_manager.cache import load_cache, write_cache
from brick_manager.errors import BrickManagerError
from brick_manager.identity import brick_manager_name
from brick_manager.metadata import load_info_json
from brick_manager.paths import StatePaths
from brick_manager.registry import (
    default_registry_path,
    parse_registry,
    resolve_local_source,
)


@dataclass(frozen=True)
class UpdateResult:
    registry_path: Path
    cache_path: Path
    brick_versions: int


def update_cache(state_paths: StatePaths, registry: str | None = None) -> UpdateResult:
    registry_path = (
        Path(registry).expanduser().resolve() if registry else default_registry_path()
    )

    previous_cache = _load_previous_cache(state_paths.cache)
    previous_bricks = previous_cache.get("bricks", {})
    next_cache: dict[str, object] = {"bricks": {}}

    for source in parse_registry(registry_path):
        source_path = resolve_local_source(registry_path, source)
        if not source_path.is_dir():
            raise BrickManagerError(f"Registry source is not a directory: {source}")

        info_json = load_info_json(source_path)
        manager_name = brick_manager_name(info_json["name"])
        key = f"{info_json['type']}:{manager_name}"
        version = info_json["version"]
        versions = next_cache["bricks"].setdefault(key, {})
        if version in versions:
            raise BrickManagerError(f"Duplicate brick identity: {key}@{version}")

        old_metadata = previous_bricks.get(key, {}).get(version, {}).get("metadata", {})
        installed = bool(old_metadata.get("installed", False))
        installed_source_path = (
            old_metadata.get("installed_source_path") if installed else None
        )
        installed_models = old_metadata.get("installed_models", {})

        versions[version] = {
            "info_json": info_json,
            "metadata": {
                "source": source,
                "source_type": "local_directory",
                "source_path": str(source_path),
                "manager_name": manager_name,
                "installed": installed,
                "installed_source_path": installed_source_path,
                "installed_models": installed_models,
            },
        }

    write_cache(state_paths.cache, next_cache)
    return UpdateResult(
        registry_path=registry_path,
        cache_path=state_paths.cache,
        brick_versions=sum(len(versions) for versions in next_cache["bricks"].values()),
    )


def _load_previous_cache(cache_path: Path) -> dict[str, object]:
    try:
        return load_cache(cache_path)
    except FileNotFoundError:
        return {"bricks": {}}
