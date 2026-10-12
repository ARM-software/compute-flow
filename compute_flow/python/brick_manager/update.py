#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of compute-flow update."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brick_manager.cache import load_cache, write_cache
from brick_manager.catalog import select_versioned_entry
from brick_manager.errors import BrickManagerError
from brick_manager.git_source import inspect_git_source
from brick_manager.identity import BrickSelector, brick_manager_name
from brick_manager.metadata import load_info_json, load_installed_info_json
from brick_manager.paths import StatePaths
from brick_manager.registry import (
    default_registry_path,
    is_git_source,
    registry_entries,
    resolve_local_source,
)


@dataclass(frozen=True)
class BrickUpdate:
    installed_identity: str
    available_identity: str


@dataclass(frozen=True)
class UpdateResult:
    registry_path: Path
    cache_path: Path
    brick_versions: int
    available_updates: list[BrickUpdate] = field(default_factory=list)
    user_registry_path: Path | None = None


def update_cache(state_paths: StatePaths, registry: str | None = None) -> UpdateResult:
    registry_path = (
        Path(registry).expanduser().resolve() if registry else default_registry_path()
    )

    previous_cache = _load_previous_cache(state_paths.cache)
    previous_bricks = previous_cache.get("bricks", {})
    next_cache: dict[str, object] = {"bricks": {}}

    for source_registry, source in registry_entries(
        registry_path,
        state_paths.registry if registry is None else None,
    ):
        if is_git_source(source):
            source_versions = [
                (
                    git_version.info_json,
                    {
                        "source": source,
                        "source_type": "git",
                        "source_path": None,
                        "source_ref": git_version.tag,
                        "source_commit": git_version.commit,
                    },
                )
                for git_version in inspect_git_source(source)
            ]
        else:
            source_path = resolve_local_source(source_registry, source)
            if not source_path.is_dir():
                raise BrickManagerError(f"Registry source is not a directory: {source}")
            source_versions = [
                (
                    load_info_json(source_path),
                    {
                        "source": source,
                        "source_type": "local_directory",
                        "source_path": str(source_path),
                    },
                )
            ]

        for info_json, source_metadata in source_versions:
            manager_name = brick_manager_name(info_json["name"])
            key = f"{info_json['type']}:{manager_name}"
            version = info_json["version"]
            versions = next_cache["bricks"].setdefault(key, {})
            if version in versions:
                raise BrickManagerError(f"Duplicate brick identity: {key}@{version}")

            old_entry = previous_bricks.get(key, {}).get(version, {})
            old_metadata = old_entry.get("metadata", {})
            installed = bool(old_metadata.get("installed", False))
            installed_source_path = (
                old_metadata.get("installed_source_path") if installed else None
            )
            installed_info_json = old_metadata.get("installed_info_json")
            if (
                installed
                and installed_info_json is None
                and installed_source_path is not None
                and Path(installed_source_path).is_dir()
            ):
                selected = BrickSelector(info_json["type"], manager_name, version)
                installed_info_json = load_installed_info_json(
                    Path(installed_source_path), selected
                )

            versions[version] = {
                "info_json": info_json,
                "metadata": {
                    **source_metadata,
                    "manager_name": manager_name,
                    "installed": installed,
                    "installed_source_path": installed_source_path,
                    "installed_info_json": installed_info_json,
                    "installed_models": old_metadata.get("installed_models", {}),
                },
            }

    num_indexed_versions = sum(
        len(versions) for versions in next_cache["bricks"].values()
    )
    available_updates = _preserve_installed_entries(
        previous_bricks,
        next_cache["bricks"],
    )

    write_cache(state_paths.cache, next_cache)
    return UpdateResult(
        registry_path=registry_path,
        cache_path=state_paths.cache,
        brick_versions=num_indexed_versions,
        available_updates=available_updates,
        user_registry_path=(
            state_paths.registry
            if registry is None and state_paths.registry.is_file()
            else None
        ),
    )


def _load_previous_cache(cache_path: Path) -> dict[str, object]:
    try:
        return load_cache(cache_path)
    except FileNotFoundError:
        return {"bricks": {}}


def _preserve_installed_entries(
    previous_bricks: dict[str, Any],
    next_bricks: dict[str, Any],
) -> list[BrickUpdate]:
    available_updates = []
    for key, previous_versions in sorted(previous_bricks.items()):
        for installed_version, previous_entry in sorted(previous_versions.items()):
            previous_metadata = previous_entry["metadata"]
            if not previous_metadata.get("installed", False):
                continue

            _ensure_installed_snapshot(key, installed_version, previous_entry)
            indexed_versions = next_bricks.get(key, {})
            available_version = _newer_indexed_version(
                key,
                installed_version,
                indexed_versions,
            )
            indexed_installed_entry = indexed_versions.get(installed_version)
            if available_version is None and indexed_installed_entry is not None:
                installed_info_json = indexed_installed_entry["metadata"].get(
                    "installed_info_json"
                )
                if (
                    installed_info_json is not None
                    and indexed_installed_entry["info_json"] != installed_info_json
                ):
                    available_version = installed_version

            if indexed_installed_entry is None:
                next_bricks.setdefault(key, {})[installed_version] = previous_entry

            if available_version is not None:
                available_updates.append(
                    BrickUpdate(
                        installed_identity=f"{key}@{installed_version}",
                        available_identity=f"{key}@{available_version}",
                    )
                )

    return available_updates


def _ensure_installed_snapshot(
    key: str,
    version: str,
    entry: dict[str, Any],
) -> None:
    metadata = entry["metadata"]
    if metadata.get("installed_info_json") is not None:
        return

    installed_source_path = metadata.get("installed_source_path")
    if installed_source_path is None or not Path(installed_source_path).is_dir():
        return

    brick_type, name = key.split(":", maxsplit=1)
    selector = BrickSelector(brick_type, name, version)
    metadata["installed_info_json"] = load_installed_info_json(
        Path(installed_source_path),
        selector,
    )


def _newer_indexed_version(
    key: str,
    installed_version: str,
    indexed_versions: dict[str, Any],
) -> str | None:
    brick_type, name = key.split(":", maxsplit=1)
    candidates = set(indexed_versions)
    candidates.add(installed_version)
    candidate_cache = {
        "bricks": {
            key: {version: {"metadata": {"installed": False}} for version in candidates}
        }
    }
    latest_version, _ = select_versioned_entry(
        candidate_cache,
        BrickSelector(brick_type, name),
    )
    return latest_version if latest_version != installed_version else None
