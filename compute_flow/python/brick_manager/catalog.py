#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Cache-backed catalogue views."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brick_manager.cache import load_cache
from brick_manager.errors import BrickManagerError
from brick_manager.identity import (
    BrickSelector,
    parse_brick_selector,
    validate_brick_component,
    validate_model_component,
)
from brick_manager.metadata import load_installed_entry_info_json
from brick_manager.model_artifacts import artifact_manifest_sha256, model_artifact_path
from brick_manager.paths import StatePaths


_CACHE_MISSING_MESSAGE = "No registry cache found. Run `compute-flow update` first."


@dataclass(frozen=True)
class BrickSummary:
    brick: str
    version: str
    installed_version: str | None
    model_count: int
    description: str


@dataclass(frozen=True)
class BrickModelSummary:
    name: str
    installed: bool
    origin: str
    sha256: str | None
    description: str


@dataclass(frozen=True)
class BrickModelList:
    identity: str
    version: str
    models: list[BrickModelSummary]


@dataclass(frozen=True)
class BrickModelInfoItem:
    name: str
    version: str
    installed: bool
    location: str | None
    size_bytes: int | None
    manifests: list[str]
    origin: str
    template: str | None


@dataclass(frozen=True)
class BrickModelInfo:
    identity: str
    version: str
    models: list[BrickModelInfoItem]


@dataclass(frozen=True)
class BrickInfo:
    identity: str
    brick: str
    version: str
    info_json: dict[str, Any]
    metadata: dict[str, Any]


def get_brick_summaries(
    state_paths: StatePaths,
    brick_types: Sequence[str] = (),
    *,
    installed_only: bool = False,
) -> list[BrickSummary]:
    """Return cached brick summaries filtered by type and install state."""
    cache = load_required_cache(state_paths)
    selected_types = set(brick_types)
    for brick_type in selected_types:
        validate_brick_component("type", brick_type)

    summaries = []
    for brick, version, entry in _iter_entries(cache):
        brick_type, _, _ = brick.partition(":")
        if selected_types and brick_type not in selected_types:
            continue

        info_json = entry["info_json"]
        metadata = entry["metadata"]
        installed = bool(metadata.get("installed", False))
        if installed_only and not installed:
            continue
        summaries.append(
            BrickSummary(
                brick=brick,
                version=version,
                installed_version=version if installed else None,
                model_count=len(info_json.get("models", {})),
                description=info_json["description"],
            )
        )

    summaries.sort(
        key=lambda summary: (summary.brick, _version_sort_key(summary.version))
    )
    return summaries


def list_bricks(
    state_paths: StatePaths,
    brick_types: Sequence[str] = (),
    *,
    installed_only: bool = False,
) -> str:
    """Format cached brick summaries as a plain-text table."""
    rows = []
    for summary in get_brick_summaries(
        state_paths, brick_types, installed_only=installed_only
    ):
        rows.append(
            (
                summary.brick,
                summary.version,
                summary.installed_version or "-",
                summary.description,
            )
        )

    return _format_table(("Brick", "Version", "Installed", "Description"), rows)


def get_brick_model_list(state_paths: StatePaths, brick: str) -> BrickModelList:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    version, entry = select_versioned_entry(cache, selector)
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = entry["info_json"]
    if selector.version is None and entry["metadata"].get("installed", False):
        info_json = _installed_info_json(entry, selected)
    key = selector.key
    models = []
    for model_name, metadata in sorted(info_json.get("models", {}).items()):
        models.append(
            BrickModelSummary(
                name=model_name,
                installed=_model_installed(
                    state_paths,
                    selected,
                    entry,
                    model_name,
                    metadata["sha256"],
                ),
                origin="Declared",
                sha256=metadata["sha256"],
                description=metadata["description"],
            )
        )

    if selector.version is None and entry["metadata"].get("installed", False):
        summaries = {model.name: model for model in models}
        for model_name, state in sorted(
            entry["metadata"].get("installed_models", {}).items()
        ):
            if state.get("origin") != "imported":
                continue
            path = model_artifact_path(state_paths, selected, model_name)
            summaries[model_name] = BrickModelSummary(
                name=model_name,
                installed=_imported_model_installed(path, state),
                origin="Imported",
                sha256=None,
                description="-",
            )
        models = [summaries[name] for name in sorted(summaries)]

    return BrickModelList(identity=f"{key}@{version}", version=version, models=models)


def get_brick_model_info(
    state_paths: StatePaths,
    brick: str,
    model_names: list[str],
) -> BrickModelInfo:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model info takes <type>:<name> without @version because models "
            "belong to the installed brick version."
        )

    version, entry = _select_installed_entry(cache, selector)
    key = selector.key
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = _installed_info_json(entry, selected)
    declared_models = info_json.get("models", {})
    installed_models = entry["metadata"].get("installed_models", {})
    models = []

    for model_name in model_names:
        validate_model_component("name", model_name)
        metadata = declared_models.get(model_name)
        state = installed_models.get(model_name)
        imported = state is not None and state.get("origin") == "imported"
        if metadata is None and not imported:
            available_models = set(declared_models) | {
                name
                for name, installed_state in installed_models.items()
                if installed_state.get("origin") == "imported"
            }
            available = ", ".join(sorted(available_models)) or "none"
            raise BrickManagerError(
                f"Unknown model for {key}: {model_name}. Available models: {available}"
            )

        path = model_artifact_path(state_paths, selected, model_name)
        installed = (
            _imported_model_installed(path, state)
            if imported
            else _model_installed(
                state_paths,
                selected,
                entry,
                model_name,
                metadata["sha256"],
            )
        )
        models.append(
            BrickModelInfoItem(
                name=model_name,
                version=version,
                installed=installed,
                location=str(path) if installed else None,
                size_bytes=_directory_size_bytes(path) if installed else None,
                manifests=_root_model_manifests(path) if installed else [],
                origin="Imported" if imported else "Declared",
                template=state.get("template") if imported else None,
            )
        )

    return BrickModelInfo(identity=f"{key}@{version}", version=version, models=models)


def get_brick_model_path(
    state_paths: StatePaths,
    brick: str,
    model_name: str,
) -> Path | None:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model path takes <type>:<name> without @version because models "
            "belong to the installed brick version."
        )

    validate_model_component("name", model_name)

    version, entry = _select_installed_entry(cache, selector)
    key = selector.key
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = _installed_info_json(entry, selected)
    declared_models = info_json.get("models", {})
    metadata = declared_models.get(model_name)
    state = entry["metadata"].get("installed_models", {}).get(model_name)
    imported = state is not None and state.get("origin") == "imported"
    if metadata is None and not imported:
        imported_models = {
            name
            for name, installed_state in entry["metadata"]
            .get("installed_models", {})
            .items()
            if installed_state.get("origin") == "imported"
        }
        available = ", ".join(sorted(set(declared_models) | imported_models)) or "none"
        raise BrickManagerError(
            f"Unknown model for {key}: {model_name}. Available models: {available}"
        )

    path = model_artifact_path(state_paths, selected, model_name)
    installed = (
        _imported_model_installed(path, state)
        if imported
        else _model_installed(
            state_paths,
            selected,
            entry,
            model_name,
            metadata["sha256"],
        )
    )
    if not installed:
        return None
    return path


def get_brick_path(state_paths: StatePaths, brick: str) -> Path | None:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Brick path takes <type>:<name> without @version because only one "
            "brick version can be installed."
        )

    versions = cache.get("bricks", {}).get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")

    installed_entries = [
        entry
        for entry in versions.values()
        if entry["metadata"].get("installed", False)
    ]
    if not installed_entries:
        return None
    if len(installed_entries) > 1:
        raise BrickManagerError(
            f"Invalid cache: multiple installed versions found for {selector.key}"
        )

    path = Path(installed_entries[0]["metadata"]["installed_source_path"])
    return path if path.is_dir() else None


def _select_installed_entry(
    cache: dict[str, Any],
    selector: BrickSelector,
) -> tuple[str, dict[str, Any]]:
    versions = cache.get("bricks", {}).get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")

    installed_entries = [
        (version, entry)
        for version, entry in versions.items()
        if entry["metadata"].get("installed", False)
    ]
    if not installed_entries:
        raise BrickManagerError(
            f"{selector.key} is not installed. Install the brick first:\n"
            f"compute-flow brick install {selector.key}"
        )
    if len(installed_entries) > 1:
        raise BrickManagerError(
            f"Invalid cache: multiple installed versions found for {selector.key}"
        )
    return installed_entries[0]


def get_brick_info(state_paths: StatePaths, brick: str) -> BrickInfo:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    entry = select_entry(cache, selector)
    info_json = entry["info_json"]
    key = selector.key
    version = info_json["version"]
    return BrickInfo(
        identity=f"{key}@{version}",
        brick=key,
        version=version,
        info_json=info_json,
        metadata=entry["metadata"],
    )


def show_info(state_paths: StatePaths, brick: str) -> str:
    info = get_brick_info(state_paths, brick)
    return json.dumps(info.info_json, indent=2) + "\n"


def select_entry(cache: dict[str, Any], selector: BrickSelector) -> dict[str, Any]:
    _, entry = select_versioned_entry(cache, selector, require_explicit_latest=False)
    return entry


def select_versioned_entry(
    cache: dict[str, Any],
    selector: BrickSelector,
    *,
    require_explicit_latest: bool = True,
) -> tuple[str, dict[str, Any]]:
    bricks = cache.get("bricks", {})
    versions = bricks.get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")

    if selector.version is not None:
        entry = versions.get(selector.version)
        if entry is None:
            raise BrickManagerError(f"Unknown brick version: {selector.identity}")
        return selector.version, entry

    if len(versions) == 1:
        version = next(iter(versions))
        return version, versions[version]

    if not require_explicit_latest:
        available_versions = ", ".join(_sort_versions(versions))
        raise BrickManagerError(
            f"Multiple versions found for {selector.key}: {available_versions}"
        )

    installed_version = _installed_version(versions)
    if installed_version is not None:
        return installed_version, versions[installed_version]

    version = _sort_versions(versions)[-1]
    return version, versions[version]


def load_required_cache(state_paths: StatePaths) -> dict[str, Any]:
    try:
        return load_cache(state_paths.cache)
    except FileNotFoundError as error:
        raise BrickManagerError(_CACHE_MISSING_MESSAGE) from error


def _iter_entries(cache: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    entries = []
    for brick, versions in cache.get("bricks", {}).items():
        for version, entry in versions.items():
            entries.append((brick, version, entry))
    return entries


def _installed_version(versions: dict[str, Any]) -> str | None:
    installed_versions = [
        version
        for version, entry in versions.items()
        if entry["metadata"].get("installed", False)
    ]
    if not installed_versions:
        return None
    return _sort_versions(
        {version: versions[version] for version in installed_versions}
    )[-1]


def _sort_versions(versions: dict[str, Any]) -> list[str]:
    return sorted(versions, key=_version_sort_key)


def _version_sort_key(version: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"([0-9]+)", version)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _model_installed(
    state_paths: StatePaths,
    selector: BrickSelector,
    entry: dict[str, Any],
    model_name: str,
    expected_sha256: str,
) -> bool:
    path = model_artifact_path(state_paths, selector, model_name)
    state = entry["metadata"].get("installed_models", {}).get(model_name)
    if (
        state is None
        or state.get("origin", "declared") != "declared"
        or Path(state["path"]) != path
        or state["sha256"] != expected_sha256
    ):
        return False
    try:
        return artifact_manifest_sha256(path) == expected_sha256
    except (BrickManagerError, OSError):
        return False


def _imported_model_installed(
    path: Path,
    state: dict[str, Any] | None,
) -> bool:
    return (
        state is not None
        and state.get("origin") == "imported"
        and Path(state["path"]) == path
        and path.is_dir()
        and (path / state["manifest"]).is_file()
    )


def _installed_info_json(
    entry: dict[str, Any],
    selector: BrickSelector,
) -> dict[str, Any]:
    return load_installed_entry_info_json(entry, selector)


def _root_model_manifests(path: Path) -> list[str]:
    return sorted(
        child.name
        for child in path.iterdir()
        if child.is_file() and "manifest" in child.name and child.name.endswith(".json")
    )


def _directory_size_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = [_format_row(headers, widths)]
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines) + "\n"


def _format_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        cell.ljust(widths[index]) for index, cell in enumerate(row)
    ).rstrip()
