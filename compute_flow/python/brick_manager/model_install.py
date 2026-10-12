#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of compute-flow model installation and removal."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brick_manager.cache import write_cache
from brick_manager.catalog import load_required_cache, select_versioned_entry
from brick_manager.errors import BrickManagerError
from brick_manager.identity import (
    BrickSelector,
    parse_brick_selector,
    validate_model_component,
)
from brick_manager.model_artifacts import (
    artifact_manifest_sha256,
    install_artifact,
    model_artifact_path,
    remove_artifact,
)
from brick_manager.metadata import load_installed_entry_info_json
from brick_manager.model_import import imported_model_compatible
from brick_manager.paths import StatePaths


ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class ModelInstallItem:
    name: str
    path: Path
    status: str


@dataclass(frozen=True)
class ModelInstallResult:
    identity: str
    items: list[ModelInstallItem]


@dataclass(frozen=True)
class ModelUninstallItem:
    name: str
    path: Path
    status: str


@dataclass(frozen=True)
class ModelUninstallResult:
    identity: str
    items: list[ModelUninstallItem]


@dataclass(frozen=True)
class ModelReconcileResult:
    removed: list[str]
    incompatible: list[str]


def validate_models_for_install(
    state_paths: StatePaths,
    brick: str,
    model_names: list[str],
) -> str:
    """Validate models against the brick version selected by model install."""
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model install takes <type>:<name> without @version. "
            "You can install a particular version of the brick using the following command:\n"
            f"compute-flow brick install {selector.identity}"
        )

    version, entry = select_versioned_entry(cache, selector)
    selected = BrickSelector(selector.brick_type, selector.name, version)
    info_json = entry["info_json"]
    if _installed_source_exists(entry):
        info_json = _installed_info_json(entry, selected)
    _validate_requested_models(
        selector.key,
        info_json.get("models", {}),
        model_names,
    )
    return selected.identity


def install_models(
    state_paths: StatePaths,
    brick: str,
    model_names: list[str],
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> ModelInstallResult:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model installation takes <type>:<name> without @version. "
            "You can install a particular version of the brick using the following command:\n"
            f"compute-flow brick install {selector.identity}"
        )

    version, entry = _select_installed_entry(cache, selector)
    installed_selector = BrickSelector(selector.brick_type, selector.name, version)
    info_json = _installed_info_json(entry, installed_selector)
    declared_models = info_json.get("models", {})
    _validate_requested_models(selector.key, declared_models, model_names)
    installed_models = _installed_models(entry)
    items = []

    for model_name in model_names:
        _report(progress, model_name, "checking")
        model = declared_models[model_name]

        path = model_artifact_path(state_paths, installed_selector, model_name)
        state = installed_models.get(model_name)
        if not force and _installed_artifact_matches(
            path,
            state,
            model["sha256"],
        ):
            status = "already_installed"
        else:
            path = install_artifact(
                state_paths,
                installed_selector,
                model_name,
                sources=model["sources"],
                expected_sha256=model["sha256"],
                progress=lambda phase: _report(progress, model_name, phase),
            )
            status = "installed"

        installed_models[model_name] = _model_state(path, model)
        items.append(ModelInstallItem(model_name, path, status))
        _report(progress, model_name, "done")

    write_cache(state_paths.cache, cache)
    return ModelInstallResult(
        identity=installed_selector.identity,
        items=items,
    )


def uninstall_models(
    state_paths: StatePaths,
    brick: str,
    model_names: list[str],
) -> ModelUninstallResult:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model uninstall takes <type>:<name> without @version because models "
            "belong to the installed brick version."
        )

    for model_name in model_names:
        validate_model_component("name", model_name)

    version, entry = _select_installed_entry(cache, selector)
    installed_selector = BrickSelector(selector.brick_type, selector.name, version)
    declared_models = _installed_info_json(entry, installed_selector).get("models", {})
    installed_models = _installed_models(entry)
    items = []

    for model_name in model_names:
        if model_name not in declared_models and model_name not in installed_models:
            available = ", ".join(sorted(declared_models)) or "none"
            raise BrickManagerError(
                f"Unknown model for {selector.key}: {model_name}. "
                f"Available models: {available}"
            )

        path = model_artifact_path(state_paths, installed_selector, model_name)
        had_state = model_name in installed_models
        if path.exists():
            remove_artifact(path)
            status = "uninstalled"
        elif had_state:
            status = "cleared_missing"
        else:
            status = "not_installed"

        installed_models.pop(model_name, None)
        items.append(ModelUninstallItem(model_name, path, status))

    write_cache(state_paths.cache, cache)
    return ModelUninstallResult(
        identity=installed_selector.identity,
        items=items,
    )


def uninstall_all_models(
    state_paths: StatePaths,
    brick: str,
) -> ModelUninstallResult:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    if selector.version is not None:
        raise BrickManagerError(
            "Model uninstall --all takes <type>:<name> without @version because "
            "installed model artifacts are keyed by brick identity."
        )

    versions = cache.get("bricks", {}).get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")

    items = []
    removed_paths = set()
    for entry in versions.values():
        installed_models = _installed_models(entry)
        for model_name, state in sorted(list(installed_models.items())):
            path = Path(state["path"])
            status = _remove_model_artifact(path, had_state=True)
            installed_models.pop(model_name, None)
            items.append(ModelUninstallItem(model_name, path, status))
            removed_paths.add(path)

    model_root = state_paths.models / selector.brick_type / selector.name
    if model_root.is_dir():
        for path in sorted(model_root.iterdir()):
            if path in removed_paths:
                continue
            status = _remove_model_artifact(path, had_state=False)
            items.append(ModelUninstallItem(path.name, path, status))

    _remove_empty_directory(model_root)
    _remove_empty_directory(model_root.parent)
    write_cache(state_paths.cache, cache)
    return ModelUninstallResult(
        identity=selector.key,
        items=items,
    )


def reconcile_installed_models(
    state_paths: StatePaths,
    cache: dict[str, Any],
    selector: BrickSelector,
    entry: dict[str, Any],
    info_json: dict[str, Any],
) -> ModelReconcileResult:
    declared_models = info_json.get("models", {})
    installed_models = _collect_installed_models(cache, selector.key)
    model_root = state_paths.models / selector.brick_type / selector.name
    model_paths = {}
    for model_name in installed_models:
        validate_model_component("name", model_name)
        model_paths[model_name] = model_artifact_path(
            state_paths,
            selector,
            model_name,
        )
    if model_root.is_symlink():
        remove_artifact(model_root)
    elif model_root.is_dir():
        model_paths.update((path.name, path) for path in model_root.iterdir())
    next_installed_models = _installed_models(entry)
    next_installed_models.clear()

    removed = []
    incompatible = []
    for model_name, path in sorted(model_paths.items()):
        state = installed_models.get(model_name, {})
        if state.get("origin") == "imported":
            if imported_model_compatible(path, state, info_json):
                next_installed_models[model_name] = state
                continue

            remove_artifact(path)
            incompatible.append(model_name)
            continue

        model = declared_models.get(model_name)
        if model is None:
            remove_artifact(path)
            removed.append(model_name)
            continue

        if _artifact_matches(path, model["sha256"]):
            next_installed_models[model_name] = _model_state(path, model)
            continue

        remove_artifact(path)
        incompatible.append(model_name)

    _remove_empty_directory(model_root)
    _remove_empty_directory(model_root.parent)
    _clear_other_installed_model_state(cache, selector.key, selector.version)
    return ModelReconcileResult(removed=removed, incompatible=incompatible)


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


def _validate_requested_models(
    brick_key: str,
    declared_models: dict[str, Any],
    model_names: list[str],
) -> None:
    for model_name in model_names:
        validate_model_component("name", model_name)

    for model_name in model_names:
        if model_name not in declared_models:
            available = ", ".join(sorted(declared_models)) or "none"
            raise BrickManagerError(
                f"Unknown model for {brick_key}: {model_name}. "
                f"Available models: {available}"
            )


def _collect_installed_models(
    cache: dict[str, Any],
    brick_key: str,
) -> dict[str, dict[str, str]]:
    installed_models = {}
    for entry in cache.get("bricks", {}).get(brick_key, {}).values():
        installed_models.update(_installed_models(entry))
    return installed_models


def _clear_other_installed_model_state(
    cache: dict[str, Any],
    brick_key: str,
    selected_version: str | None,
) -> None:
    for version, entry in cache.get("bricks", {}).get(brick_key, {}).items():
        if version != selected_version:
            _installed_models(entry).clear()


def _installed_models(entry: dict[str, Any]) -> dict[str, dict[str, str]]:
    return entry["metadata"].setdefault("installed_models", {})


def _remove_model_artifact(path: Path, *, had_state: bool) -> str:
    if path.exists():
        remove_artifact(path)
        return "uninstalled"
    if had_state:
        return "cleared_missing"
    return "not_installed"


def _remove_empty_directory(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _artifact_matches(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        return artifact_manifest_sha256(path) == expected_sha256
    except (BrickManagerError, OSError):
        return False


def _installed_artifact_matches(
    path: Path,
    state: dict[str, str] | None,
    expected_sha256: str,
) -> bool:
    return (
        state is not None
        and state.get("origin", "declared") == "declared"
        and Path(state["path"]) == path
        and state["sha256"] == expected_sha256
        and _artifact_matches(path, expected_sha256)
    )


def _installed_info_json(
    entry: dict[str, Any],
    selector: BrickSelector,
) -> dict[str, Any]:
    return load_installed_entry_info_json(entry, selector)


def _installed_source_exists(entry: dict[str, Any]) -> bool:
    source_path = entry["metadata"].get("installed_source_path")
    return bool(source_path) and Path(source_path).is_dir()


def _model_state(path: Path, model: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(path),
        "ref": model["sources"][0].get("ref", "local"),
        "sha256": model["sha256"],
    }


def _report(
    progress: ProgressCallback | None,
    model_name: str,
    phase: str,
) -> None:
    if progress is not None:
        progress(model_name, phase)
