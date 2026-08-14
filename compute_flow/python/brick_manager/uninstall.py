#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Implementation of compute-flow brick uninstall."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from brick_manager.cache import write_cache
from brick_manager.catalog import load_required_cache
from brick_manager.errors import BrickManagerError
from brick_manager.identity import BrickSelector, parse_brick_selector
from brick_manager.paths import StatePaths


ConfirmCallback = Callable[[Path], bool]


@dataclass(frozen=True)
class UninstallResult:
    identity: str
    install_path: Path
    status: str
    kept_models: list[str]


@dataclass(frozen=True)
class UninstallAllResult:
    cache_path: Path
    total: int
    statuses: dict[str, int]
    kept_models: int


def uninstall_brick(
    state_paths: StatePaths,
    brick: str,
    *,
    confirm: ConfirmCallback | None = None,
) -> UninstallResult:
    cache = load_required_cache(state_paths)
    selector = parse_brick_selector(brick)
    version, entry = _select_installed_entry(cache, selector)
    selected = BrickSelector(selector.brick_type, selector.name, version)
    metadata = entry["metadata"]

    if not metadata.get("installed", False):
        raise BrickManagerError(f"Brick is not installed: {selector.key}")

    installed_path = Path(metadata["installed_source_path"])
    managed_root = state_paths.bricks.resolve()
    if not installed_path.resolve().is_relative_to(managed_root):
        raise BrickManagerError(
            f"Installed source path is outside managed bricks directory: {installed_path}"
        )

    _confirm_uninstall(installed_path, confirm)
    if installed_path.exists():
        shutil.rmtree(installed_path)
        status = "uninstalled"
    else:
        status = "cleared_missing"

    metadata["installed"] = False
    metadata["installed_source_path"] = None
    kept_models = _installed_model_names(entry)
    write_cache(state_paths.cache, cache)
    return UninstallResult(
        identity=selected.identity,
        install_path=installed_path,
        status=status,
        kept_models=kept_models,
    )


def uninstall_all_bricks(
    state_paths: StatePaths,
    *,
    confirm: ConfirmCallback | None = None,
) -> UninstallAllResult:
    cache = load_required_cache(state_paths)
    identities = [
        brick
        for brick, versions in sorted(cache.get("bricks", {}).items())
        for entry in versions.values()
        if entry["metadata"].get("installed", False)
    ]
    statuses = {
        "uninstalled": 0,
        "cleared_missing": 0,
    }
    kept_models = 0

    if identities:
        _confirm_uninstall_all(state_paths.bricks, len(identities), confirm)

    for identity in identities:
        try:
            result = uninstall_brick(
                state_paths,
                identity,
                confirm=lambda path: True,
            )
        except BrickManagerError as error:
            raise BrickManagerError(
                f"Failed to uninstall {identity}: {error}"
            ) from error
        statuses[result.status] = statuses.get(result.status, 0) + 1
        kept_models += len(result.kept_models)

    return UninstallAllResult(
        cache_path=state_paths.cache,
        total=len(identities),
        statuses=statuses,
        kept_models=kept_models,
    )


def _select_installed_entry(
    cache: dict,
    selector,
) -> tuple[str, dict]:
    if selector.version is not None:
        raise BrickManagerError(
            "Brick uninstall takes <type>:<name> without @version because only "
            "one brick version can be installed."
        )

    versions = cache.get("bricks", {}).get(selector.key)
    if not versions:
        raise BrickManagerError(f"Unknown brick: {selector.key}")

    installed_entries = [
        (version, entry)
        for version, entry in versions.items()
        if entry["metadata"].get("installed", False)
    ]
    if not installed_entries:
        raise BrickManagerError(f"Brick is not installed: {selector.key}")
    if len(installed_entries) > 1:
        raise BrickManagerError(
            f"Invalid cache: multiple installed versions found for {selector.key}"
        )
    return installed_entries[0]


def _installed_model_names(entry: dict) -> list[str]:
    return sorted(entry["metadata"].setdefault("installed_models", {}))


def _confirm_uninstall_all(
    path: Path,
    count: int,
    confirm: ConfirmCallback | None,
) -> None:
    if confirm is not None:
        if confirm(path):
            return
        raise BrickManagerError("Uninstall cancelled.")

    if not sys.stdin.isatty():
        raise BrickManagerError(
            "Refusing to uninstall all bricks without interactive confirmation."
        )

    print(
        f"Remove {count} installed brick source tree(s) under {path}? [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    answer = sys.stdin.readline().strip().lower()
    if answer not in ("y", "yes"):
        raise BrickManagerError("Uninstall cancelled.")


def _confirm_uninstall(path: Path, confirm: ConfirmCallback | None) -> None:
    if confirm is not None:
        if confirm(path):
            return
        raise BrickManagerError("Uninstall cancelled.")

    if not sys.stdin.isatty():
        raise BrickManagerError(
            "Refusing to uninstall without interactive confirmation."
        )

    print(
        f"Remove installed brick source at {path}? [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    answer = sys.stdin.readline().strip().lower()
    if answer not in ("y", "yes"):
        raise BrickManagerError("Uninstall cancelled.")
