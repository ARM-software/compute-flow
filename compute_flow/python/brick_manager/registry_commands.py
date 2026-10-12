#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Manage brick source lists."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from brick_manager.errors import BrickManagerError
from brick_manager.paths import StatePaths
from brick_manager.registry import (
    default_registry_path,
    is_git_source,
    parse_registry,
    registry_entries,
)


@dataclass(frozen=True)
class RegistryAddResult:
    """Registration status for a Git source."""

    source: str
    registry_path: Path
    status: str


@dataclass(frozen=True)
class RegistryRemoveResult:
    """The source removed from a registry file."""

    source: str
    registry_path: Path


@dataclass(frozen=True)
class RegistrySource:
    """A configured brick source and the registry that owns it."""

    source: str
    origin: str


def add_registry_source(
    state_paths: StatePaths, url: str, registry: str | None = None
) -> RegistryAddResult:
    """Register a Git brick source for the next update."""
    if not url or url != url.strip() or any(character.isspace() for character in url):
        raise BrickManagerError(
            "Brick source URL must be non-empty and contain no whitespace"
        )
    if not is_git_source(url):
        raise BrickManagerError(f"Brick source must be a Git URL: {url}")

    registry_path = _target_registry_path(state_paths, registry)
    if registry is None:
        registered = registry_entries(default_registry_path(), state_paths.registry)
    else:
        registered = registry_entries(registry_path) if registry_path.is_file() else []
    if any(source == url for _, source in registered):
        return RegistryAddResult(url, registry_path, "already_registered")

    original_registry = (
        registry_path.read_text(encoding="utf-8") if registry_path.is_file() else None
    )
    updated_registry = original_registry or ""
    if updated_registry and not updated_registry.endswith("\n"):
        updated_registry += "\n"
    updated_registry += f"{url}\n"
    _write_registry(registry_path, updated_registry)

    return RegistryAddResult(url, registry_path, "added")


def remove_registry_source(
    state_paths: StatePaths, source: str, registry: str | None = None
) -> RegistryRemoveResult:
    """Remove one source for the next update without uninstalling bricks."""
    registry_path = _target_registry_path(state_paths, registry)
    if not registry_path.is_file():
        if registry is None and source in parse_registry(default_registry_path()):
            raise BrickManagerError(
                f"Source belongs to the packaged registry: {source}"
            )
        raise BrickManagerError(f"Registry file not found: {registry_path}")

    sources = parse_registry(registry_path)
    if source not in sources:
        if registry is None and source in parse_registry(default_registry_path()):
            raise BrickManagerError(
                f"Source belongs to the packaged registry: {source}"
            )
        raise BrickManagerError(f"Registry source not found: {source}")

    updated_registry = "".join(f"{entry}\n" for entry in sources if entry != source)
    _write_registry(registry_path, updated_registry)

    return RegistryRemoveResult(source, registry_path)


def list_registry_sources(
    state_paths: StatePaths, registry: str | None = None
) -> list[RegistrySource]:
    """List configured sources from the default pair or one selected file."""
    if registry is not None:
        registry_path = _target_registry_path(state_paths, registry)
        return [
            RegistrySource(source, "Selected")
            for source in parse_registry(registry_path)
        ]

    packaged_path = default_registry_path()
    return [
        RegistrySource(
            source,
            "Packaged" if source_registry == packaged_path else "User",
        )
        for source_registry, source in registry_entries(
            packaged_path, state_paths.registry
        )
    ]


def _target_registry_path(state_paths: StatePaths, registry: str | None) -> Path:
    return (
        Path(registry).expanduser().resolve()
        if registry is not None
        else state_paths.registry
    )


def _write_registry(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    ) as registry_file:
        registry_file.write(contents)
        temporary_path = Path(registry_file.name)
    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
