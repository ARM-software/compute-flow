#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Registry source-list handling."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from brick_manager.errors import BrickManagerError


_GIT_URL_SCHEMES = frozenset(("file", "git", "http", "https", "ssh"))
_SCP_GIT_URL = re.compile(r"^[^/\\:@]+@[^/\\:]+:.+$")


def is_git_source(source: str) -> bool:
    """Return whether a registry entry names a Git repository URL."""
    parsed_source = urlparse(source)
    if parsed_source.scheme in _GIT_URL_SCHEMES:
        return True
    if _SCP_GIT_URL.fullmatch(source):
        return True
    if parsed_source.scheme and len(parsed_source.scheme) != 1:
        raise BrickManagerError(f"Unsupported registry source URL: {source}")
    return False


def default_registry_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "registry.txt"
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            return candidate
    raise BrickManagerError(
        "No default registry found. Run from a source checkout or pass --registry."
    )


def parse_registry(registry_path: Path) -> list[str]:
    try:
        lines = registry_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise BrickManagerError(f"Registry file not found: {registry_path}") from error

    sources: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise BrickManagerError(
                f"Invalid blank registry line: {registry_path}:{line_number}"
            )
        if line != line.strip():
            raise BrickManagerError(
                f"Invalid whitespace in registry line: {registry_path}:{line_number}"
            )
        if line.startswith("#"):
            raise BrickManagerError(
                f"Invalid comment registry line: {registry_path}:{line_number}"
            )
        if line in seen:
            raise BrickManagerError(f"Duplicate registry source: {line}")
        seen.add(line)
        sources.append(line)
    return sources


def registry_entries(
    registry_path: Path, user_registry_path: Path | None = None
) -> list[tuple[Path, str]]:
    """Read packaged and optional user sources with their path origins."""
    entries = [(registry_path, source) for source in parse_registry(registry_path)]
    if user_registry_path is not None and user_registry_path.is_file():
        entries.extend(
            (user_registry_path, source)
            for source in parse_registry(user_registry_path)
        )
    return entries


def resolve_local_source(registry_path: Path, source: str) -> Path:
    if is_git_source(source):
        raise BrickManagerError(f"Expected a local registry source: {source}")

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = registry_path.parent / source_path
    return source_path.resolve()
