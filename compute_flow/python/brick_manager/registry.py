#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Registry source-list handling."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from brick_manager.errors import BrickManagerError


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


def resolve_local_source(registry_path: Path, source: str) -> Path:
    parsed_source = urlparse(source)
    if parsed_source.scheme:
        raise BrickManagerError(
            f"Remote registry sources are not implemented yet: {source}"
        )

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = registry_path.parent / source_path
    return source_path.resolve()
