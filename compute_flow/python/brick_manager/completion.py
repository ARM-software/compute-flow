#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Cache-backed shell completion for the Compute Flow brick manager."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from brick_manager.cache import load_cache
from brick_manager.catalog import select_versioned_entry
from brick_manager.errors import BrickManagerError
from brick_manager.identity import parse_brick_selector
from brick_manager.paths import StatePaths


_TOP_LEVEL_COMMANDS = ("build", "brick", "check", "model", "update")
_BRICK_COMMANDS = ("info", "install", "list", "path", "uninstall")
_MODEL_COMMANDS = ("info", "install", "list", "path", "uninstall")
_BUILD_TARGETS = ("android_arm64", "linux_arm64", "list", "native")
_BUILD_OPTIONS = ("--from-source", "--jobs", "-j")


def complete_cli(state_paths: StatePaths, words: Sequence[str]) -> list[str]:
    """Return completion candidates for words after the compute-flow executable."""
    current, previous = _split_current(words)
    if not previous:
        return _matching(_TOP_LEVEL_COMMANDS, current)

    command = previous[0]
    if command == "brick":
        return _complete_brick_command(state_paths, previous[1:], current)
    if command == "model":
        return _complete_model_command(state_paths, previous[1:], current)
    if command == "build":
        return _complete_build_command(previous[1:], current)
    if command == "update" and current.startswith("-"):
        return _matching(("--registry",), current)
    return []


def _complete_build_command(previous: list[str], current: str) -> list[str]:
    if previous and previous[-1] in ("--jobs", "-j"):
        return []
    if current.startswith("-"):
        return _matching(_BUILD_OPTIONS, current)
    if any(word in _BUILD_TARGETS for word in previous):
        return _matching(_BUILD_OPTIONS, current) if not current else []
    return _matching(_BUILD_TARGETS, current)


def complete_cli_with_descriptions(
    state_paths: StatePaths, words: Sequence[str]
) -> list[tuple[str, str]]:
    """Return completion candidates with descriptions when completing models."""
    candidates = complete_cli(state_paths, words)
    descriptions = _model_descriptions_for_completion(state_paths, words)
    return [(candidate, descriptions.get(candidate, "")) for candidate in candidates]


def _complete_brick_command(
    state_paths: StatePaths,
    previous: list[str],
    current: str,
) -> list[str]:
    if not previous:
        return _matching(_BRICK_COMMANDS, current)

    command = previous[0]
    if command not in _BRICK_COMMANDS:
        return []

    options = {
        "install": ("--all", "--force"),
        "uninstall": ("--all", "--yes", "-y"),
    }.get(command, ())
    if current.startswith("-"):
        return _matching(options, current)

    positionals = _positionals(previous[1:], options)
    if command == "list" or positionals:
        return []
    if "--all" in previous:
        return []

    cache = _load_cache(state_paths)
    return _complete_bricks(
        cache,
        current,
        installed_only=command in ("path", "uninstall"),
        allow_versions=command in ("info", "install"),
    )


def _complete_model_command(
    state_paths: StatePaths,
    previous: list[str],
    current: str,
) -> list[str]:
    if not previous:
        return _matching(_MODEL_COMMANDS, current)

    command = previous[0]
    if command not in _MODEL_COMMANDS:
        return []

    options = {
        "install": ("--force",),
        "uninstall": ("--all",),
    }.get(command, ())
    if current.startswith("-"):
        return _matching(options, current)

    positionals = _positionals(previous[1:], options)
    cache = _load_cache(state_paths)
    if command == "list":
        return _complete_bricks(
            cache,
            current,
            allow_versions=True,
            excluded=set(positionals),
        )

    if not positionals:
        return _complete_bricks(
            cache,
            current,
            installed_only=command in ("info", "path", "uninstall"),
            require_models=True,
        )

    if command == "uninstall" and "--all" in previous:
        return []
    if command == "path" and len(positionals) > 1:
        return []

    brick = positionals[0]
    entered_models = set(positionals[1:])
    models = _models_for_command(cache, brick, command)
    return _matching(
        (model for model in models if model not in entered_models), current
    )


def _complete_bricks(
    cache: dict[str, Any],
    current: str,
    *,
    installed_only: bool = False,
    allow_versions: bool = False,
    require_models: bool = False,
    excluded: set[str] | None = None,
) -> list[str]:
    excluded = excluded or set()
    bricks = cache.get("bricks", {})
    keys = [
        key
        for key, versions in bricks.items()
        if key not in excluded
        and (not installed_only or _installed_entry(versions) is not None)
        and (not require_models or _selected_models(cache, key, installed_only))
    ]

    if "@" in current:
        if not allow_versions:
            return []
        key, _, _version_prefix = current.partition("@")
        versions = bricks.get(key, {})
        return _matching((f"{key}@{version}" for version in versions), current)

    if ":" not in current:
        brick_types = {key.partition(":")[0] for key in keys}
        return _matching((f"{brick_type}:" for brick_type in brick_types), current)

    return _matching(keys, current)


def _models_for_command(
    cache: dict[str, Any],
    brick: str,
    command: str,
) -> list[str]:
    versions = cache.get("bricks", {}).get(brick)
    if not versions:
        return []

    if command == "install":
        entry = _selected_entry(cache, brick)
    else:
        entry = _installed_entry(versions)
    if entry is None:
        return []

    if command in ("path", "uninstall"):
        return sorted(entry["metadata"].get("installed_models", {}))
    return sorted(entry["info_json"].get("models", {}))


def _model_descriptions_for_completion(
    state_paths: StatePaths, words: Sequence[str]
) -> dict[str, str]:
    _current, previous = _split_current(words)
    if len(previous) < 3 or previous[0] != "model":
        return {}

    command = previous[1]
    if command not in ("info", "install", "path", "uninstall"):
        return {}

    options = {
        "install": ("--force",),
        "uninstall": ("--all",),
    }.get(command, ())
    positionals = _positionals(previous[2:], options)
    if not positionals:
        return {}

    cache = _load_cache(state_paths)
    versions = cache.get("bricks", {}).get(positionals[0])
    if not versions:
        return {}

    entry = (
        _selected_entry(cache, positionals[0])
        if command == "install"
        else _installed_entry(versions)
    )
    if entry is None:
        return {}

    declared_models = entry["info_json"].get("models", {})
    if command in ("path", "uninstall"):
        model_names = entry["metadata"].get("installed_models", {})
    else:
        model_names = declared_models

    return {
        model_name: declared_models[model_name]["description"]
        for model_name in model_names
        if model_name in declared_models
    }


def _selected_models(
    cache: dict[str, Any],
    brick: str,
    installed_only: bool,
) -> dict[str, Any]:
    versions = cache.get("bricks", {}).get(brick, {})
    entry = (
        _installed_entry(versions) if installed_only else _selected_entry(cache, brick)
    )
    if entry is None:
        return {}
    return entry["info_json"].get("models", {})


def _selected_entry(cache: dict[str, Any], brick: str) -> dict[str, Any] | None:
    try:
        selector = parse_brick_selector(brick)
        _, entry = select_versioned_entry(cache, selector)
    except BrickManagerError:
        return None
    return entry


def _installed_entry(versions: dict[str, Any]) -> dict[str, Any] | None:
    entries = [
        entry
        for entry in versions.values()
        if entry["metadata"].get("installed", False)
    ]
    return entries[0] if len(entries) == 1 else None


def _load_cache(state_paths: StatePaths) -> dict[str, Any]:
    try:
        return load_cache(state_paths.cache)
    except (BrickManagerError, OSError):
        return {"bricks": {}}


def _positionals(words: Sequence[str], options: Sequence[str]) -> list[str]:
    return [word for word in words if word not in options]


def _split_current(words: Sequence[str]) -> tuple[str, list[str]]:
    if not words:
        return "", []
    return words[-1], list(words[:-1])


def _matching(candidates: Iterable[str], prefix: str) -> list[str]:
    return sorted(
        {candidate for candidate in candidates if candidate.startswith(prefix)}
    )
