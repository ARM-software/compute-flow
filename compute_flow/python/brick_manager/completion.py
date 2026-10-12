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


_TOP_LEVEL_COMMANDS = ("build", "brick", "check", "model", "registry", "update")
_BRICK_COMMANDS = ("info", "install", "list", "path", "uninstall")
_MODEL_COMMANDS = ("import", "info", "install", "list", "path", "uninstall")
_REGISTRY_COMMANDS = ("add", "list", "remove")
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
    if command == "registry":
        return _complete_registry_command(previous[1:], current)
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


def _complete_registry_command(previous: list[str], current: str) -> list[str]:
    if not previous:
        return _matching(_REGISTRY_COMMANDS, current)
    if previous[0] not in _REGISTRY_COMMANDS or previous[-1] == "--registry":
        return []
    if current.startswith("-") or current == "":
        return _matching(("--registry",), current)
    return []


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
        "list": ("--installed", "--type"),
        "install": ("--all", "--force"),
        "uninstall": ("--all", "--yes", "-y"),
    }.get(command, ())
    if command == "list" and current.startswith("--type="):
        if any(word == "--type" or word.startswith("--type=") for word in previous[1:]):
            return []
        return _complete_brick_types(
            _load_cache(state_paths),
            current,
        )
    if current.startswith("-"):
        return _matching(options, current)

    positionals = _positionals(previous[1:], options)
    if command == "list":
        if previous[-1] == "--type":
            return _complete_brick_types(
                _load_cache(state_paths),
                current,
            )
        return [] if positionals else _matching(options, current)
    if positionals:
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


def _complete_brick_types(
    cache: dict[str, Any],
    current: str,
) -> list[str]:
    option_prefix = ""
    current_value = current
    if current.startswith("--type="):
        option_prefix = "--type="
        current_value = current.removeprefix(option_prefix)

    value_prefix, separator, type_prefix = current_value.rpartition(",")
    completed_prefix = f"{value_prefix}{separator}"
    excluded = set(value_prefix.split(",") if value_prefix else ())
    brick_types = {
        key.partition(":")[0]
        for key in cache.get("bricks", {})
        if key.partition(":")[0] not in excluded
    }
    return [
        f"{option_prefix}{completed_prefix}{brick_type}"
        for brick_type in _matching(brick_types, type_prefix)
    ]


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
        "import": ("--force",),
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

    if command == "import":
        if len(positionals) == 1:
            models = _models_for_command(cache, positionals[0], command)
            return _matching(models, current)
        return []

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

    if command in ("import", "install"):
        entry = _selected_entry(cache, brick)
    else:
        entry = _installed_entry(versions)
    if entry is None:
        return []

    installed_models = entry["metadata"].get("installed_models", {})
    if command == "import":
        return sorted(_entry_info_json(entry).get("models", {}))
    if command == "uninstall":
        return sorted(installed_models)

    declared_models = _entry_info_json(entry).get("models", {})
    if command == "path":
        return sorted(
            model_name
            for model_name, state in installed_models.items()
            if (
                state.get("origin") == "imported"
                or (
                    model_name in declared_models
                    and state.get("sha256") == declared_models[model_name]["sha256"]
                )
            )
        )
    return sorted(declared_models)


def _model_descriptions_for_completion(
    state_paths: StatePaths, words: Sequence[str]
) -> dict[str, str]:
    _current, previous = _split_current(words)
    if len(previous) < 3 or previous[0] != "model":
        return {}

    command = previous[1]
    if command not in ("import", "info", "install", "path", "uninstall"):
        return {}

    options = {
        "import": ("--force",),
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
        if command in ("import", "install")
        else _installed_entry(versions)
    )
    if entry is None:
        return {}

    declared_models = _entry_info_json(entry).get("models", {})
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
    declared_models = _entry_info_json(entry).get("models", {})
    if installed_only:
        imported_models = {
            name: {}
            for name, state in entry["metadata"].get("installed_models", {}).items()
            if state.get("origin") == "imported"
        }
        return declared_models | imported_models
    return declared_models


def _entry_info_json(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry["metadata"]
    if metadata.get("installed", False):
        return metadata.get("installed_info_json") or entry["info_json"]
    return entry["info_json"]


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
