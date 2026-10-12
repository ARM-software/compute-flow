#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Cache helpers for the Compute Flow brick manager."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from brick_manager.errors import BrickManagerError


JsonObject = dict[str, Any]


def load_cache(path: Path) -> JsonObject:
    try:
        with path.open(encoding="utf-8") as cache_file:
            cache = json.load(cache_file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise BrickManagerError(f"Invalid cache JSON: {path}") from error

    if not isinstance(cache, dict):
        raise BrickManagerError(f"Invalid cache JSON: {path}")

    validate_cache(cache)
    return cache


def write_cache(path: Path, cache: JsonObject) -> None:
    validate_cache(cache)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
    ) as cache_file:
        json.dump(cache, cache_file, indent=2)
        cache_file.write("\n")
        temporary_path = Path(cache_file.name)

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_cache(cache: JsonObject) -> None:
    bricks = cache.get("bricks", {})
    if not isinstance(bricks, dict):
        raise BrickManagerError("Invalid cache: bricks must be an object")

    for versions in bricks.values():
        if not isinstance(versions, dict):
            raise BrickManagerError("Invalid cache: brick versions must be objects")

        for entry in versions.values():
            if not isinstance(entry, dict):
                raise BrickManagerError("Invalid cache: brick entry must be an object")

            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict):
                raise BrickManagerError("Invalid cache: metadata must be an object")

            installed = metadata.get("installed", False)
            installed_source_path = metadata.get("installed_source_path")
            if installed and not installed_source_path:
                raise BrickManagerError(
                    "Invalid cache: installed entries need installed_source_path"
                )
            if not installed and installed_source_path is not None:
                raise BrickManagerError(
                    "Invalid cache: not-installed entries must use installed_source_path: null"
                )

            installed_info_json = metadata.get("installed_info_json")
            if installed_info_json is not None and not isinstance(
                installed_info_json, dict
            ):
                raise BrickManagerError(
                    "Invalid cache: installed_info_json must be an object or null"
                )
            if not installed and installed_info_json is not None:
                raise BrickManagerError(
                    "Invalid cache: not-installed entries must use installed_info_json: null"
                )

            installed_models = metadata.get("installed_models", {})
            if not isinstance(installed_models, dict):
                raise BrickManagerError(
                    "Invalid cache: installed_models must be an object"
                )
            for model_name, model_state in installed_models.items():
                if not isinstance(model_name, str) or not model_name:
                    raise BrickManagerError(
                        "Invalid cache: installed model names must be non-empty strings"
                    )
                if not isinstance(model_state, dict):
                    raise BrickManagerError(
                        f"Invalid cache: installed model state must be an object: {model_name}"
                    )
                origin = model_state.get("origin", "declared")
                if origin not in ("declared", "imported"):
                    raise BrickManagerError(
                        f"Invalid cache: unsupported model origin: {model_name}.{origin}"
                    )
                required_fields = (
                    ("path", "template", "manifest")
                    if origin == "imported"
                    else ("path", "ref", "sha256")
                )
                for field in required_fields:
                    if (
                        not isinstance(model_state.get(field), str)
                        or not model_state[field]
                    ):
                        raise BrickManagerError(
                            "Invalid cache: installed model state needs "
                            f"{model_name}.{field}"
                        )
                if origin == "imported" and not isinstance(
                    model_state.get("brick"), dict
                ):
                    raise BrickManagerError(
                        f"Invalid cache: imported model state needs {model_name}.brick"
                    )
                if origin == "imported" and model_state.get("api") is not None:
                    if (
                        not isinstance(model_state["api"], str)
                        or not model_state["api"]
                    ):
                        raise BrickManagerError(
                            "Invalid cache: imported model API must be a non-empty "
                            f"string: {model_name}.api"
                        )
