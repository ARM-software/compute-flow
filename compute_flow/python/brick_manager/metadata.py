#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Brick info.json loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brick_manager.dependencies import validate_dependencies_field
from brick_manager.errors import BrickManagerError
from brick_manager.identity import (
    validate_brick_component,
    validate_brick_class_name,
    validate_model_component,
    validate_version,
)


JsonObject = dict[str, Any]

_REQUIRED_FIELDS = ("name", "type", "description", "author", "version", "models")


def load_info_json(source_path: Path) -> JsonObject:
    info_path = source_path / "info.json"
    try:
        with info_path.open(encoding="utf-8") as info_file:
            info_json = json.load(info_file)
    except FileNotFoundError as error:
        raise BrickManagerError(f"Missing info.json: {info_path}") from error
    except json.JSONDecodeError as error:
        raise BrickManagerError(f"Invalid info.json: {info_path}") from error

    if not isinstance(info_json, dict):
        raise BrickManagerError(f"Invalid info.json: {info_path}")

    validate_info_json(info_json)
    return info_json


def validate_info_json(info_json: JsonObject) -> None:
    for field in _REQUIRED_FIELDS:
        if field not in info_json:
            raise BrickManagerError(f"info.json is missing required field: {field}")

    for field in ("name", "type", "description", "author", "version"):
        if not isinstance(info_json[field], str) or not info_json[field]:
            raise BrickManagerError(
                f"info.json field must be a non-empty string: {field}"
            )

    validate_brick_component("type", info_json["type"])
    validate_brick_class_name(info_json["name"])
    validate_version(info_json["version"])
    if "deps" in info_json:
        validate_dependencies_field(info_json["deps"])

    models = info_json["models"]
    if not isinstance(models, dict):
        raise BrickManagerError("info.json field must be an object: models")

    for model_name, model in models.items():
        validate_model_component("name", model_name)
        if not isinstance(model, dict):
            raise BrickManagerError(f"model entry must be an object: {model_name}")
        for field in ("description", "url", "ref", "sha256"):
            if field not in model:
                raise BrickManagerError(
                    f"model entry is missing required field: {model_name}.{field}"
                )
            if not isinstance(model[field], str) or not model[field]:
                raise BrickManagerError(
                    f"model field must be a non-empty string: {model_name}.{field}"
                )
        validate_model_component("ref", model["ref"])
