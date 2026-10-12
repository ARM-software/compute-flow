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
    BrickSelector,
    brick_manager_name,
    validate_brick_component,
    validate_brick_class_name,
    validate_model_component,
    validate_version,
)


JsonObject = dict[str, Any]

_REQUIRED_FIELDS = ("name", "type", "description", "author", "version", "models")
_MODEL_SOURCE_KINDS = frozenset(("git", "huggingface", "local"))
_MODEL_SOURCE_FIELDS = frozenset(("kind", "url", "ref"))


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


def load_installed_info_json(
    source_path: Path,
    selector: BrickSelector,
) -> JsonObject:
    """Load metadata from an installed brick and verify its identity."""
    info_json = load_info_json(source_path)
    installed_selector = BrickSelector(
        brick_type=info_json["type"],
        name=brick_manager_name(info_json["name"]),
        version=info_json["version"],
    )
    if installed_selector != selector:
        raise BrickManagerError(
            "Installed brick metadata identity does not match its install state: "
            f"expected {selector.identity}, got {installed_selector.identity}"
        )
    return info_json


def load_installed_entry_info_json(
    entry: JsonObject,
    selector: BrickSelector,
) -> JsonObject:
    """Load installed metadata and verify it against the recorded snapshot."""
    metadata = entry["metadata"]
    source_path = Path(metadata["installed_source_path"])
    info_json = load_installed_info_json(source_path, selector)
    installed_snapshot = metadata.get("installed_info_json")
    if installed_snapshot is not None and info_json != installed_snapshot:
        raise BrickManagerError(
            f"Installed brick metadata changed after {selector.identity} was installed. "
            f"Run `compute-flow brick install {selector.identity} --force` to replace it."
        )
    return installed_snapshot or info_json


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
        for field in ("description", "sha256"):
            if field not in model:
                raise BrickManagerError(
                    f"model entry is missing required field: {model_name}.{field}"
                )
            if not isinstance(model[field], str) or not model[field]:
                raise BrickManagerError(
                    f"model field must be a non-empty string: {model_name}.{field}"
                )

        if "sources" not in model:
            raise BrickManagerError(
                f"model entry is missing required field: {model_name}.sources"
            )
        _validate_model_sources(model_name, model["sources"])


def _validate_model_sources(model_name: str, sources: Any) -> None:
    if not isinstance(sources, list) or not sources:
        raise BrickManagerError(
            f"model field must be a non-empty list: {model_name}.sources"
        )

    for index, source in enumerate(sources):
        source_name = f"{model_name}.sources[{index}]"
        if not isinstance(source, dict):
            raise BrickManagerError(f"model source must be an object: {source_name}")

        unexpected_fields = source.keys() - _MODEL_SOURCE_FIELDS
        if unexpected_fields:
            fields = ", ".join(sorted(unexpected_fields))
            raise BrickManagerError(
                f"model source has unsupported fields: {source_name}: {fields}"
            )

        for field in ("kind", "url"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise BrickManagerError(
                    f"model source field must be a non-empty string: "
                    f"{source_name}.{field}"
                )

        kind = source["kind"]
        if kind not in _MODEL_SOURCE_KINDS:
            kinds = ", ".join(sorted(_MODEL_SOURCE_KINDS))
            raise BrickManagerError(
                f"unsupported model source kind: {source_name}.kind={kind}. "
                f"Expected one of: {kinds}"
            )

        ref = source.get("ref")
        if kind == "local":
            if ref is not None:
                _validate_source_ref(source_name, ref)
        else:
            _validate_source_ref(source_name, ref)


def _validate_source_ref(source_name: str, ref: Any) -> None:
    if not isinstance(ref, str) or not ref:
        raise BrickManagerError(
            f"model source field must be a non-empty string: {source_name}.ref"
        )
    validate_model_component("ref", ref)
