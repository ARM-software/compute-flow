#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Format Compute Flow manifest JSON files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ROOT_ORDER = ["brick", "models", "assets", "configs", "constants"]
MODEL_ORDER = [
    "runner",
    "filename",
    "inputs",
    "outputs",
    "states",
    "runner_params",
    "hw_hint",
]
TENSOR_ORDER = ["index", "index_in", "index_out", "dtype", "shape"]
TYPED_VALUE_ORDER = ["type", "value"]
TENSOR_KEYS = set(TENSOR_ORDER)
TYPED_VALUE_KEYS = set(TYPED_VALUE_ORDER)


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def is_scalar_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and all(is_scalar(item) for item in value)
    )


def json_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def ordered_mapping(
    data: Mapping[str, Any], preferred_order: Sequence[str]
) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in data:
            ordered[key] = data[key]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def normalized_posix_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def model_root_filename(path: str) -> str:
    filename = PurePosixPath(normalized_posix_path(path)).name
    return filename or path


def asset_path(path: str) -> str:
    normalized = normalized_posix_path(path)
    if normalized == "assets" or normalized.startswith("assets/"):
        return normalized
    filename = PurePosixPath(normalized).name
    return f"assets/{filename}" if filename else path


def normalize_asset_entry(value: Any) -> Any:
    if isinstance(value, str):
        return asset_path(value)
    if isinstance(value, Mapping):
        return {
            key: asset_path(item)
            if key in {"filename", "path", "file"} and isinstance(item, str)
            else normalize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_asset_entry(item) for item in value]
    return value


def normalize_model(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return normalize_value(value)

    model = ordered_mapping(value, MODEL_ORDER)
    if isinstance(model.get("filename"), str):
        model["filename"] = model_root_filename(model["filename"])
    return {key: normalize_value(item) for key, item in model.items()}


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    keys = set(value)
    if keys.issubset(TENSOR_KEYS) and "shape" in value:
        return ordered_mapping(value, TENSOR_ORDER)
    if keys.issubset(TYPED_VALUE_KEYS) and "type" in value and "value" in value:
        return ordered_mapping(value, TYPED_VALUE_ORDER)
    return {key: normalize_value(item) for key, item in value.items()}


def normalize_manifest(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        msg = "manifest root must be a JSON object"
        raise ValueError(msg)

    manifest = ordered_mapping(data, ROOT_ORDER)

    models = manifest.get("models")
    if isinstance(models, Mapping):
        manifest["models"] = {
            name: normalize_model(model) for name, model in models.items()
        }

    assets = manifest.get("assets")
    if isinstance(assets, Mapping):
        manifest["assets"] = {
            name: normalize_asset_entry(asset) for name, asset in assets.items()
        }

    return {key: normalize_value(value) for key, value in manifest.items()}


def render_list(value: Sequence[Any], indent: int, parent_key: str | None) -> str:
    if parent_key == "shape" and is_scalar_sequence(value):
        return "[" + ", ".join(json_scalar(item) for item in value) + "]"
    if not value:
        return "[]"

    child_indent = "  " * (indent + 1)
    current_indent = "  " * indent
    lines = ["["]
    for index, item in enumerate(value):
        suffix = "," if index + 1 < len(value) else ""
        lines.append(f"{child_indent}{render_json(item, indent + 1)}{suffix}")
    lines.append(f"{current_indent}]")
    return "\n".join(lines)


def render_mapping(value: Mapping[str, Any], indent: int) -> str:
    if not value:
        return "{}"

    child_indent = "  " * (indent + 1)
    current_indent = "  " * indent
    lines = ["{"]
    items = list(value.items())
    for index, (key, item) in enumerate(items):
        suffix = "," if index + 1 < len(items) else ""
        lines.append(
            f"{child_indent}{json_scalar(key)}: {render_json(item, indent + 1, key)}{suffix}"
        )
    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


def render_json(value: Any, indent: int = 0, parent_key: str | None = None) -> str:
    if is_scalar(value):
        return json_scalar(value)
    if isinstance(value, Mapping):
        return render_mapping(value, indent)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return render_list(value, indent, parent_key)

    msg = f"unsupported JSON value: {type(value).__name__}"
    raise TypeError(msg)


def format_manifest(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return render_json(normalize_manifest(data)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="check formatting without rewriting files"
    )
    parser.add_argument(
        "manifest", type=Path, nargs="+", help="manifest JSON file to format"
    )
    args = parser.parse_args()

    failed = False
    for manifest in args.manifest:
        formatted = format_manifest(manifest)
        current = manifest.read_text(encoding="utf-8")
        if args.check:
            if formatted != current:
                print(f"{manifest}: manifest is not formatted")
                failed = True
            continue
        if formatted != current:
            manifest.write_text(formatted, encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
