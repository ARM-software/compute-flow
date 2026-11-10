#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Validate custom operator declarations in a brick source tree."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

import yaml

from brick_manager.custom_ops.features import BUILD_FEATURES, HW_FEATURES
from brick_manager.errors import BrickManagerError


_OPERATOR_NAME = re.compile(r"[a-z][a-z0-9_]*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TENSOR_DTYPES = frozenset(
    {
        "bool",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
    }
)
_MAX_TENSOR_RANK = 8
_ASSEMBLY_TOKEN = re.compile(r"\b(?:asm|__asm|__asm__|_asm|_emit|__emit)\b")
_TYPES_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*(?:"compute_flow/custom_ops/v1/types\.h"|'
    r"<compute_flow/custom_ops/v1/types\.h>)",
    re.MULTILINE,
)
_C_COMMENT_OR_LITERAL = re.compile(
    r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
)


def validate_custom_ops(brick_path: Path) -> list[dict[str, object]]:
    """Validate schema and layout, then return custom operator declarations."""
    root = brick_path / "custom_ops"
    # Bricks without a custom_ops folder have no operators to validate.
    if not root.exists() and not root.is_symlink():
        return []
    if not root.is_dir() or root.is_symlink():
        raise BrickManagerError(f"custom_ops must be a directory: {root}")

    schema_path = root / "schema.yaml"
    if not schema_path.is_file() or schema_path.is_symlink():
        raise BrickManagerError(f"Missing custom operator schema: {schema_path}")
    try:
        schema: object = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise BrickManagerError(
            f"Invalid custom operator schema: {schema_path}: {error}"
        ) from error

    # Check the declared model tensor contract before inspecting source files.
    schema_map = _mapping(schema, "schema")
    _fields(schema_map, {"operators"}, set(), "schema")
    operators = schema_map["operators"]
    if not isinstance(operators, list) or not operators:
        raise BrickManagerError(
            "custom_ops/schema.yaml: operators must be a non-empty list"
        )

    names: set[str] = set()
    declarations: list[dict[str, object]] = []
    for index, operator in enumerate(operators):
        label = f"operators[{index}]"
        declaration = _mapping(operator, label)
        _fields(
            declaration,
            {"name", "version", "inputs", "outputs"},
            set(),
            label,
        )
        name = declaration["name"]
        if not isinstance(name, str) or not _OPERATOR_NAME.fullmatch(name):
            raise BrickManagerError(f"{label}.name must be a lowercase operator name")
        if name in names:
            raise BrickManagerError(f"Duplicate custom operator name: {name}")
        names.add(name)
        declarations.append(declaration)

        version = declaration["version"]
        if type(version) is not int or version < 1:
            raise BrickManagerError(f"{label}.version must be a positive integer")
        _tensors(declaration["inputs"], f"{label}.inputs", allow_empty=True)
        _tensors(declaration["outputs"], f"{label}.outputs", allow_empty=False)

        # Reference is the fallback, common is shared by all kernels, and
        # optimized kernels are declared under cpu/.
        operator_path = root / name
        if not operator_path.is_dir() or operator_path.is_symlink():
            raise BrickManagerError(
                f"Missing custom operator directory: {operator_path}"
            )
        reference_source = operator_path / "reference.c"
        if not reference_source.is_file() or reference_source.is_symlink():
            raise BrickManagerError(
                f"Missing custom operator reference source: {reference_source}"
            )
        common_source = operator_path / "calculate_output_shapes.c"
        if not common_source.is_file() or common_source.is_symlink():
            raise BrickManagerError(
                f"Missing custom operator output shape source: {common_source}"
            )
        _validate_reference_and_common_sources(
            reference_source=reference_source,
            common_source=common_source,
        )
        cpu_path = operator_path / "cpu"
        registration = cpu_path / "registration.yaml"
        if (cpu_path.exists() or cpu_path.is_symlink()) and (
            not cpu_path.is_dir()
            or cpu_path.is_symlink()
            or not registration.is_file()
            or registration.is_symlink()
        ):
            raise BrickManagerError(f"Invalid custom operator CPU layout: {cpu_path}")
        if cpu_path.is_dir():
            kernels = load_optimized_kernels(
                cpu_path, name, version, len(declaration["inputs"])
            )
            _validate_optimized_sources(cpu_path, kernels)

        allowed_entries = {reference_source, common_source}
        if cpu_path.is_dir():
            allowed_entries.add(cpu_path)
        for entry in sorted(operator_path.iterdir()):
            if entry not in allowed_entries:
                raise BrickManagerError(f"Undeclared custom operator entry: {entry}")

    # Reject files and folders that the schema does not account for.
    for entry in root.iterdir():
        if entry.name != "schema.yaml" and entry.name not in names:
            raise BrickManagerError(f"Undeclared custom operator entry: {entry}")

    return sorted(declarations, key=lambda declaration: str(declaration["name"]))


def load_optimized_kernels(
    cpu_path: Path, operator_name: str, operator_version: int, num_inputs: int
) -> list[dict[str, object]]:
    """Read and validate optimized C kernels and their optional assembly."""
    # Read the registration before using any source paths.
    registration = cpu_path / "registration.yaml"
    try:
        value: object = yaml.safe_load(registration.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise BrickManagerError(
            f"Invalid custom operator registration: {registration}: {error}"
        ) from error

    if value is None:
        kernels = []
    else:
        # Match the operator declared in schema.yaml.
        if not isinstance(value, dict) or set(value) != {"name", "version", "kernels"}:
            raise BrickManagerError(
                f"Invalid custom operator registration: {registration}"
            )

        if (
            value["name"] != operator_name
            or type(value["version"]) is not int
            or value["version"] != operator_version
        ):
            raise BrickManagerError(
                f"Custom operator registration does not match {operator_name}"
            )

        kernels = value["kernels"]
        if not isinstance(kernels, list):
            raise BrickManagerError(
                f"Custom operator kernels must be a list: {registration}"
            )

    # Check every declared source and its feature names.
    seen: set[str] = set()
    allowed_entries = {registration}
    for kernel in kernels:
        required_fields = {"kernel", "target", "hw_features", "build_features"}
        if not isinstance(kernel, dict) or not (
            required_fields <= set(kernel)
            and set(kernel) <= required_fields | {"setup_inputs"}
        ):
            raise BrickManagerError(
                f"Invalid optimized kernel declaration: {registration}"
            )

        stem = kernel["kernel"]
        target = kernel["target"]
        if (
            not isinstance(stem, str)
            or not isinstance(target, str)
            or target not in BUILD_FEATURES
            or not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", stem)
            or stem.split("/", maxsplit=1)[0] != target
            or stem in seen
        ):
            raise BrickManagerError(f"Invalid optimized kernel identity: {stem}")

        seen.add(stem)

        for field, catalog in (
            ("hw_features", HW_FEATURES),
            ("build_features", BUILD_FEATURES),
        ):
            if not _valid_features(kernel[field], catalog[target]):
                raise BrickManagerError(f"Invalid {field} for optimized kernel: {stem}")

        setup_inputs = kernel.get("setup_inputs", [])
        if (
            not isinstance(setup_inputs, list)
            or any(
                type(index) is not int or index < 0 or index >= num_inputs
                for index in setup_inputs
            )
            or len(setup_inputs) != len(set(setup_inputs))
        ):
            raise BrickManagerError(
                f"Invalid setup_inputs for optimized kernel: {stem}"
            )

        source = cpu_path / f"{stem}.c"
        if not source.is_file() or source.is_symlink() or source.parent.is_symlink():
            raise BrickManagerError(f"Missing optimized C source: {source}")
        allowed_entries.update({source.parent, source})
        assembly = cpu_path / f"{stem}.S"
        if assembly.is_symlink() or (assembly.exists() and not assembly.is_file()):
            raise BrickManagerError(f"Invalid optimized assembly source: {assembly}")
        if assembly.is_file():
            allowed_entries.add(assembly)

    for entry in sorted(cpu_path.rglob("*")):
        if entry not in allowed_entries:
            raise BrickManagerError(f"Undeclared custom operator entry: {entry}")

    return kernels


def _valid_features(value: object, allowed: Collection[str]) -> bool:
    """Check feature names and reject duplicates."""
    if not isinstance(value, list):
        return False

    if any(not isinstance(feature, str) or feature not in allowed for feature in value):
        return False

    return len(value) == len(set(value))


def _validate_reference_and_common_sources(
    *, reference_source: Path, common_source: Path
) -> None:
    """Check portable reference and output shape sources."""
    for path in (reference_source, common_source):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BrickManagerError(
                f"Cannot read custom operator source: {path}: {error}"
            ) from error
        code = _C_COMMENT_OR_LITERAL.sub(" ", source)
        if _ASSEMBLY_TOKEN.search(code):
            raise BrickManagerError(f"Assembly is not allowed outside cpu/: {path}")
        uncommented = _C_COMMENT_OR_LITERAL.sub(
            lambda match: " " if match.group("comment") else match.group(), source
        )
        if not _TYPES_INCLUDE.search(uncommented):
            raise BrickManagerError(
                f"Custom operator source must include "
                f"compute_flow/custom_ops/v1/types.h: {path}"
            )


def _validate_optimized_sources(
    cpu_path: Path, kernels: list[dict[str, object]]
) -> None:
    """Reject inline assembly in declared optimized C sources."""
    for kernel in kernels:
        path = cpu_path / f"{kernel['kernel']}.c"
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BrickManagerError(
                f"Cannot read optimized source: {path}: {error}"
            ) from error
        code = _C_COMMENT_OR_LITERAL.sub(" ", source)
        if _ASSEMBLY_TOKEN.search(code):
            raise BrickManagerError(
                f"Inline assembly is not allowed in an optimized source: {path}"
            )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BrickManagerError(f"custom_ops/schema.yaml: {label} must be a mapping")
    return value


def _fields(
    value: dict[str, object], required: set[str], optional: set[str], label: str
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise BrickManagerError(
            f"custom_ops/schema.yaml: {label} has missing fields {sorted(missing)} "
            f"or unknown fields {sorted(unknown)}"
        )


def _tensors(value: object, label: str, *, allow_empty: bool) -> None:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise BrickManagerError(f"{label} must be a list of tensors")
    names: set[str] = set()
    for index, tensor in enumerate(value):
        tensor_label = f"{label}[{index}]"
        declaration = _mapping(tensor, tensor_label)
        _fields(declaration, {"name", "dtype", "shape"}, set(), tensor_label)
        name = declaration["name"]
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise BrickManagerError(f"{tensor_label}.name must be an identifier")
        if name in names:
            raise BrickManagerError(f"Duplicate tensor name in {label}: {name}")
        names.add(name)
        dtype = declaration["dtype"]
        if not isinstance(dtype, str) or dtype not in _TENSOR_DTYPES:
            raise BrickManagerError(f"{tensor_label}.dtype is unsupported")
        shape = declaration["shape"]
        if not isinstance(shape, list) or not 1 <= len(shape) <= _MAX_TENSOR_RANK:
            raise BrickManagerError(f"{tensor_label}.shape must have 1 to 8 dimensions")
        for dimension in shape:
            if type(dimension) is int and dimension >= 0:
                continue
            if isinstance(dimension, str) and _IDENTIFIER.fullmatch(dimension):
                continue
            raise BrickManagerError(f"{tensor_label}.shape has an invalid dimension")
