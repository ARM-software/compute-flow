#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Brick and model identity parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from brick_manager.errors import BrickManagerError


_BRICK_TYPE_PATTERN = re.compile(r"^[a-z0-9_][a-z0-9_-]*$")
_BRICK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_BRICK_CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_MODEL_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")


@dataclass(frozen=True)
class BrickSelector:
    brick_type: str
    name: str
    version: str | None = None

    @property
    def key(self) -> str:
        return f"{self.brick_type}:{self.name}"

    @property
    def identity(self) -> str:
        if self.version is None:
            return self.key
        return f"{self.key}@{self.version}"


def parse_brick_selector(value: str, *, require_version: bool = False) -> BrickSelector:
    key, separator, version = value.partition("@")
    if require_version and not separator:
        raise BrickManagerError(
            "Expected brick identifier in the form <type>:<name>@<version>"
        )
    if separator and not version:
        raise BrickManagerError("Brick version cannot be empty")

    brick_type, name_separator, name = key.partition(":")
    if not name_separator:
        raise BrickManagerError(
            "Expected brick identifier in the form <type>:<name>[@version]"
        )

    validate_brick_component("type", brick_type)
    validate_brick_component("name", name)
    if version:
        validate_version(version)

    return BrickSelector(brick_type=brick_type, name=name, version=version or None)


def validate_brick_component(label: str, value: str) -> None:
    pattern = _BRICK_TYPE_PATTERN if label == "type" else _BRICK_NAME_PATTERN
    if not pattern.fullmatch(value):
        raise BrickManagerError(f"Invalid brick {label}: {value}")


def validate_brick_class_name(value: str) -> None:
    if not _BRICK_CLASS_NAME_PATTERN.fullmatch(value):
        raise BrickManagerError(f"Invalid brick name: {value}")


def brick_manager_name(name: str) -> str:
    validate_brick_class_name(name)
    manager_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", name)
    manager_name = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", manager_name)
    manager_name = manager_name.replace("_", "-").lower()
    manager_name = re.sub(r"-+", "-", manager_name).strip("-")
    validate_brick_component("name", manager_name)
    return manager_name


def validate_version(value: str) -> None:
    if not _VERSION_PATTERN.fullmatch(value) or ".." in value:
        raise BrickManagerError(f"Invalid brick version: {value}")


def validate_model_component(label: str, value: str) -> None:
    if not _MODEL_COMPONENT_PATTERN.fullmatch(value) or ".." in value:
        raise BrickManagerError(f"Invalid model {label}: {value}")
