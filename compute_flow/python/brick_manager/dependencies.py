#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Compute Flow-managed brick dependency metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brick_manager.errors import BrickManagerError


@dataclass(frozen=True)
class BrickDependency:
    cmake_target: str
    cmake_package: str
    cmake_module: str | None = None
    cmake_source_variable: str | None = None
    extra_include_subdirs: tuple[str, ...] = ()


_DEPENDENCIES = {
    "espeakng": BrickDependency(
        cmake_target="espeak-ng",
        cmake_package="espeakng",
        cmake_module="cmake/deps/espeakng/EspeakNg.cmake",
        cmake_source_variable="espeakng_SOURCE_DIR",
        extra_include_subdirs=("src/ucd-tools/src/include",),
    ),
    "libnpy": BrickDependency(
        cmake_target="libnpy::libnpy",
        cmake_package="libnpy",
        cmake_module="cmake/deps/libnpy/Libnpy.cmake",
    ),
    "nlohmann_json": BrickDependency(
        cmake_target="nlohmann_json::nlohmann_json",
        cmake_package="nlohmann_json",
        cmake_module="cmake/deps/nlohmann_json/NlohmannJson.cmake",
    ),
}


def known_dependency_names() -> tuple[str, ...]:
    return tuple(sorted(_DEPENDENCIES))


def dependency_metadata(name: str) -> BrickDependency:
    try:
        return _DEPENDENCIES[name]
    except KeyError as error:
        allowed = ", ".join(known_dependency_names())
        raise BrickManagerError(
            f"Unknown Compute Flow dependency: {name}. Allowed: {allowed}"
        ) from error


def dependency_cmake_targets(names: list[str]) -> list[str]:
    return [dependency_metadata(name).cmake_target for name in names]


def validate_dependencies_field(deps: Any) -> list[str]:
    if not isinstance(deps, list):
        raise BrickManagerError("info.json field must be a list: deps")

    seen: set[str] = set()
    validated = []
    for index, dep in enumerate(deps):
        if not isinstance(dep, str) or not dep:
            raise BrickManagerError(
                f"info.json deps entry must be a non-empty string: deps[{index}]"
            )
        if dep in seen:
            raise BrickManagerError(f"Duplicate Compute Flow dependency: {dep}")
        dependency_metadata(dep)
        seen.add(dep)
        validated.append(dep)

    return validated
