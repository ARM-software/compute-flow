#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Versioned CPU feature catalog for brick custom operators."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


FEATURE_CATALOG_VERSION = 1


@dataclass(frozen=True)
class BuildFeature:
    """AArch64 architecture level and extension needed by a compiler."""

    architecture: str
    extension: str


# The processor target in registration.yaml is "arm". Version 1 provides
# compiler settings for its AArch64 kernels; toolchain support is checked at build time.
HW_FEATURES: dict[str, frozenset[str]] = {
    "arm": frozenset(
        {"neon", "neon-dotprod", "neon-fp16", "neon-i8mm", "sme", "sme2", "sve", "sve2"}
    ),
}
BUILD_FEATURES: dict[str, dict[str, BuildFeature]] = {
    "arm": {
        "neon-dotprod": BuildFeature("armv8.2-a", "dotprod"),
        "neon-fp16": BuildFeature("armv8.2-a", "fp16"),
        "neon-i8mm": BuildFeature("armv8.2-a", "i8mm"),
        "sme": BuildFeature("armv9.2-a", "sme"),
        "sme2": BuildFeature("armv9.2-a", "sme2"),
        "sve": BuildFeature("armv8.2-a", "sve"),
        "sve2": BuildFeature("armv8.2-a", "sve2"),
    },
}
_ARCHITECTURE_RANK = {"armv8.2-a": 0, "armv9.2-a": 1}
_MARCH_COMPILERS = frozenset({"GNU", "Clang", "AppleClang"})


def compiler_flags(
    target: str, features: Iterable[str], compiler: str
) -> tuple[str, ...]:
    """Return one architecture flag for supported AArch64 build features."""
    if target not in BUILD_FEATURES:
        raise ValueError(f"Unsupported custom operator target: {target}")

    selected: list[BuildFeature] = []
    for name in features:
        try:
            selected.append(BUILD_FEATURES[target][name])
        except KeyError as error:
            raise ValueError(f"Unsupported {target} build feature: {name}") from error
    if not selected:
        return ()
    if compiler not in _MARCH_COMPILERS:
        raise ValueError(f"Unsupported custom operator compiler: {compiler}")

    architecture = max(
        selected, key=lambda feature: _ARCHITECTURE_RANK[feature.architecture]
    ).architecture
    extensions = "+".join(sorted({feature.extension for feature in selected}))
    return (f"-march={architecture}+{extensions}",)
