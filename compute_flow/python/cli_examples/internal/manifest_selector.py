#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Manifest path helpers for Compute Flow brick CLIs."""

from __future__ import annotations

from pathlib import Path

from compute_flow_py import get_manifest_path

__all__ = ["get_manifest_path", "get_sample_path"]

_SAMPLE_SELECTOR_PREFIX = "@sample/"


def get_sample_path(value: str, manifest_path: str) -> str:
    if not value.startswith("@"):
        return value

    if not value.startswith(_SAMPLE_SELECTOR_PREFIX):
        raise ValueError(
            f"Invalid sample value: {value}\nExpected a path or @sample/<path>"
        )

    relative_path = Path(value.removeprefix(_SAMPLE_SELECTOR_PREFIX))
    if not relative_path.parts or relative_path.is_absolute():
        raise ValueError(f"Invalid sample selector: {value}\nExpected: @sample/<path>")
    if ".." in relative_path.parts:
        raise ValueError(
            f"Invalid sample selector: {value}\nPath traversal is not allowed"
        )

    sample_path = Path(manifest_path).parent / "samples" / relative_path
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Sample is not installed: {value}\nExpected file: {sample_path}"
        )

    return str(sample_path)
