#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def from_type_name(brick_type: str, brick_name: str):
    if not brick_type:
        raise ValueError("Brick type cannot be empty")
    if not brick_name:
        raise ValueError("Brick name cannot be empty")

    module = importlib.import_module(f"{brick_name}_py")
    try:
        cls = getattr(module, brick_name)
    except AttributeError as exc:
        raise ValueError(
            f"Python binding module '{brick_name}_py' does not expose "
            f"brick class '{brick_name}'"
        ) from exc
    return cls()
