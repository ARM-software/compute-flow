#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

from ._compute_flow_py import *

# Public submodules
from . import brick


def load_brick(manifest):
    brick_info = manifest.get_brick_info()
    brick_type = getattr(brick_info, "type", None)
    brick_name = getattr(brick_info, "name", None)
    if brick_type is None or brick_name is None:
        try:
            brick_type, brick_name = brick_info
        except Exception as exc:
            raise ValueError("Manifest does not provide brick type/name info") from exc
    return brick.from_type_name(brick_type, brick_name)


__all__ = [
    # submodules
    "brick",
    "load_brick",
]
