#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""State path helpers for the Compute Flow brick manager."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from brick_manager.identity import BrickSelector


@dataclass(frozen=True)
class StatePaths:
    home: Path

    @property
    def cache(self) -> Path:
        return self.home / "cache.json"

    @property
    def bricks(self) -> Path:
        return self.home / "bricks"

    @property
    def builds(self) -> Path:
        return self.home / "builds"

    @property
    def models(self) -> Path:
        return self.home / "models"

    def brick_source(self, selector: BrickSelector) -> Path:
        if selector.version is None:
            raise ValueError("brick source paths require a version")
        return self.bricks / selector.brick_type / selector.name / selector.version


def resolve_state_paths() -> StatePaths:
    compute_flow_home = os.environ.get("COMPUTE_FLOW_HOME")
    if compute_flow_home:
        return StatePaths(home=Path(compute_flow_home).expanduser())
    return StatePaths(home=Path.home() / ".compute-flow")
