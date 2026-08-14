#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Install every model package declared by Compute Flow bricks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from model_smoke_cases import MODEL_SMOKE_CASES, validate_cases


def install_models(
    brick_manager: Path, model_home: Path, repository_root: Path
) -> None:
    sys.path.insert(0, str(repository_root / "compute_flow" / "python"))
    from brick_manager.identity import brick_manager_name

    environment = os.environ.copy()
    environment["COMPUTE_FLOW_HOME"] = str(model_home)
    validate_cases(repository_root)

    for brick_type, brick_name, model_name in MODEL_SMOKE_CASES:
        brick = f"{brick_type}:{brick_manager_name(brick_name)}"
        print(f"Installing {model_name} for {brick}", flush=True)
        subprocess.run(
            [str(brick_manager), "model", "install", brick, model_name],
            check=True,
            env=environment,
        )

    print(
        f"Installed or verified {len(MODEL_SMOKE_CASES)} model packages",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick-manager", type=Path, required=True)
    parser.add_argument("--model-home", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        install_models(
            arguments.brick_manager,
            arguments.model_home,
            arguments.repository_root,
        )
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Model installation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
