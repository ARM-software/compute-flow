#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed object-detection CLI scenarios."""

from pathlib import Path

from .common import IMAGE_SUFFIXES, find_sample, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del output_dir
    sample = find_sample(package, IMAGE_SUFFIXES)
    result = run_cli([str(cli), "-m", str(manifest), "-i", str(sample)])
    if "Detected:" not in result.stdout:
        raise RuntimeError(
            f"Object detection output has no detections:\n{result.stdout}"
        )
