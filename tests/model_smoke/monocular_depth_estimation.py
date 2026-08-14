#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed monocular-depth-estimation CLI scenarios."""

from pathlib import Path

from .common import IMAGE_SUFFIXES, find_sample, require_png, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    sample = find_sample(package, IMAGE_SUFFIXES)
    output = output_dir / "relative_depth.png"
    run_cli([str(cli), "-m", str(manifest), "-i", str(sample), "-o", str(output)])
    require_png(output)
