#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed voice-activity-detection CLI scenarios."""

from pathlib import Path

from .common import find_sample, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del output_dir
    sample = find_sample(package, (".wav",))
    result = run_cli([str(cli), "-m", str(manifest), "-a", str(sample)])
    if "Timestamp [" not in result.stdout:
        raise RuntimeError(f"VAD output has no timestamped decisions:\n{result.stdout}")
