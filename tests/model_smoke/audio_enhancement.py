#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed audio-enhancement CLI scenarios."""

from pathlib import Path

from .common import find_sample, require_wav, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    sample = find_sample(package, (".wav",))
    output = output_dir / "enhanced.wav"
    run_cli([str(cli), "-m", str(manifest), "-a", str(sample), "-o", str(output)])
    require_wav(output)
