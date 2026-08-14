#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed audio-generation CLI scenarios."""

from pathlib import Path

from .common import require_wav, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del package
    output = output_dir / "generated.wav"
    run_cli([str(cli), "-m", str(manifest), "-p", "A short bell.", "-o", str(output)])
    require_wav(output)
