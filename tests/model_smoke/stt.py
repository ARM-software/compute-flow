#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed speech-to-text CLI scenarios."""

import re
from pathlib import Path

from .common import find_sample, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del output_dir
    sample = find_sample(package, (".wav",))
    result = run_cli([str(cli), "-m", str(manifest), "-a", str(sample)])
    transcription = re.search(
        r"\[TRANSCRIPTION\]\s*(.*?)\s*-{8,}", result.stdout, flags=re.DOTALL
    )
    if transcription is None or not transcription.group(1).strip():
        raise RuntimeError(
            f"Speech-to-text output has no transcription:\n{result.stdout}"
        )
