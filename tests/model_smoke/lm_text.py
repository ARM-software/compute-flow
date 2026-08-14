#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed language-model CLI scenarios."""

from pathlib import Path

from .common import require_nonempty_text, run_cli


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del package
    output = output_dir / "generated.txt"
    run_cli(
        [
            str(cli),
            "-m",
            str(manifest),
            "-p",
            "Translate to English: Bonjour",
            "-n",
            "4",
            "-o",
            str(output),
        ]
    )
    require_nonempty_text(output)
