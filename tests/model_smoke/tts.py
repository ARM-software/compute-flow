#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed text-to-speech CLI scenarios."""

from pathlib import Path

from .common import load_runs, require_run_markers, require_wav, run_cli, write_runs


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    del package
    runs = load_runs(Path(__file__).with_name("tts_runs.json"))
    outputs = []
    for run_config in runs:
        output = output_dir / str(run_config["o"])
        run_config["o"] = str(output)
        outputs.append(output)

    runs_path = output_dir / "tts_runs.json"
    write_runs(runs_path, runs)
    result = run_cli([str(cli), "-m", str(manifest), "-j", str(runs_path)])
    require_run_markers(result.stdout, runs)
    for output in outputs:
        require_wav(output)
