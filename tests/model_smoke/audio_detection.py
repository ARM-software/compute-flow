#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Model-backed audio-detection CLI scenarios."""

from pathlib import Path

from .common import find_sample, load_runs, require_run_markers, run_cli, write_runs


def run(cli: Path, manifest: Path, package: Path, output_dir: Path) -> None:
    sample = find_sample(package, (".wav",))
    runs = load_runs(Path(__file__).with_name("audio_detection_runs.json"))
    for run_config in runs:
        run_config["a"] = str(sample)

    runs_path = output_dir / "audio_detection_runs.json"
    write_runs(runs_path, runs)
    result = run_cli([str(cli), "-m", str(manifest), "-j", str(runs_path)])
    require_run_markers(result.stdout, runs)
    if result.stdout.count("Timestamp [") < len(runs):
        raise RuntimeError(
            f"Audio detection output has too few timestamped detections:\n{result.stdout}"
        )
