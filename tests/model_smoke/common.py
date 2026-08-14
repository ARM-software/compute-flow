#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Shared helpers for model-backed CLI smoke runners."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def run_cli(
    arguments: list[str], timeout: int = 600
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        command = " ".join(arguments)
        raise RuntimeError(
            f"Command failed with status {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def require_wav(path: Path) -> None:
    data = path.read_bytes()
    if len(data) <= 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError(f"Invalid or empty WAV output: {path}")
    data_offset = data.find(b"data", 12)
    if data_offset < 0 or data_offset + 8 > len(data):
        raise RuntimeError(f"WAV output has no data chunk: {path}")
    data_size = struct.unpack_from("<I", data, data_offset + 4)[0]
    if data_size == 0 or data_offset + 8 + data_size > len(data):
        raise RuntimeError(f"WAV output has an invalid data chunk: {path}")


def require_png(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG output: {path}")
    width, height = struct.unpack_from(">II", data, 16)
    if width == 0 or height == 0:
        raise RuntimeError(f"PNG output has invalid dimensions: {path}")


def require_nonempty_text(path: Path) -> None:
    if (
        not path.is_file()
        or not path.read_text(encoding="utf-8", errors="replace").strip()
    ):
        raise RuntimeError(f"Empty text output: {path}")


def find_sample(package: Path, suffixes: tuple[str, ...]) -> Path:
    candidates = sorted(
        path
        for path in (package / "samples").glob("**/*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not candidates:
        suffix_list = ", ".join(suffixes)
        raise RuntimeError(
            f"No sample with one of these suffixes ({suffix_list}) in {package}"
        )

    generated_markers = ("enhanced", "expected", "output", "relative_depth")
    return next(
        (
            path
            for path in candidates
            if not any(marker in path.stem.lower() for marker in generated_markers)
        ),
        candidates[0],
    )


def load_runs(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or not document:
        raise RuntimeError(f"Run definitions must be a non-empty JSON array: {path}")

    runs = []
    for index, run in enumerate(document):
        if not isinstance(run, dict):
            raise RuntimeError(f"Run definition {index + 1} must be an object: {path}")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError(
                f"Run definition {index + 1} has no non-empty id: {path}"
            )
        runs.append(run.copy())
    return runs


def write_runs(path: Path, runs: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")


def require_run_markers(output: str, runs: list[dict[str, object]]) -> None:
    for run in runs:
        marker = f"Run {run['id']}"
        if marker not in output:
            raise RuntimeError(f"CLI output has no marker for {marker}:\n{output}")
