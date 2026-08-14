#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run model-backed CLI smoke tests and validate their output structure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.identity import brick_manager_name  # noqa: E402
from model_smoke import RUNNERS  # noqa: E402


def find_manifest(package: Path) -> Path:
    for candidate in sorted(package.glob("*.json")):
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and isinstance(document.get("brick"), dict):
            return candidate
    raise RuntimeError(f"No model manifest found in package: {package}")


def installed_model_path(model_home: Path, brick: str, model: str) -> Path:
    cache_path = model_home / "cache.json"
    if not cache_path.is_file():
        raise RuntimeError(f"Brick cache is unavailable: {cache_path}")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    versions = cache.get("bricks", {}).get(brick, {})
    for entry in versions.values():
        metadata = entry.get("metadata", {})
        if not metadata.get("installed", False):
            continue
        installed_model = metadata.get("installed_models", {}).get(model)
        if installed_model is None:
            break
        model_path = Path(installed_model["path"])
        if model_path.is_dir():
            return model_path
        break
    raise RuntimeError(f"Model installation fixture did not install: {brick} {model}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick-type", required=True)
    parser.add_argument("--brick-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-home", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        runner = RUNNERS.get(arguments.brick_type)
        if runner is None:
            raise RuntimeError(
                "No model-backed CLI smoke runner for brick type: "
                f"{arguments.brick_type}"
            )

        brick = f"{arguments.brick_type}:{brick_manager_name(arguments.brick_name)}"
        package = installed_model_path(arguments.model_home, brick, arguments.model)
        manifest = find_manifest(package)
        with tempfile.TemporaryDirectory(
            prefix=f"compute-flow-{arguments.model}-"
        ) as temporary_directory:
            runner(
                arguments.cli,
                manifest,
                package,
                Path(temporary_directory),
            )
        print(f"Validated model-backed CLI output for {brick} {arguments.model}")
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Model-backed CLI smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
