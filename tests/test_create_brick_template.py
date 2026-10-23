#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Smoke tests for the brick starter generator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_brick_template.py"


class CreateBrickTemplateTest(unittest.TestCase):
    def test_generated_brick_has_consistent_identity_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "audio_detection",
                    "Nemotron3Diarization",
                    "--output-root",
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            brick = root / "audio_detection" / "nemotron3_diarization"
            self.assertEqual(
                {path.name for path in brick.iterdir()},
                {
                    "Nemotron3Diarization.cpp",
                    "Nemotron3Diarization.h",
                    "README.md",
                    "info.json",
                },
            )
            info = json.loads((brick / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                (info["name"], info["type"], info["api"]),
                ("Nemotron3Diarization", "audio_detection", "audio_detection.v1"),
            )
            header = (brick / "Nemotron3Diarization.h").read_text(encoding="utf-8")
            source = (brick / "Nemotron3Diarization.cpp").read_text(encoding="utf-8")
            self.assertIn("class Nemotron3Diarization : public BrickBaseV1", header)
            self.assertIn('return "audio_detection.v1"', header)
            self.assertIn("bool _is_eog = false;", header)
            self.assertIn(
                '// COMPUTE_FLOW_VALIDATE(manifest.has_module("model"))', source
            )
            self.assertIn('// _runner->setup(manifest["model"], profiler);', source)
            self.assertIn("run_impl(input, profiler);", source)
            self.assertIn("TODO: implement model-specific inference", source)
            self.assertIn(
                f"Copyright {date.today().year} Arm Limited",
                header,
            )

            second_run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "audio_detection",
                    "Nemotron3Diarization",
                    "--output-root",
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(second_run.returncode, 0)
            self.assertIn("already exists", second_run.stderr)
            self.assertEqual(
                json.loads((brick / "info.json").read_text())["name"], info["name"]
            )

    def test_invalid_identity_does_not_create_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "audio_detection",
                    "../Invalid",
                    "--output-root",
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(list(root.iterdir()))


if __name__ == "__main__":
    unittest.main()
