#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for Brick Manager catalogue filtering and completion."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# CTest runs this file directly, so expose the source package before importing it.
# The following imports intentionally need E402 suppressions because of this path setup.
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import write_cache  # noqa: E402
from brick_manager.catalog import get_brick_summaries  # noqa: E402
from brick_manager.completion import complete_cli  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402


class BrickCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.state_paths = StatePaths(Path(self.temporary_directory.name))
        write_cache(
            self.state_paths.cache,
            {
                "bricks": {
                    "stt:moonshine": {
                        "1.0.0": self._entry("Moonshine", installed=True)
                    },
                    "tts:pocket-tts": {"1.0.0": self._entry("Pocket TTS")},
                    "tts:qwen3-tts": {
                        "1.0.0": self._entry("Qwen3-TTS"),
                        "2.0.0": self._entry("Qwen3-TTS"),
                    },
                    "vad:silero-vad": {"1.0.0": self._entry("Silero VAD")},
                }
            },
        )

    def test_lists_every_variant_without_a_type_filter(self) -> None:
        summaries = get_brick_summaries(self.state_paths)

        self.assertEqual(len(summaries), 5)

    def test_filters_variants_by_one_type(self) -> None:
        summaries = get_brick_summaries(self.state_paths, ["tts"])

        self.assertEqual(
            [summary.brick for summary in summaries],
            ["tts:pocket-tts", "tts:qwen3-tts", "tts:qwen3-tts"],
        )

    def test_combines_variants_from_multiple_types(self) -> None:
        summaries = get_brick_summaries(self.state_paths, ["stt", "vad"])

        self.assertEqual(
            [summary.brick for summary in summaries],
            ["stt:moonshine", "vad:silero-vad"],
        )

    def test_filters_installed_variants(self) -> None:
        summaries = get_brick_summaries(self.state_paths, installed_only=True)

        self.assertEqual(
            [(summary.brick, summary.version) for summary in summaries],
            [("stt:moonshine", "1.0.0")],
        )

    def test_combines_installed_and_type_filters(self) -> None:
        summaries = get_brick_summaries(self.state_paths, ["tts"], installed_only=True)

        self.assertEqual(summaries, [])

    def test_rejects_invalid_type(self) -> None:
        with self.assertRaisesRegex(BrickManagerError, "Invalid brick type"):
            get_brick_summaries(self.state_paths, ["TTS"])

    def test_completes_cached_types_without_duplicates(self) -> None:
        self.assertEqual(
            complete_cli(self.state_paths, ["brick", "list", ""]),
            ["--installed", "--type"],
        )
        self.assertEqual(
            complete_cli(self.state_paths, ["brick", "list", "--type="]),
            ["--type=stt", "--type=tts", "--type=vad"],
        )
        self.assertEqual(
            complete_cli(self.state_paths, ["brick", "list", "--type=tts,"]),
            ["--type=tts,stt", "--type=tts,vad"],
        )

    @unittest.skipUnless(shutil.which("zsh"), "Zsh is unavailable")
    def test_zsh_completion_does_not_append_a_space(self) -> None:
        self.assertEqual(self._zsh_completion_suffix("model\t"), "\n")
        self.assertEqual(
            self._zsh_completion_suffix("example-model\tExample model"), "\n"
        )

    def _zsh_completion_suffix(self, completion_output: str) -> str:
        bin_directory = Path(self.temporary_directory.name) / "bin"
        bin_directory.mkdir(exist_ok=True)
        executable = bin_directory / "compute-flow"
        executable.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$COMPLETION_OUTPUT\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

        environment = os.environ.copy()
        environment["COMPLETION_OUTPUT"] = completion_output
        environment["PATH"] = f"{bin_directory}{os.pathsep}{environment['PATH']}"
        completion_script = (
            REPOSITORY_ROOT / "scripts" / "completion" / "brick_cli_completion.sh"
        )
        environment["COMPLETION_SCRIPT"] = str(completion_script)
        result = subprocess.run(
            [
                "zsh",
                "-f",
                "-c",
                """
compdef() { :; }
source "$COMPLETION_SCRIPT"
compadd() {
    local suffix=default
    while (( $# > 0 )); do
        if [[ "$1" == "-S" ]]; then
            suffix="$2"
            break
        fi
        shift
    done
    print -r -- "$suffix"
}
words=(compute-flow m)
CURRENT=2
_compute_flow_manager_complete_zsh
""",
            ],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        return result.stdout

    @staticmethod
    def _entry(description: str, *, installed: bool = False) -> dict[str, object]:
        return {
            "info_json": {"description": description, "models": {}},
            "metadata": {
                "installed": installed,
                "installed_source_path": "/tmp/brick" if installed else None,
                "installed_models": {},
            },
        }


if __name__ == "__main__":
    unittest.main()
