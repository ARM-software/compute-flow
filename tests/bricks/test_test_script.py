#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for generating brick-local CLI test scripts during installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import write_cache  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.identity import BrickSelector  # noqa: E402
from brick_manager.install import install_brick  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402


class BrickTestScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.state_paths = StatePaths(self.root / "state")
        self.selector = BrickSelector("vad", "example", "1.0.0")
        self.source = self.root / "brick-source"
        self.source.mkdir()
        (self.source / "Example.cpp").write_text("", encoding="utf-8")
        (self.source / "Example.h").write_text("", encoding="utf-8")
        info_json = {
            "name": "Example",
            "type": "vad",
            "description": "Example brick.",
            "author": "Example author",
            "version": "1.0.0",
            "models": {
                "example-model": {
                    "description": "Example model.",
                    "sources": [
                        {"kind": "local", "url": str(self.root / "model-source")}
                    ],
                    "sha256": "0" * 64,
                },
                "other-model": {
                    "description": "Other model.",
                    "sources": [
                        {"kind": "local", "url": str(self.root / "other-source")}
                    ],
                    "sha256": "2" * 64,
                },
                "unused-model": {
                    "description": "Unused model.",
                    "sources": [
                        {"kind": "local", "url": str(self.root / "unused-source")}
                    ],
                    "sha256": "1" * 64,
                },
            },
        }
        (self.source / "info.json").write_text(json.dumps(info_json), encoding="utf-8")
        write_cache(
            self.state_paths.cache,
            {
                "bricks": {
                    self.selector.key: {
                        "1.0.0": {
                            "info_json": info_json,
                            "metadata": {
                                "source": str(self.source),
                                "source_type": "local_directory",
                                "source_path": str(self.source),
                                "manager_name": self.selector.name,
                                "installed": False,
                                "installed_source_path": None,
                                "installed_models": {},
                            },
                        }
                    }
                }
            },
        )

    def test_install_generates_script_for_selected_cases(self) -> None:
        self._write_cases(
            "models:\n"
            "  example-model:\n"
            "    example_manifest.json:\n"
            "      - args: ['-a', '@sample/beckett.wav']\n"
            "      - args: ['-a', '@sample/space name.wav', '-t', '2']\n"
            "  other-model:\n"
            "    other_manifest.json:\n"
            "      - args: ['-a', '@sample/other.wav']\n"
        )

        result = install_brick(self.state_paths, self.selector.identity)

        script = result.install_path / "test" / "run_tests.sh"
        self.assertEqual(result.test_script_path, script)
        self.assertTrue(script.is_file())
        self.assertFalse(script.stat().st_mode & 0o111)
        self.assertFalse((self.source / "test" / "run_tests.sh").exists())
        self.assertNotIn("unused-model", script.read_text(encoding="utf-8"))

        if shutil.which("bash") is None:
            self.skipTest("Bash is required to execute the generated script")

        model_dir = self.root / "installed-model"
        model_dir.mkdir()
        (model_dir / "example_manifest.json").write_text("{}", encoding="utf-8")
        (model_dir / "other_manifest.json").write_text("{}", encoding="utf-8")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        log_path = self.root / "commands.log"
        self._write_command_stub(bin_dir / "compute-flow", "compute-flow")
        self._write_command_stub(bin_dir / "python", "python")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                "TEST_LOG": str(log_path),
                "TEST_MODEL_DIR": str(model_dir),
            }
        )

        completed = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            log_path.read_text(encoding="utf-8").splitlines(),
            [
                "compute-flow <model> <install> <vad:example> <example-model>",
                "compute-flow <model> <path> <vad:example> <example-model>",
                "python <-m> <cli_examples.vad_cli> <-m> <@manifest/example/example-model/example_manifest.json> <-a> <@sample/beckett.wav>",
                "python <-m> <cli_examples.vad_cli> <-m> <@manifest/example/example-model/example_manifest.json> <-a> <@sample/space name.wav> <-t> <2>",
                "compute-flow <model> <uninstall> <vad:example> <example-model>",
                "compute-flow <model> <install> <vad:example> <other-model>",
                "compute-flow <model> <path> <vad:example> <other-model>",
                "python <-m> <cli_examples.vad_cli> <-m> <@manifest/example/other-model/other_manifest.json> <-a> <@sample/other.wav>",
                "compute-flow <model> <uninstall> <vad:example> <other-model>",
            ],
        )

        (model_dir / "example_manifest.json").unlink()
        log_path.unlink()
        missing_manifest = subprocess.run(
            ["bash", str(script)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(missing_manifest.returncode, 0)
        self.assertIn("Manifest is not installed", missing_manifest.stderr)
        self.assertNotIn("python", log_path.read_text(encoding="utf-8"))
        self.assertNotIn("other-model", log_path.read_text(encoding="utf-8"))
        self.assertEqual(
            log_path.read_text(encoding="utf-8").splitlines()[-1],
            "compute-flow <model> <uninstall> <vad:example> <example-model>",
        )

        (model_dir / "example_manifest.json").write_text("{}", encoding="utf-8")
        log_path.unlink()
        failed_cli = subprocess.run(
            ["bash", str(script)],
            env={**environment, "TEST_FAIL_PYTHON": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(failed_cli.returncode, 42)
        self.assertEqual(
            log_path.read_text(encoding="utf-8").splitlines()[-1],
            "compute-flow <model> <uninstall> <vad:example> <example-model>",
        )
        self.assertNotIn("other-model", log_path.read_text(encoding="utf-8"))

    def test_install_without_cases_does_not_generate_script(self) -> None:
        result = install_brick(self.state_paths, self.selector.identity)

        self.assertIsNone(result.test_script_path)
        self.assertFalse((result.install_path / "test" / "run_tests.sh").exists())

    def test_existing_install_regenerates_missing_script(self) -> None:
        self._write_cases(
            "models:\n"
            "  example-model:\n"
            "    example_manifest.json:\n"
            "      - args: ['-a', '@sample/beckett.wav']\n"
        )
        first = install_brick(self.state_paths, self.selector.identity)
        first.test_script_path.unlink()

        second = install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(second.status, "already_installed")
        self.assertTrue(second.test_script_path.is_file())

    def test_invalid_cases_fail_installation(self) -> None:
        self._write_cases(
            "models:\n"
            "  unknown-model:\n"
            "    example_manifest.json:\n"
            "      - args: ['-a', '@sample/beckett.wav']\n"
        )

        with self.assertRaisesRegex(BrickManagerError, "not declared in info.json"):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.state_paths.brick_source(self.selector).exists())

    def test_install_does_not_replace_handwritten_test_script(self) -> None:
        self._write_cases(
            "models:\n"
            "  example-model:\n"
            "    example_manifest.json:\n"
            "      - args: ['-a', '@sample/beckett.wav']\n"
        )
        handwritten = self.source / "test" / "run_tests.sh"
        handwritten.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

        with self.assertRaisesRegex(BrickManagerError, "already in use"):
            install_brick(self.state_paths, self.selector.identity)

        self.assertEqual(handwritten.read_text(encoding="utf-8"), "#!/bin/sh\nexit 0\n")
        self.assertFalse(self.state_paths.brick_source(self.selector).exists())

    def _write_cases(self, content: str) -> None:
        test_dir = self.source / "test"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test_cases.yaml").write_text(content, encoding="utf-8")

    @staticmethod
    def _write_command_stub(path: Path, name: str) -> None:
        path.write_text(
            "#!/bin/sh\n"
            f"printf '{name}' >> \"$TEST_LOG\"\n"
            'printf \' <%s>\' "$@" >> "$TEST_LOG"\n'
            "printf '\\n' >> \"$TEST_LOG\"\n"
            'if [ "$1" = model ] && [ "$2" = path ]; then\n'
            "  printf '%s\\n' \"$TEST_MODEL_DIR\"\n"
            "fi\n"
            f'if [ "{name}" = python ] && [ "${{TEST_FAIL_PYTHON:-0}}" = 1 ]; then exit 42; fi\n',
            encoding="utf-8",
        )
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
