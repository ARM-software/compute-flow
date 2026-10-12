#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for model artifact source fallback behavior."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import load_cache, write_cache  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.identity import BrickSelector  # noqa: E402
from brick_manager.install import install_brick  # noqa: E402
from brick_manager.model_artifacts import install_artifact, remove_artifact  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from brick_manager.uninstall import uninstall_brick  # noqa: E402


class ModelArtifactsTest(unittest.TestCase):
    def test_remove_artifact_unlinks_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target"
            target.mkdir()
            (target / "model.bin").write_bytes(b"model")
            link = root / "artifact"
            link.symlink_to(target, target_is_directory=True)

            remove_artifact(link)

            self.assertFalse(link.exists())
            self.assertTrue((target / "model.bin").exists())

    def test_install_falls_back_to_next_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source"
            source_path.mkdir()
            payload = b"model contents"
            (source_path / "model.bin").write_bytes(payload)
            manifest = f"{hashlib.sha256(payload).hexdigest()}  model.bin\n"
            (source_path / "SHA256SUMS").write_text(manifest, encoding="utf-8")

            destination = install_artifact(
                StatePaths(root / "state"),
                BrickSelector("tts", "example", "1.0.0"),
                "example-model",
                sources=[
                    {"kind": "local", "url": str(root / "missing")},
                    {"kind": "local", "url": str(source_path)},
                ],
                expected_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
            )

            self.assertEqual((destination / "model.bin").read_bytes(), payload)


class BrickModelCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.state_paths = StatePaths(self.root / "state")
        self.selector = BrickSelector("tts", "example", "1.0.0")
        self.install_path = self.state_paths.brick_source(self.selector)

    def test_force_install_keeps_only_compatible_declared_models(self) -> None:
        stale_package, stale_sha256 = self._write_model_package("stale", b"stale")
        compatible_package, compatible_sha256 = self._write_model_package(
            "compatible", b"compatible"
        )
        undeclared_package, undeclared_sha256 = self._write_model_package(
            "undeclared", b"undeclared"
        )
        stale_path = self._copy_model_package(stale_package)
        compatible_path = self._copy_model_package(
            compatible_package,
            "compatible-model",
        )
        undeclared_path = self._copy_model_package(
            undeclared_package,
            "undeclared-model",
        )

        source_info = self._info_json(self.root / "replacement", "0" * 64)
        source_info["models"]["compatible-model"] = {
            "description": "Compatible model.",
            "sources": [{"kind": "local", "url": str(compatible_package)}],
            "sha256": compatible_sha256,
        }
        source_path = self.root / "brick-source"
        source_path.mkdir()
        (source_path / "Example.cpp").write_text("", encoding="utf-8")
        (source_path / "Example.h").write_text("", encoding="utf-8")
        self._write_info_json(source_path, source_info)
        self._write_installed_brick(source_info)
        self._write_cache(
            source_info,
            installed_models={
                "example-model": {
                    "path": str(stale_path),
                    "ref": "local",
                    "sha256": stale_sha256,
                },
                "compatible-model": {
                    "path": str(compatible_path),
                    "ref": "local",
                    "sha256": compatible_sha256,
                },
                "undeclared-model": {
                    "path": str(undeclared_path),
                    "ref": "local",
                    "sha256": undeclared_sha256,
                },
            },
        )

        result = install_brick(
            self.state_paths,
            self.selector.identity,
            force=True,
            confirm=lambda _path: True,
        )

        self.assertEqual(result.incompatible_models, ["example-model"])
        self.assertEqual(result.removed_models, ["undeclared-model"])
        self.assertFalse(stale_path.exists())
        self.assertFalse(undeclared_path.exists())
        self.assertTrue(compatible_path.exists())
        metadata = load_cache(self.state_paths.cache)["bricks"][self.selector.key][
            "1.0.0"
        ]["metadata"]
        self.assertEqual(
            sorted(metadata["installed_models"]),
            ["compatible-model"],
        )

    def test_uninstall_removes_brick_and_all_related_models(self) -> None:
        package_path, package_sha256 = self._write_model_package("model", b"model")
        model_path = self._copy_model_package(package_path)
        untracked_model_path = self._copy_model_package(package_path, "untracked-model")
        info_json = self._info_json(package_path, package_sha256)
        self._write_installed_brick(info_json)
        self._write_cache(
            info_json,
            installed_models={
                "example-model": {
                    "path": str(model_path),
                    "ref": "local",
                    "sha256": package_sha256,
                }
            },
        )

        result = uninstall_brick(
            self.state_paths,
            self.selector.key,
            confirm=lambda _path: True,
        )

        self.assertEqual(
            result.removed_models,
            ["example-model", "untracked-model"],
        )
        self.assertFalse(self.install_path.exists())
        self.assertFalse(model_path.parent.exists())
        self.assertFalse(untracked_model_path.exists())
        metadata = load_cache(self.state_paths.cache)["bricks"][self.selector.key][
            "1.0.0"
        ]["metadata"]
        self.assertFalse(metadata["installed"])
        self.assertEqual(metadata["installed_models"], {})

    def test_cancelled_uninstall_preserves_brick_and_models(self) -> None:
        package_path, package_sha256 = self._write_model_package("model", b"model")
        model_path = self._copy_model_package(package_path)
        info_json = self._info_json(package_path, package_sha256)
        self._write_installed_brick(info_json)
        self._write_cache(
            info_json,
            installed_models={
                "example-model": {
                    "path": str(model_path),
                    "ref": "local",
                    "sha256": package_sha256,
                }
            },
        )

        with self.assertRaisesRegex(BrickManagerError, "cancelled"):
            uninstall_brick(
                self.state_paths,
                self.selector.key,
                confirm=lambda _path: False,
            )

        self.assertTrue(self.install_path.exists())
        self.assertTrue(model_path.exists())

    def _write_model_package(
        self,
        name: str,
        payload: bytes,
    ) -> tuple[Path, str]:
        package_path = self.root / name
        package_path.mkdir()
        (package_path / "model.bin").write_bytes(payload)
        manifest = f"{hashlib.sha256(payload).hexdigest()}  model.bin\n"
        (package_path / "SHA256SUMS").write_text(manifest, encoding="utf-8")
        return package_path, hashlib.sha256(manifest.encode()).hexdigest()

    def _copy_model_package(
        self,
        source_path: Path,
        model_name: str = "example-model",
    ) -> Path:
        destination = (
            self.state_paths.models
            / self.selector.brick_type
            / self.selector.name
            / model_name
        )
        destination.mkdir(parents=True)
        for source_file in source_path.iterdir():
            (destination / source_file.name).write_bytes(source_file.read_bytes())
        return destination

    def _write_installed_brick(self, info_json: dict[str, object]) -> None:
        self.install_path.mkdir(parents=True)
        self._write_info_json(self.install_path, info_json)

    def _write_cache(
        self,
        info_json: dict[str, object],
        *,
        installed_models: dict[str, dict[str, str]] | None = None,
    ) -> None:
        entry = {
            "info_json": info_json,
            "metadata": {
                "source": str(self.root / "brick-source"),
                "source_type": "local_directory",
                "source_path": str(self.root / "brick-source"),
                "manager_name": self.selector.name,
                "installed": True,
                "installed_source_path": str(self.install_path),
                "installed_models": installed_models or {},
            },
        }
        write_cache(
            self.state_paths.cache,
            {"bricks": {self.selector.key: {"1.0.0": entry}}},
        )

    @staticmethod
    def _write_info_json(path: Path, info_json: dict[str, object]) -> None:
        (path / "info.json").write_text(json.dumps(info_json), encoding="utf-8")

    @staticmethod
    def _info_json(model_source: Path, sha256: str) -> dict[str, object]:
        return {
            "name": "Example",
            "type": "tts",
            "description": "Example brick.",
            "author": "Example author",
            "version": "1.0.0",
            "models": {
                "example-model": {
                    "description": "Example model.",
                    "sources": [{"kind": "local", "url": str(model_source)}],
                    "sha256": sha256,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
