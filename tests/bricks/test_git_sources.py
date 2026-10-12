#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Integration tests for Git repositories in the brick registry."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import load_cache  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.install import install_brick  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from brick_manager.registry import is_git_source  # noqa: E402
from brick_manager.update import update_cache  # noqa: E402


class GitSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.repository = self.root / "brick-repository"
        self.repository.mkdir()
        self.state_paths = StatePaths(self.root / "state")
        self.registry = self.root / "registry.txt"
        self.url = self.repository.as_uri()
        self.registry.write_text(f"{self.url}\n", encoding="utf-8")
        self._git("init", "--quiet")
        self._git("config", "user.name", "Brick Test")
        self._git("config", "user.email", "brick-test@example.com")

    def test_indexes_and_installs_tagged_versions(self) -> None:
        first_commit = self._commit_version("1.0.0", "first")
        self._git("tag", "1.0.0")
        second_commit = self._commit_version("2.0.0", "second")
        self._git("tag", "-a", "2.0.0", "-m", "Second release")
        self._git("branch", "2.0.0", "1.0.0")

        result = update_cache(self.state_paths, str(self.registry))

        self.assertEqual(result.brick_versions, 2)
        cache = load_cache(self.state_paths.cache)
        versions = cache["bricks"]["tts:example"]
        self.assertEqual(set(versions), {"1.0.0", "2.0.0"})
        self.assertEqual(versions["1.0.0"]["metadata"]["source_commit"], first_commit)
        self.assertEqual(versions["2.0.0"]["metadata"]["source_commit"], second_commit)
        self.assertEqual(versions["2.0.0"]["metadata"]["source_ref"], "2.0.0")

        installed = install_brick(self.state_paths, "tts:example@1.0.0")
        self.assertEqual((installed.install_path / "Example.cpp").read_text(), "first")
        self.assertFalse((installed.install_path / ".git").exists())

        replacement = install_brick(self.state_paths, "tts:example@2.0.0")
        self.assertEqual(
            (replacement.install_path / "Example.cpp").read_text(), "second"
        )
        self.assertFalse(installed.install_path.exists())

    def test_changed_tag_fails_before_replacing_installed_brick(self) -> None:
        self._commit_version("1.0.0", "first")
        self._git("tag", "1.0.0")
        self._commit_version("2.0.0", "second")
        self._git("tag", "2.0.0")
        update_cache(self.state_paths, str(self.registry))
        installed = install_brick(self.state_paths, "tts:example@1.0.0")

        self._commit_version("2.0.0", "changed")
        self._git("tag", "--force", "2.0.0")

        with self.assertRaisesRegex(BrickManagerError, "changed after it was cached"):
            install_brick(self.state_paths, "tts:example@2.0.0")

        self.assertEqual((installed.install_path / "Example.cpp").read_text(), "first")

    def test_update_accepts_git_and_relative_local_sources_together(self) -> None:
        self._commit_version("1.0.0", "remote")
        self._git("tag", "1.0.0")
        local_source = self.root / "local-brick"
        local_source.mkdir()
        local_info = json.loads((self.repository / "info.json").read_text())
        local_info["name"] = "LocalExample"
        (local_source / "info.json").write_text(
            json.dumps(local_info), encoding="utf-8"
        )
        self.registry.write_text(f"{self.url}\nlocal-brick\n", encoding="utf-8")

        result = update_cache(self.state_paths, str(self.registry))

        self.assertEqual(result.brick_versions, 2)
        cache = load_cache(self.state_paths.cache)
        self.assertEqual(
            cache["bricks"]["tts:local-example"]["1.0.0"]["metadata"]["source_path"],
            str(local_source.resolve()),
        )

    def test_untagged_repository_is_rejected(self) -> None:
        self._commit_version("1.0.0", "first")

        with self.assertRaisesRegex(BrickManagerError, "has no tags"):
            update_cache(self.state_paths, str(self.registry))

        self.assertFalse(self.state_paths.cache.exists())

    def test_indexes_tag_not_reachable_from_a_branch(self) -> None:
        self._commit_version("1.0.0", "first")
        self._git("tag", "1.0.0")
        branch = self._git("branch", "--show-current").strip()
        self._git("checkout", "--quiet", "--detach")
        second_commit = self._commit_version("2.0.0", "archived")
        self._git("tag", "2.0.0")
        self._git("checkout", "--quiet", branch)

        update_cache(self.state_paths, str(self.registry))

        cache = load_cache(self.state_paths.cache)
        self.assertEqual(
            cache["bricks"]["tts:example"]["2.0.0"]["metadata"]["source_commit"],
            second_commit,
        )

    def test_install_rejects_link_outside_git_source(self) -> None:
        self._commit_version("1.0.0", "first")
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.repository / "outside.txt"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        self._git("add", "outside.txt")
        self._git("commit", "--quiet", "-m", "Add external link")
        self._git("tag", "1.0.0")
        update_cache(self.state_paths, str(self.registry))

        with self.assertRaisesRegex(BrickManagerError, "link outside its root"):
            install_brick(self.state_paths, "tts:example@1.0.0")

        self.assertFalse(
            (self.state_paths.bricks / "tts" / "example" / "1.0.0").exists()
        )

    def test_recognizes_git_urls_and_local_paths(self) -> None:
        for source in (
            "https://example.com/team/brick.git",
            "ssh://git@example.com/team/brick.git",
            "git@example.com:team/brick.git",
            self.url,
        ):
            with self.subTest(source=source):
                self.assertTrue(is_git_source(source))
        self.assertFalse(is_git_source("../bricks/tts/example"))

    def _commit_version(self, version: str, implementation: str) -> str:
        info_json = {
            "name": "Example",
            "type": "tts",
            "api": "tts.v1",
            "description": "Example brick",
            "author": "Brick Test",
            "version": version,
            "models": {},
        }
        (self.repository / "info.json").write_text(
            json.dumps(info_json), encoding="utf-8"
        )
        (self.repository / "Example.cpp").write_text(implementation, encoding="utf-8")
        (self.repository / "Example.h").write_text("", encoding="utf-8")
        self._git("add", "info.json", "Example.cpp", "Example.h")
        self._git("commit", "--quiet", "-m", f"Release {version}")
        return self._git("rev-parse", "HEAD").strip()

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()
