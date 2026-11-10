#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Integration tests for repositories containing brick registries."""

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
from brick_manager.catalog import (  # noqa: E402
    get_brick_info,
    get_brick_model_list,
    get_brick_summaries,
)
from brick_manager.completion import complete_cli  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.install import install_all_bricks, install_brick  # noqa: E402
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

    def test_each_registry_tag_versions_the_bricks_it_contains(self) -> None:
        self._commit_version("1.0.0", "first", tag="1.0.0")
        commit = self._commit_version("2.0.0", "second", brick_type="stt", tag="3.0.0")

        result = update_cache(self.state_paths, str(self.registry))

        self.assertEqual(result.brick_versions, 3)
        cache = load_cache(self.state_paths.cache)
        self.assertEqual(set(cache["bricks"]), {"tts:example", "stt:example"})
        self.assertEqual(set(cache["bricks"]["tts:example"]), {"1.0.0", "3.0.0"})
        self.assertEqual(set(cache["bricks"]["stt:example"]), {"3.0.0"})
        for brick_type in ("tts", "stt"):
            with self.subTest(brick_type=brick_type):
                metadata = cache["bricks"][f"{brick_type}:example"]["3.0.0"]["metadata"]
                self.assertEqual(metadata["source_commit"], commit)
                self.assertEqual(metadata["source_ref"], "3.0.0")
                self.assertEqual(metadata["source_subdir"], f"{brick_type}/example")
                installed = install_brick(
                    self.state_paths, f"{brick_type}:example@3.0.0"
                )
                self.assertEqual(
                    json.loads((installed.install_path / "info.json").read_text())[
                        "version"
                    ],
                    "3.0.0",
                )
                self.assertEqual(
                    (installed.install_path / "Example.cpp").read_text(),
                    "first" if brick_type == "tts" else "second",
                )
                self.assertFalse((installed.install_path / ".git").exists())
                self.assertFalse((installed.install_path / "registry.txt").exists())
                self.assertFalse((installed.install_path / "stt").exists())
                self.assertFalse((installed.install_path / "tts").exists())

    def test_install_uses_cached_commit_after_default_branch_advances(self) -> None:
        commit = self._commit_version("1.0.0", "first", tag="1.0.0")
        update_cache(self.state_paths, str(self.registry))
        self._commit_version("1.0.0", "changed")

        installed = install_brick(self.state_paths, "tts:example")

        self.assertEqual((installed.install_path / "Example.cpp").read_text(), "first")
        metadata = load_cache(self.state_paths.cache)["bricks"]["tts:example"]["1.0.0"][
            "metadata"
        ]
        self.assertEqual(metadata["source_commit"], commit)

        update_cache(self.state_paths, str(self.registry))
        self.assertEqual(
            load_cache(self.state_paths.cache)["bricks"]["tts:example"]["1.0.0"][
                "metadata"
            ]["source_commit"],
            commit,
        )
        self._git("tag", "1.1.0")
        update_cache(self.state_paths, str(self.registry))
        refreshed = install_brick(self.state_paths, "tts:example@1.1.0")
        self.assertEqual(
            (refreshed.install_path / "Example.cpp").read_text(), "changed"
        )

    def test_update_preserves_installed_version(self) -> None:
        self._commit_version("1.0.0", "first", tag="1.0.0")
        update_cache(self.state_paths, str(self.registry))
        installed = install_brick(self.state_paths, "tts:example")
        self._commit_version("2.0.0", "second", tag="2.0.0")

        result = update_cache(self.state_paths, str(self.registry))

        versions = load_cache(self.state_paths.cache)["bricks"]["tts:example"]
        self.assertEqual(set(versions), {"1.0.0", "2.0.0"})
        self.assertTrue(versions["1.0.0"]["metadata"]["installed"])
        self.assertEqual((installed.install_path / "Example.cpp").read_text(), "first")
        self.assertEqual(
            result.available_updates[0].available_identity, "tts:example@2.0.0"
        )

    def test_update_accepts_git_and_relative_local_sources_together(self) -> None:
        self._commit_version("1.0.0", "remote", tag="1.0.0")
        local_source = self.root / "local-brick"
        local_source.mkdir()
        local_info = json.loads((self.repository / "tts/example/info.json").read_text())
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

    def test_repository_requires_registry_file(self) -> None:
        self._commit_version("1.0.0", "first")
        self._git("rm", "registry.txt")
        self._git("commit", "--quiet", "-m", "Remove registry")
        self._git("tag", "1.0.0")

        with self.assertRaisesRegex(BrickManagerError, "root registry.txt"):
            update_cache(self.state_paths, str(self.registry))

        self.assertFalse(self.state_paths.cache.exists())

    def test_invalid_registry_leaves_cache_unchanged(self) -> None:
        self._commit_version("1.0.0", "first", tag="1.0.0")
        update_cache(self.state_paths, str(self.registry))
        original_cache = self.state_paths.cache.read_bytes()
        for contents in (
            "../outside\n",
            "/tts/example\n",
            "bricks/tts/example\n",
            "tts/../example\n",
            "https://example.com/brick.git\n",
            "tts/example\ntts/example\n",
            "tts/missing\n",
            "tts/example\n\n",
        ):
            with self.subTest(contents=contents):
                (self.repository / "registry.txt").write_text(
                    contents, encoding="utf-8"
                )
                self._git("add", "registry.txt")
                self._git("commit", "--quiet", "-m", "Invalid registry")
                self._git("tag", "--force", "1.0.0")

                with self.assertRaises(BrickManagerError):
                    update_cache(self.state_paths, str(self.registry))

                self.assertEqual(self.state_paths.cache.read_bytes(), original_cache)

    def test_type_must_match_registry_directory(self) -> None:
        self._commit_version("1.0.0", "first")
        info_path = self.repository / "tts/example/info.json"
        info = json.loads(info_path.read_text())
        info["type"] = "stt"
        info_path.write_text(json.dumps(info), encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", "Mismatched type")
        self._git("tag", "1.0.0")

        with self.assertRaisesRegex(BrickManagerError, "type does not match"):
            update_cache(self.state_paths, str(self.registry))

    def test_registry_directory_cannot_escape_checkout(self) -> None:
        self._commit_version("1.0.0", "first")
        link = self.repository / "tts/outside"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        (self.repository / "registry.txt").write_text("tts/outside\n", encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", "External brick directory")
        self._git("tag", "1.0.0")

        with self.assertRaisesRegex(BrickManagerError, "escapes its repository"):
            update_cache(self.state_paths, str(self.registry))

    def test_install_rejects_link_outside_brick(self) -> None:
        self._commit_version("1.0.0", "first")
        outside = self.repository / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.repository / "tts/example/outside.txt"
        try:
            link.symlink_to("../../outside.txt")
        except OSError as error:
            self.skipTest(f"Symlinks are unavailable: {error}")
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", "Add external link")
        self._git("tag", "1.0.0")
        update_cache(self.state_paths, str(self.registry))

        with self.assertRaisesRegex(BrickManagerError, "link outside its root"):
            install_brick(self.state_paths, "tts:example@1.0.0")

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

    def test_latest_tag_uses_version_order_and_supports_annotated_tags(self) -> None:
        commit = self._commit_version("99.0.0", "latest release")
        self._git("tag", "-a", "1.10.0", "-m", "Latest release")
        self._commit_version("100.0.0", "older release", tag="1.9.0")
        self._commit_version("101.0.0", "unreleased")

        update_cache(self.state_paths, str(self.registry))

        versions = load_cache(self.state_paths.cache)["bricks"]["tts:example"]
        self.assertEqual(set(versions), {"1.9.0", "1.10.0"})
        self.assertEqual(versions["1.10.0"]["metadata"]["source_commit"], commit)
        self.assertEqual(
            [summary.version for summary in get_brick_summaries(self.state_paths)],
            ["1.9.0", "1.10.0"],
        )
        self.assertEqual(
            set(complete_cli(self.state_paths, ["brick", "install", "tts:example@"])),
            {"tts:example@1.9.0", "tts:example@1.10.0"},
        )
        older = install_brick(self.state_paths, "tts:example@1.9.0")
        self.assertEqual(
            (older.install_path / "Example.cpp").read_text(), "older release"
        )
        self.assertEqual(
            get_brick_model_list(self.state_paths, "tts:example").version, "1.9.0"
        )
        self.assertEqual(
            get_brick_info(self.state_paths, "tts:example").version, "1.10.0"
        )
        installed = install_brick(self.state_paths, "tts:example")
        self.assertEqual(installed.identity, "tts:example@1.10.0")
        self.assertEqual(
            (installed.install_path / "Example.cpp").read_text(), "latest release"
        )

    def test_install_all_selects_only_latest_version_of_each_brick(self) -> None:
        self._commit_version("99.0.0", "older release", tag="1.9.0")
        self._commit_version("99.0.0", "latest release")
        self._commit_version(
            "99.0.0", "speech recognition", brick_type="stt", tag="1.10.0"
        )
        update_cache(self.state_paths, str(self.registry))

        result = install_all_bricks(self.state_paths)

        self.assertEqual(result.total, 2)
        self.assertEqual(result.statuses["installed"], 2)
        self.assertEqual(
            [
                (summary.brick, summary.version)
                for summary in get_brick_summaries(
                    self.state_paths, installed_only=True
                )
            ],
            [("stt:example", "1.10.0"), ("tts:example", "1.10.0")],
        )

    def test_source_metadata_may_omit_version(self) -> None:
        self._commit_version("1.0.0", "first")
        info_path = self.repository / "tts/example/info.json"
        info = json.loads(info_path.read_text())
        del info["version"]
        info_path.write_text(json.dumps(info), encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", "Use registry version")
        self._git("tag", "0.0.0")

        update_cache(self.state_paths, str(self.registry))
        installed = install_brick(self.state_paths, "tts:example@0.0.0")

        self.assertEqual(
            json.loads((installed.install_path / "info.json").read_text())["version"],
            "0.0.0",
        )
        self.assertNotIn("version", json.loads(info_path.read_text()))
        self.assertEqual(
            install_brick(self.state_paths, "tts:example").status, "already_installed"
        )

    def test_registry_requires_a_tag(self) -> None:
        self._commit_version("1.0.0", "unreleased")

        with self.assertRaisesRegex(BrickManagerError, "Git registry has no tags"):
            update_cache(self.state_paths, str(self.registry))

        self.assertFalse(self.state_paths.cache.exists())

    def test_registry_tags_require_major_minor_patch(self) -> None:
        self._commit_version("1.0.0", "released", tag="1.0.0")
        update_cache(self.state_paths, str(self.registry))
        original_cache = self.state_paths.cache.read_bytes()

        for tag in (
            "v1.2.3",
            "1.2",
            "1.2.3.4",
            "1.2.3-rc.1",
            "1.2.3+build.1",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "latest",
        ):
            with self.subTest(tag=tag):
                self._git("tag", tag)

                with self.assertRaisesRegex(
                    BrickManagerError, "Expected MAJOR.MINOR.PATCH"
                ):
                    update_cache(self.state_paths, str(self.registry))

                self.assertEqual(self.state_paths.cache.read_bytes(), original_cache)
                self._git("tag", "--delete", tag)

    def _commit_version(
        self,
        version: str,
        implementation: str,
        *,
        brick_type: str = "tts",
        tag: str | None = None,
    ) -> str:
        info_json = {
            "name": "Example",
            "type": brick_type,
            "api": f"{brick_type}.v1",
            "description": "Example brick",
            "author": "Brick Test",
            "version": version,
            "models": {},
        }
        subdir = f"{brick_type}/example"
        source = self.repository / subdir
        source.mkdir(parents=True, exist_ok=True)
        (source / "info.json").write_text(json.dumps(info_json), encoding="utf-8")
        (source / "Example.cpp").write_text(implementation, encoding="utf-8")
        (source / "Example.h").write_text("", encoding="utf-8")
        registry = self.repository / "registry.txt"
        entries = registry.read_text().splitlines() if registry.exists() else []
        if subdir not in entries:
            entries.append(subdir)
        registry.write_text(
            "".join(f"{entry}\n" for entry in entries), encoding="utf-8"
        )
        self._git("add", ".")
        self._git("commit", "--quiet", "-m", f"Release {version}")
        if tag is not None:
            self._git("tag", tag)
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
