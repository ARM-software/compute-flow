#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Integration tests for registry source commands."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import load_cache, write_cache  # noqa: E402
from brick_manager.cli import main  # noqa: E402
from brick_manager.completion import complete_cli  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from brick_manager.registry_commands import (  # noqa: E402
    add_registry_source,
    list_registry_sources,
    remove_registry_source,
)
from brick_manager.update import update_cache  # noqa: E402


class RegistryCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.state_paths = StatePaths(self.root / "state")
        self.official_repository = self._repository("official")
        (self.official_repository / "registry.txt").write_text("", encoding="utf-8")
        self._git(self.official_repository, "add", ".")
        self._git(self.official_repository, "commit", "--quiet", "-m", "Empty registry")
        self._git(self.official_repository, "tag", "0.0.0")
        for target in (
            "brick_manager.registry.OFFICIAL_REGISTRY_URL",
            "brick_manager.registry_commands.OFFICIAL_REGISTRY_URL",
        ):
            patcher = mock.patch(target, self.official_repository.as_uri())
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_add_registers_without_indexing_git_brick(self) -> None:
        repository = self._repository("first")
        self._release(repository, "1.0.0")

        result = add_registry_source(self.state_paths, repository.as_uri())

        self.assertEqual(result.status, "added")
        self.assertEqual(
            self.state_paths.registry.read_text(encoding="utf-8"),
            f"{repository.as_uri()}\n",
        )
        self.assertFalse(self.state_paths.cache.exists())

        update_cache(self.state_paths)

        cache = load_cache(self.state_paths.cache)
        self.assertIn("1.0.0", cache["bricks"]["tts:example"])
        self.assertFalse(
            cache["bricks"]["tts:example"]["1.0.0"]["metadata"]["installed"]
        )

        again = add_registry_source(self.state_paths, repository.as_uri())
        self.assertEqual(again.status, "already_registered")
        self.assertEqual(
            self.state_paths.registry.read_text(encoding="utf-8"),
            f"{repository.as_uri()}\n",
        )

    def test_add_does_not_contact_source(self) -> None:
        missing_url = (self.root / "missing.git").as_uri()

        result = add_registry_source(self.state_paths, missing_url)

        self.assertEqual(result.status, "added")
        self.assertFalse(self.state_paths.cache.exists())
        with self.assertRaises(BrickManagerError):
            update_cache(self.state_paths)

    def test_new_version_of_existing_variant_is_allowed(self) -> None:
        first = self._repository("first")
        second = self._repository("second")
        self._release(first, "1.0.0")
        self._release(second, "2.0.0")

        add_registry_source(self.state_paths, first.as_uri())
        add_registry_source(self.state_paths, second.as_uri())

        update_cache(self.state_paths)

        cache = load_cache(self.state_paths.cache)
        self.assertEqual(set(cache["bricks"]["tts:example"]), {"1.0.0", "2.0.0"})

    def test_duplicate_identity_is_reported_by_update(self) -> None:
        first = self._repository("first")
        second = self._repository("second")
        self._release(first, "1.0.0")
        self._release(second, "1.0.0")
        add_registry_source(self.state_paths, first.as_uri())
        update_cache(self.state_paths)
        add_registry_source(self.state_paths, second.as_uri())

        with self.assertRaisesRegex(BrickManagerError, "Duplicate brick identity"):
            update_cache(self.state_paths)

        self.assertEqual(
            self.state_paths.registry.read_text(encoding="utf-8"),
            f"{first.as_uri()}\n{second.as_uri()}\n",
        )
        cache = load_cache(self.state_paths.cache)
        self.assertEqual(set(cache["bricks"]["tts:example"]), {"1.0.0"})

    def test_cli_registry_commands_and_completion(self) -> None:
        repository = self._repository("cli")
        self._release(repository, "1.0.0")
        self.assertIn("registry", complete_cli(self.state_paths, ["r"]))
        self.assertEqual(
            complete_cli(self.state_paths, ["registry", ""]),
            ["add", "list", "remove"],
        )
        self.assertEqual(complete_cli(self.state_paths, ["brick", "a"]), [])
        self.assertEqual(
            complete_cli(self.state_paths, ["registry", "add", "--r"]),
            ["--registry"],
        )
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                main(["brick", "add", repository.as_uri()])
        self.assertEqual(error.exception.code, 2)

        with mock.patch(
            "brick_manager.cli.resolve_state_paths", return_value=self.state_paths
        ):
            self.assertEqual(main(["registry", "add", repository.as_uri()]), 0)
            self.assertEqual(main(["registry", "list"]), 0)
            self.assertEqual(main(["registry", "remove", repository.as_uri()]), 0)

        self.assertTrue(self.state_paths.registry.is_file())
        self.assertEqual(self.state_paths.registry.read_text(encoding="utf-8"), "")
        self.assertFalse(self.state_paths.cache.exists())

    def test_custom_registry_commands_use_only_selected_file(self) -> None:
        repository = self._repository("custom")
        self._release(repository, "1.0.0")
        custom_registry = self.root / "custom-registry.txt"

        with mock.patch(
            "brick_manager.cli.resolve_state_paths", return_value=self.state_paths
        ):
            self.assertEqual(
                main(
                    [
                        "registry",
                        "add",
                        repository.as_uri(),
                        "--registry",
                        str(custom_registry),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(["registry", "list", "--registry", str(custom_registry)]), 0
            )

        self.assertFalse(self.state_paths.registry.exists())
        self.assertEqual(
            [
                (entry.origin, entry.source)
                for entry in list_registry_sources(
                    self.state_paths, str(custom_registry)
                )
            ],
            [("Selected", repository.as_uri())],
        )
        self.assertFalse(self.state_paths.cache.exists())

        update_cache(self.state_paths, str(custom_registry))

        self.assertIn("tts:example", load_cache(self.state_paths.cache)["bricks"])

        with mock.patch(
            "brick_manager.cli.resolve_state_paths", return_value=self.state_paths
        ):
            self.assertEqual(
                main(
                    [
                        "registry",
                        "remove",
                        repository.as_uri(),
                        "--registry",
                        str(custom_registry),
                    ]
                ),
                0,
            )

        self.assertEqual(custom_registry.read_text(encoding="utf-8"), "")
        self.assertIn("tts:example", load_cache(self.state_paths.cache)["bricks"])

        update_cache(self.state_paths, str(custom_registry))

        self.assertEqual(load_cache(self.state_paths.cache)["bricks"], {})

    def test_list_shows_official_and_user_sources(self) -> None:
        official_source = self.official_repository.as_uri()
        user_source = "file:///user/brick.git"
        self.state_paths.registry.parent.mkdir(parents=True)
        self.state_paths.registry.write_text(f"{user_source}\n", encoding="utf-8")

        sources = list_registry_sources(self.state_paths)

        self.assertEqual(
            [(entry.origin, entry.source) for entry in sources],
            [("Official", official_source), ("User", user_source)],
        )
        custom_registry = self.root / "custom-registry.txt"
        custom_registry.write_text("file:///custom/brick.git\n", encoding="utf-8")
        self.assertEqual(
            [
                (entry.origin, entry.source)
                for entry in list_registry_sources(
                    self.state_paths, str(custom_registry)
                )
            ],
            [("Selected", "file:///custom/brick.git")],
        )
        with self.assertRaisesRegex(BrickManagerError, "official registry"):
            remove_registry_source(self.state_paths, official_source)

    def test_official_registry_needs_no_user_registry_file(self) -> None:
        self._release(self.official_repository, "1.0.0")
        source = self.official_repository.as_uri()

        self.assertEqual(
            [
                (entry.origin, entry.source)
                for entry in list_registry_sources(self.state_paths)
            ],
            [("Official", source)],
        )
        self.assertEqual(
            add_registry_source(self.state_paths, source).status, "already_registered"
        )
        with self.assertRaisesRegex(BrickManagerError, "official registry"):
            remove_registry_source(self.state_paths, source)

        result = update_cache(self.state_paths)

        self.assertEqual(result.brick_versions, 1)
        self.assertIsNone(result.registry_path)
        self.assertIsNone(result.user_registry_path)
        self.assertFalse(self.state_paths.registry.exists())
        self.assertEqual(
            set(load_cache(self.state_paths.cache)["bricks"]), {"tts:example"}
        )

    def test_remove_preserves_installed_brick(self) -> None:
        repository = self._repository("installed")
        self._release(repository, "1.0.0")
        add_registry_source(self.state_paths, repository.as_uri())
        update_cache(self.state_paths)
        cache = load_cache(self.state_paths.cache)
        entry = cache["bricks"]["tts:example"]["1.0.0"]
        entry["metadata"]["installed"] = True
        entry["metadata"]["installed_source_path"] = str(repository / "tts/example")
        entry["metadata"]["installed_info_json"] = entry["info_json"]
        write_cache(self.state_paths.cache, cache)

        remove_registry_source(self.state_paths, repository.as_uri())

        self.assertEqual(load_cache(self.state_paths.cache), cache)

        update_cache(self.state_paths)

        retained = load_cache(self.state_paths.cache)["bricks"]["tts:example"]["1.0.0"]
        self.assertTrue(retained["metadata"]["installed"])

    def test_custom_update_does_not_read_user_registry(self) -> None:
        repository = self._repository("first")
        self._release(repository, "1.0.0")
        add_registry_source(self.state_paths, repository.as_uri())

        custom_registry = self.root / "empty-registry.txt"
        custom_registry.write_text("", encoding="utf-8")
        self._release(self.official_repository, "2.0.0")

        result = update_cache(self.state_paths, str(custom_registry))

        self.assertEqual(result.brick_versions, 0)

    def test_update_reads_all_default_and_user_registries(self) -> None:
        official = self.official_repository
        additional = self._repository("additional")
        self._release(official, "1.0.0")
        self._release(additional, "2.0.0")
        add_registry_source(self.state_paths, additional.as_uri())

        result = update_cache(self.state_paths)

        self.assertEqual(result.brick_versions, 2)
        versions = load_cache(self.state_paths.cache)["bricks"]["tts:example"]
        self.assertEqual(set(versions), {"1.0.0", "2.0.0"})

    def _repository(self, name: str) -> Path:
        repository = self.root / name
        repository.mkdir()
        self._git(repository, "init", "--quiet")
        self._git(repository, "config", "user.name", "Brick Test")
        self._git(repository, "config", "user.email", "brick-test@example.com")
        return repository

    def _release(self, repository: Path, version: str) -> None:
        info_json = {
            "name": "Example",
            "type": "tts",
            "api": "tts.v1",
            "description": "Example brick",
            "author": "Brick Test",
            "version": version,
            "models": {},
        }
        source = repository / "tts/example"
        source.mkdir(parents=True, exist_ok=True)
        (repository / "registry.txt").write_text("tts/example\n", encoding="utf-8")
        (source / "info.json").write_text(json.dumps(info_json), encoding="utf-8")
        (source / "Example.cpp").write_text("", encoding="utf-8")
        (source / "Example.h").write_text(
            "namespace compute_flow::bricks::tts {\n"
            "class Example : public BrickBaseV1 {\n"
            '  std::string_view api() const noexcept override { return "tts.v1"; }\n'
            "};\n}\n",
            encoding="utf-8",
        )
        self._git(repository, "add", ".")
        self._git(repository, "commit", "--quiet", "-m", f"Release {version}")
        self._git(repository, "tag", version)

    def _git(self, repository: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
