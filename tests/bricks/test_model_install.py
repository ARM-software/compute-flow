#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for Brick Manager installed model metadata and replacement behavior."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cache import load_cache, write_cache  # noqa: E402
from brick_manager.catalog import (  # noqa: E402
    get_brick_model_info,
    get_brick_model_list,
    get_brick_model_path,
)
from brick_manager.completion import complete_cli  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.identity import BrickSelector  # noqa: E402
from brick_manager.install import install_brick  # noqa: E402
from brick_manager.model_install import (  # noqa: E402
    install_models,
    reconcile_installed_models,
    uninstall_models,
)
from brick_manager.model_import import (  # noqa: E402
    ModelImportChoices,
    ModelImportConfig,
    import_model,
    parse_model_config_value,
    parse_model_import_source,
    prepare_model_import,
)
from brick_manager.model_import_interactive import prompt_model_import  # noqa: E402
from brick_manager.paths import StatePaths  # noqa: E402
from brick_manager.render import THEME, render_update_result  # noqa: E402
from brick_manager.update import update_cache  # noqa: E402


class _InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class ModelInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.state_paths = StatePaths(self.root / "state")
        self.selector = BrickSelector("tts", "example", "1.0.0")
        self.install_path = self.state_paths.brick_source(self.selector)

    def test_install_uses_installed_brick_metadata(self) -> None:
        package_path, package_sha256 = self._write_model_package(
            "source", b"installed declaration"
        )
        installed_info = self._info_json(package_path, package_sha256)
        cached_info = copy.deepcopy(installed_info)
        cached_model = cached_info["models"]["example-model"]
        cached_model["sources"][0]["url"] = str(self.root / "missing")
        cached_model["sha256"] = "0" * 64
        self._write_installed_brick(installed_info)
        self._write_cache(cached_info)

        result = install_models(
            self.state_paths,
            self.selector.key,
            ["example-model"],
        )

        self.assertEqual(result.items[0].status, "installed")
        self.assertEqual(
            (result.items[0].path / "model.bin").read_bytes(),
            b"installed declaration",
        )
        cache = load_cache(self.state_paths.cache)
        state = cache["bricks"][self.selector.key]["1.0.0"]["metadata"]
        self.assertEqual(
            state["installed_models"]["example-model"]["sha256"],
            package_sha256,
        )

    def test_brick_install_rejects_source_changed_after_update(self) -> None:
        model_source, package_sha256 = self._write_model_package(
            "source-model", b"model"
        )
        cached_info = self._info_json(model_source, package_sha256)
        source_path = self.root / "brick-source"
        source_path.mkdir()
        (source_path / "Example.cpp").write_text("", encoding="utf-8")
        (source_path / "Example.h").write_text("", encoding="utf-8")
        self._write_info_json(source_path, cached_info)
        self._write_cache(cached_info, installed=False)

        changed_info = copy.deepcopy(cached_info)
        changed_info["models"]["example-model"]["sha256"] = "4" * 64
        self._write_info_json(source_path, changed_info)

        with self.assertRaisesRegex(BrickManagerError, "changed after.*cached"):
            install_brick(self.state_paths, self.selector.identity)

        self.assertFalse(self.install_path.exists())

    def test_model_list_distinguishes_installed_and_cached_metadata(self) -> None:
        package_path, package_sha256 = self._write_model_package(
            "installed-model", b"installed"
        )
        installed_info = self._info_json(package_path, package_sha256)
        cached_info = copy.deepcopy(installed_info)
        cached_info["models"]["example-model"]["sha256"] = "0" * 64
        self._write_installed_brick(installed_info)
        model_path = self._copy_model_package(package_path)
        self._write_cache(
            cached_info,
            installed_models={
                "example-model": {
                    "path": str(model_path),
                    "ref": "local",
                    "sha256": package_sha256,
                }
            },
        )

        installed_list = get_brick_model_list(self.state_paths, self.selector.key)
        cached_list = get_brick_model_list(self.state_paths, self.selector.identity)

        self.assertEqual(installed_list.models[0].sha256, package_sha256)
        self.assertTrue(installed_list.models[0].installed)
        self.assertEqual(cached_list.models[0].sha256, "0" * 64)
        self.assertFalse(cached_list.models[0].installed)

    def test_update_preserves_installed_metadata_snapshot(self) -> None:
        model_source, installed_sha256 = self._write_model_package(
            "installed-source", b"installed"
        )
        installed_info = self._info_json(model_source, installed_sha256)
        self._write_installed_brick(installed_info)
        self._write_cache(installed_info)

        registry_source = self.root / "brick-source"
        registry_source.mkdir()
        cached_info = copy.deepcopy(installed_info)
        cached_info["models"]["cached-model"] = cached_info["models"].pop(
            "example-model"
        )
        self._write_info_json(registry_source, cached_info)
        registry_path = self.root / "registry.txt"
        registry_path.write_text(f"{registry_source}\n", encoding="utf-8")

        result = update_cache(self.state_paths, str(registry_path))

        cache = load_cache(self.state_paths.cache)
        entry = cache["bricks"][self.selector.key]["1.0.0"]
        self.assertEqual(entry["info_json"], cached_info)
        self.assertEqual(entry["metadata"]["installed_info_json"], installed_info)
        self.assertEqual(
            complete_cli(
                self.state_paths,
                ["model", "install", self.selector.key, ""],
            ),
            ["example-model"],
        )
        self.assertEqual(len(result.available_updates), 1)
        self.assertEqual(result.brick_versions, 1)
        self.assertEqual(
            result.available_updates[0].available_identity,
            self.selector.identity,
        )

        output = io.StringIO()
        render_update_result(
            result,
            Console(file=output, color_system=None, width=120, theme=THEME),
        )
        self.assertIn(
            f"compute-flow brick install {self.selector.identity} --force",
            output.getvalue(),
        )

    def test_update_retains_installed_version_when_new_version_is_indexed(self) -> None:
        model_source, installed_sha256 = self._write_model_package(
            "installed-source", b"installed"
        )
        installed_info = self._info_json(model_source, installed_sha256)
        self._write_installed_brick(installed_info)
        self._write_cache(installed_info)

        registry_source = self.root / "brick-source"
        registry_source.mkdir()
        available_info = copy.deepcopy(installed_info)
        available_info["version"] = "2.0.0"
        available_info["models"]["new-model"] = available_info["models"].pop(
            "example-model"
        )
        self._write_info_json(registry_source, available_info)
        registry_path = self.root / "registry.txt"
        registry_path.write_text(f"{registry_source}\n", encoding="utf-8")

        result = update_cache(self.state_paths, str(registry_path))

        cache = load_cache(self.state_paths.cache)
        self.assertEqual(
            sorted(cache["bricks"][self.selector.key]),
            ["1.0.0", "2.0.0"],
        )
        installed_entry = cache["bricks"][self.selector.key]["1.0.0"]
        self.assertTrue(installed_entry["metadata"]["installed"])
        self.assertEqual(
            installed_entry["metadata"]["installed_info_json"],
            installed_info,
        )
        self.assertEqual(
            [
                model.name
                for model in get_brick_model_list(
                    self.state_paths,
                    self.selector.key,
                ).models
            ],
            ["example-model"],
        )
        self.assertEqual(len(result.available_updates), 1)
        self.assertEqual(
            result.available_updates[0].installed_identity,
            self.selector.identity,
        )
        self.assertEqual(
            result.available_updates[0].available_identity,
            f"{self.selector.key}@2.0.0",
        )

    def test_update_does_not_report_unchanged_installed_brick(self) -> None:
        model_source, package_sha256 = self._write_model_package("model", b"model")
        info_json = self._info_json(model_source, package_sha256)
        self._write_installed_brick(info_json)
        self._write_cache(info_json)

        registry_source = self.root / "brick-source"
        registry_source.mkdir()
        self._write_info_json(registry_source, info_json)
        registry_path = self.root / "registry.txt"
        registry_path.write_text(f"{registry_source}\n", encoding="utf-8")

        result = update_cache(self.state_paths, str(registry_path))

        self.assertEqual(result.available_updates, [])

    def test_failed_replacement_preserves_existing_artifact(self) -> None:
        old_package, old_sha256 = self._write_model_package("old", b"old model")
        model_path = self._copy_model_package(old_package)
        installed_info = self._info_json(self.root / "missing", "1" * 64)
        self._write_installed_brick(installed_info)
        self._write_cache(
            installed_info,
            installed_models={
                "example-model": {
                    "path": str(model_path),
                    "ref": "local",
                    "sha256": old_sha256,
                }
            },
        )

        with self.assertRaisesRegex(BrickManagerError, "all sources"):
            install_models(
                self.state_paths,
                self.selector.key,
                ["example-model"],
            )

        self.assertEqual((model_path / "model.bin").read_bytes(), b"old model")

    def test_reconcile_removes_incompatible_and_undeclared_artifacts(self) -> None:
        stale_package, _ = self._write_model_package("stale", b"stale")
        undeclared_package, undeclared_sha256 = self._write_model_package(
            "undeclared", b"undeclared"
        )
        stale_path = self._copy_model_package(stale_package)
        undeclared_path = self._copy_model_package(
            undeclared_package, "undeclared-model"
        )
        installed_info = self._info_json(self.root / "missing", "2" * 64)
        self._write_cache(
            installed_info,
            installed_models={
                "example-model": {
                    "path": str(stale_path),
                    "ref": "local",
                    "sha256": "3" * 64,
                },
                "undeclared-model": {
                    "path": str(undeclared_path),
                    "ref": "local",
                    "sha256": undeclared_sha256,
                },
            },
        )
        cache = load_cache(self.state_paths.cache)
        entry = cache["bricks"][self.selector.key]["1.0.0"]

        result = reconcile_installed_models(
            self.state_paths,
            cache,
            self.selector,
            entry,
            installed_info,
        )

        self.assertEqual(result.incompatible, ["example-model"])
        self.assertEqual(result.removed, ["undeclared-model"])
        self.assertFalse(stale_path.exists())
        self.assertFalse(undeclared_path.exists())
        self.assertEqual(entry["metadata"]["installed_models"], {})

        write_cache(self.state_paths.cache, cache)
        self._write_installed_brick(installed_info)
        self.assertEqual(
            complete_cli(
                self.state_paths,
                ["model", "path", self.selector.key, ""],
            ),
            [],
        )
        self.assertEqual(
            complete_cli(
                self.state_paths,
                ["model", "uninstall", self.selector.key, ""],
            ),
            [],
        )

    def test_import_model_supports_multiple_artifacts_and_lifecycle(self) -> None:
        self._write_import_template()
        encoder = self.root / "new-encoder.onnx"
        decoder = self.root / "new-decoder.gguf"
        encoder.write_bytes(b"new encoder")
        decoder.write_bytes(b"new decoder")
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={
                    "encoder": parse_model_import_source(str(encoder)),
                    "decoder": parse_model_import_source(str(decoder)),
                },
                configs={"enabled": False, "threshold": 0.75},
            ),
        )

        manifest_path = result.path / "external-model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["models"]["encoder"]["filename"], "new-encoder.onnx")
        self.assertEqual(manifest["models"]["decoder"]["filename"], "new-decoder.gguf")
        self.assertEqual(manifest["models"]["encoder"]["runner"], "onnxrt")
        self.assertFalse(manifest["configs"]["enabled"]["value"])
        self.assertEqual(manifest["configs"]["threshold"]["value"], 0.75)
        self.assertEqual((result.path / "tokenizer.json").read_text(), "tokenizer")
        self.assertFalse((result.path / "encoder.old").exists())
        self.assertFalse((result.path / "decoder.old").exists())
        self.assertFalse((result.path / "SHA256SUMS").exists())

        model_list = get_brick_model_list(self.state_paths, self.selector.key)
        imported = next(
            model for model in model_list.models if model.name == "external-model"
        )
        self.assertEqual(imported.origin, "Imported")
        self.assertTrue(imported.installed)
        model_info = get_brick_model_info(
            self.state_paths,
            self.selector.key,
            ["external-model"],
        ).models[0]
        self.assertEqual(model_info.origin, "Imported")
        self.assertEqual(model_info.template, "example-model")
        self.assertEqual(
            get_brick_model_path(
                self.state_paths,
                self.selector.key,
                "external-model",
            ),
            result.path,
        )

        uninstall_models(
            self.state_paths,
            self.selector.key,
            ["external-model"],
        )

        self.assertFalse(result.path.exists())
        cache = load_cache(self.state_paths.cache)
        state = cache["bricks"][self.selector.key]["1.0.0"]["metadata"]
        self.assertNotIn("external-model", state["installed_models"])

    def test_import_downloads_url_only_during_install(self) -> None:
        self._write_import_template(single_artifact=True)
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "downloaded-model",
        )
        source = parse_model_import_source(
            "https://example.com/path/downloaded-model.onnx?download=1"
        )

        with mock.patch(
            "brick_manager.model_import.urllib.request.urlopen",
            return_value=io.BytesIO(b"downloaded"),
        ) as download:
            result = import_model(
                self.state_paths,
                preparation,
                ModelImportChoices(
                    artifacts={"encoder": source},
                    configs={},
                ),
            )

        download.assert_called_once_with(source.value)
        self.assertEqual(
            (result.path / "downloaded-model.onnx").read_bytes(), b"downloaded"
        )

    def test_import_uses_declared_source_when_template_is_not_installed(self) -> None:
        self._write_import_template(
            single_artifact=True,
            installed_template=False,
        )
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")

        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        self.assertIsNone(preparation.template_path)
        self.assertEqual(preparation.template_source["kind"], "local")
        self.assertEqual(
            complete_cli(
                self.state_paths,
                ["model", "import", self.selector.key, ""],
            ),
            ["example-model"],
        )
        phases = []
        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
            progress=lambda phase, artifact_key: phases.append(phase),
        )

        self.assertEqual(phases[0], "fetching_template")
        self.assertEqual((result.path / "tokenizer.json").read_text(), "tokenizer")
        cache = load_cache(self.state_paths.cache)
        installed_models = cache["bricks"][self.selector.key]["1.0.0"]["metadata"][
            "installed_models"
        ]
        self.assertNotIn("example-model", installed_models)
        self.assertEqual(installed_models["external-model"]["origin"], "imported")

    def test_import_selects_and_installs_brick_after_preparation(self) -> None:
        self._write_import_template(
            single_artifact=True,
            installed_template=False,
            installed_brick=False,
        )
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")

        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        self.assertFalse(self.install_path.exists())
        self.assertEqual(
            complete_cli(
                self.state_paths,
                ["model", "import", self.selector.key, ""],
            ),
            ["example-model"],
        )
        install_brick(
            self.state_paths,
            f"{preparation.brick}@{preparation.version}",
        )
        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
        )

        self.assertTrue(self.install_path.is_dir())
        self.assertTrue(result.path.is_dir())

    def test_import_selects_first_source_manifest_alphabetically(self) -> None:
        self._write_import_template(
            single_artifact=True,
            installed_template=False,
        )
        template_path = self.root / "import-template"
        manifest = json.loads(
            (template_path / "example_manifest.json").read_text(encoding="utf-8")
        )
        manifest["configs"]["threshold"]["value"] = 0.25
        (template_path / "a_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (template_path / "z_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        self.assertEqual(preparation.template_manifest_name, "a_manifest.json")
        threshold = next(
            config for config in preparation.configs if config.key == "threshold"
        )
        self.assertEqual(threshold.current_value, 0.25)
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")

        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
        )

        self.assertEqual(
            sorted(path.name for path in result.path.glob("*manifest*.json")),
            ["external-model_manifest.json"],
        )

    def test_import_selects_first_installed_manifest_alphabetically(self) -> None:
        self._write_import_template(single_artifact=True)
        template_path = (
            self.state_paths.models
            / self.selector.brick_type
            / self.selector.name
            / "example-model"
        )
        manifest = json.loads(
            (template_path / "example_manifest.json").read_text(encoding="utf-8")
        )
        (template_path / "a_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        self.assertEqual(preparation.template_manifest_name, "a_manifest.json")

    def test_import_does_not_require_template_checksums(self) -> None:
        self._write_import_template(single_artifact=True)
        template_path = (
            self.state_paths.models
            / self.selector.brick_type
            / self.selector.name
            / "example-model"
        )
        (template_path / "SHA256SUMS").unlink()
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")

        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )
        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
        )

        self.assertTrue(result.path.is_dir())
        self.assertFalse((result.path / "SHA256SUMS").exists())

    def test_failed_forced_import_preserves_previous_package(self) -> None:
        self._write_import_template(single_artifact=True)
        first_artifact = self.root / "first.onnx"
        first_artifact.write_bytes(b"first")
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )
        first_result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(first_artifact))},
                configs={},
            ),
        )
        second_artifact = self.root / "second.onnx"
        second_artifact.write_bytes(b"second")
        second_source = parse_model_import_source(str(second_artifact))
        second_artifact.unlink()
        forced = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
            force=True,
        )

        with self.assertRaises(OSError):
            import_model(
                self.state_paths,
                forced,
                ModelImportChoices(
                    artifacts={"encoder": second_source},
                    configs={},
                ),
            )

        self.assertEqual((first_result.path / "first.onnx").read_bytes(), b"first")

    def test_reconcile_preserves_compatible_imported_model(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )
        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
        )
        cache = load_cache(self.state_paths.cache)
        entry = cache["bricks"][self.selector.key]["1.0.0"]

        reconciliation = reconcile_installed_models(
            self.state_paths,
            cache,
            self.selector,
            entry,
            entry["info_json"],
        )

        self.assertEqual(reconciliation.removed, [])
        self.assertEqual(reconciliation.incompatible, [])
        self.assertTrue(result.path.is_dir())
        state = entry["metadata"]["installed_models"]["external-model"]
        self.assertEqual(state["origin"], "imported")

    def test_reconcile_removes_import_with_incompatible_brick_api(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "external.onnx"
        artifact.write_bytes(b"external")
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )
        result = import_model(
            self.state_paths,
            preparation,
            ModelImportChoices(
                artifacts={"encoder": parse_model_import_source(str(artifact))},
                configs={},
            ),
        )
        cache = load_cache(self.state_paths.cache)
        entry = cache["bricks"][self.selector.key]["1.0.0"]
        next_info = copy.deepcopy(entry["info_json"])
        next_info["api"] = "tts.v2"

        reconciliation = reconcile_installed_models(
            self.state_paths,
            cache,
            self.selector,
            entry,
            next_info,
        )

        self.assertEqual(reconciliation.incompatible, ["external-model"])
        self.assertFalse(result.path.exists())

    def test_import_parsing_and_non_interactive_error(self) -> None:
        self.assertTrue(parse_model_config_value("bool", "yes"))
        self.assertEqual(parse_model_config_value("int32", "42"), 42)
        with self.assertRaisesRegex(BrickManagerError, "int32 range"):
            parse_model_config_value("int32", str(2**31))
        with self.assertRaisesRegex(BrickManagerError, "finite number"):
            parse_model_config_value("float32", "nan")

        self._write_import_template(single_artifact=True)
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )
        with self.assertRaisesRegex(BrickManagerError, "interactive input"):
            prompt_model_import(
                preparation,
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
            )

    def test_interactive_import_collects_artifacts_and_config_overrides(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "interactive.onnx"
        artifact.write_bytes(b"interactive")
        preparation = prepare_model_import(
            self.state_paths,
            self.selector.key,
            "example-model",
            "external-model",
        )

        output = io.StringIO()
        choices = prompt_model_import(
            preparation,
            input_stream=_InteractiveInput(f"{artifact}\n\n0.9\ny\n"),
            output_stream=output,
        )

        self.assertEqual(choices.artifacts["encoder"].filename, "interactive.onnx")
        self.assertNotIn("enabled", choices.configs)
        self.assertEqual(choices.configs["threshold"], 0.9)
        self.assertIn(
            "Model artifact: encoder\nRunner: onnxrt\nCurrent file: encoder.old",
            output.getvalue(),
        )

    def test_json_string_import_preserves_multiline_configs_in_manifest(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "interactive.onnx"
        artifact.write_bytes(b"interactive")
        preparation = prepare_model_import(
            self.state_paths, self.selector.key, "example-model", "external-model"
        )
        expected = {
            "prompt_template": '<|im_start|>system\nCafé: "你好"\t\\n<|im_end|>\n',
            "assistant_prefix": "<think>\n\n</think>\n\n",
            "empty": "",
        }
        for name in (*expected, "keep"):
            preparation.manifest["configs"][name] = {
                "type": "string",
                "value": "original",
            }
            preparation.configs.append(ModelImportConfig(name, "string", "original"))
        answers = [str(artifact), "no", "0.9"]
        answers.extend(
            json.dumps(value, ensure_ascii=False) for value in expected.values()
        )
        answers.extend(["", "y", ""])

        choices = prompt_model_import(
            preparation,
            input_stream=_InteractiveInput("\n".join(answers)),
            output_stream=io.StringIO(),
        )
        result = import_model(self.state_paths, preparation, choices)

        manifest = json.loads(
            (result.path / "external-model_manifest.json").read_text(encoding="utf-8")
        )
        for name, value in expected.items():
            self.assertEqual(manifest["configs"][name]["value"], value)
        self.assertEqual(manifest["configs"]["keep"]["value"], "original")
        self.assertFalse(manifest["configs"]["enabled"]["value"])
        self.assertEqual(manifest["configs"]["threshold"]["value"], 0.9)

    def test_json_string_import_reprompts_for_invalid_values(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "interactive.onnx"
        artifact.write_bytes(b"interactive")
        preparation = prepare_model_import(
            self.state_paths, self.selector.key, "example-model", "external-model"
        )
        preparation.configs.append(ModelImportConfig("prompt", "string", "original"))
        for invalid in ('"bad\\q"', '"unterminated', '"text" trailing'):
            with self.subTest(value=invalid):
                output = io.StringIO()
                choices = prompt_model_import(
                    preparation,
                    input_stream=_InteractiveInput(
                        f'{artifact}\n\n\n{invalid}\n"valid\\ntext"\ny\n'
                    ),
                    output_stream=output,
                )

                self.assertEqual(choices.configs, {"prompt": "valid\ntext"})
                self.assertIn("Expected", output.getvalue())
                self.assertIn("JSON string", output.getvalue())

    def test_import_keeps_unquoted_string_inputs_literal(self) -> None:
        self._write_import_template(single_artifact=True)
        artifact = self.root / "interactive.onnx"
        artifact.write_bytes(b"interactive")
        preparation = prepare_model_import(
            self.state_paths, self.selector.key, "example-model", "external-model"
        )
        preparation.configs.append(ModelImportConfig("prompt", "string", "original"))
        for value in (
            r"C:\new\test",
            r"plain\ntext",
            'Say "hello"',
            "Café 你好",
            "null",
            "42",
        ):
            with self.subTest(value=value):
                choices = prompt_model_import(
                    preparation,
                    input_stream=_InteractiveInput(f"{artifact}\n\n\n{value}\ny\n"),
                    output_stream=io.StringIO(),
                )

                self.assertEqual(choices.configs, {"prompt": value})

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

    def _write_import_template(
        self,
        *,
        single_artifact: bool = False,
        installed_template: bool = True,
        installed_brick: bool = True,
    ) -> None:
        package_path, package_sha256 = self._write_model_package(
            "import-template", b"template"
        )
        (package_path / "encoder.old").write_bytes(b"old encoder")
        models = {
            "encoder": {
                "runner": "onnxrt",
                "filename": "encoder.old",
                "inputs": {"input": {"index": 0}},
            }
        }
        if not single_artifact:
            (package_path / "decoder.old").write_bytes(b"old decoder")
            models["decoder"] = {
                "runner": "llama_cpp",
                "filename": "decoder.old",
                "runner_params": {"context_size": 128},
            }
        manifest = {
            "brick": {"type": "tts", "name": "Example", "version": "1.0"},
            "models": models,
            "configs": {
                "enabled": {"type": "bool", "value": True},
                "threshold": {"type": "float32", "value": 0.5},
            },
            "assets": {"tokenizer": "tokenizer.json"},
        }
        (package_path / "example_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (package_path / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
        installed_info = self._info_json(package_path, package_sha256)
        installed_info["api"] = "tts.v1"
        if installed_brick:
            self._write_installed_brick(installed_info)
        else:
            brick_source = self.root / "brick-source"
            brick_source.mkdir()
            (brick_source / "Example.cpp").write_text("", encoding="utf-8")
            (brick_source / "Example.h").write_text("", encoding="utf-8")
            self._write_info_json(brick_source, installed_info)
        installed_path = (
            self._copy_model_package(package_path)
            if installed_template and installed_brick
            else None
        )
        installed_models = {}
        if installed_path is not None:
            installed_models["example-model"] = {
                "path": str(installed_path),
                "ref": "local",
                "sha256": package_sha256,
            }
        self._write_cache(
            installed_info,
            installed_models=installed_models,
            installed=installed_brick,
        )

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
        installed: bool = True,
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "info_json": info_json,
            "metadata": {
                "source": str(self.root / "brick-source"),
                "source_type": "local_directory",
                "source_path": str(self.root / "brick-source"),
                "manager_name": self.selector.name,
                "installed": installed,
                "installed_source_path": str(self.install_path) if installed else None,
                "installed_models": installed_models or {},
            },
        }
        write_cache(
            self.state_paths.cache,
            {"bricks": {self.selector.key: {"1.0.0": entry}}},
        )
        return entry

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
