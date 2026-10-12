#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Smoke tests for brick metadata and source layout."""

from __future__ import annotations

import copy
import re
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.cmake_generation import render_brick_cmake  # noqa: E402
from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.metadata import load_info_json, validate_info_json  # noqa: E402
from brick_manager.registry import (  # noqa: E402
    parse_registry,
    resolve_local_source,
)


class BrickMetadataTest(unittest.TestCase):
    def test_local_model_source_may_omit_ref(self) -> None:
        info = copy.deepcopy(
            load_info_json(REPOSITORY_ROOT / "bricks" / "tts" / "qwen3_tts")
        )
        model = next(iter(info["models"].values()))
        model["sources"] = [{"kind": "local", "url": "/models/example"}]

        validate_info_json(info)

    def test_remote_model_source_requires_ref(self) -> None:
        info = copy.deepcopy(
            load_info_json(REPOSITORY_ROOT / "bricks" / "tts" / "qwen3_tts")
        )
        model = next(iter(info["models"].values()))
        model["sources"] = [
            {"kind": "huggingface", "url": "https://huggingface.co/example/model"}
        ]

        with self.assertRaisesRegex(BrickManagerError, r"sources\[0\]\.ref"):
            validate_info_json(info)

    def test_model_entry_requires_sources(self) -> None:
        info = copy.deepcopy(
            load_info_json(REPOSITORY_ROOT / "bricks" / "tts" / "qwen3_tts")
        )
        model = next(iter(info["models"].values()))
        del model["sources"]

        with self.assertRaisesRegex(BrickManagerError, r"required field: .*\.sources"):
            validate_info_json(info)

    def test_brick_metadata_and_layout(self) -> None:
        info_paths = sorted((REPOSITORY_ROOT / "bricks").glob("*/*/info.json"))
        self.assertTrue(info_paths, "No brick metadata files found")

        for info_path in info_paths:
            with self.subTest(brick=info_path.parent.relative_to(REPOSITORY_ROOT)):
                info = load_info_json(info_path.parent)
                api = info.get("api")
                self.assertIsInstance(api, str)
                self.assertTrue(api)

                implementation = info_path.parent / f"{info['name']}.cpp"
                header = info_path.parent / f"{info['name']}.h"
                self.assertTrue(implementation.is_file(), implementation)
                self.assertTrue(header.is_file(), header)

                api_match = re.search(
                    r'api\(\).*?return\s+"([^"]+)"',
                    header.read_text(encoding="utf-8"),
                    flags=re.DOTALL,
                )
                self.assertIsNotNone(
                    api_match, f"Missing api() implementation in {header}"
                )
                self.assertEqual(api, api_match.group(1))

                for model_name, model in info["models"].items():
                    self.assertRegex(
                        model["sha256"],
                        r"^[0-9a-f]{64}$",
                        f"Invalid SHA256 for {model_name} in {info_path}",
                    )

    def test_registry_bricks_are_buildable_and_use_known_apis(self) -> None:
        registry_path = REPOSITORY_ROOT / "registry.txt"
        identities: set[tuple[str, str, str]] = set()

        for source in parse_registry(registry_path):
            with self.subTest(source=source):
                source_path = resolve_local_source(registry_path, source)
                info = load_info_json(source_path)
                identity = (info["type"], info["name"], info["version"])
                self.assertNotIn(identity, identities)
                identities.add(identity)

                api_path = REPOSITORY_ROOT / "brick_apis" / f"{info['api']}.json"
                self.assertTrue(api_path.is_file(), api_path)
                render_brick_cmake(source_path)


if __name__ == "__main__":
    unittest.main()
