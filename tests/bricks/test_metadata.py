#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for Brick Manager metadata validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.errors import BrickManagerError  # noqa: E402
from brick_manager.metadata import validate_info_json  # noqa: E402


class BrickMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = {
            "description": "Example model package.",
            "sources": [
                {
                    "kind": "huggingface",
                    "url": "https://huggingface.co/example/model",
                    "ref": "1.0.0",
                }
            ],
            "sha256": "0" * 64,
        }
        self.info = {
            "name": "ExampleTTS",
            "type": "tts",
            "api": "tts.v1",
            "description": "Example text-to-speech brick.",
            "author": "Example Publisher",
            "version": "1.0.0",
            "models": {"example-model": self.model},
        }

    def test_local_model_source_may_omit_ref(self) -> None:
        self.model["sources"] = [{"kind": "local", "url": "/models/example"}]

        validate_info_json(self.info)

    def test_remote_model_source_requires_ref(self) -> None:
        self.model["sources"] = [
            {"kind": "huggingface", "url": "https://huggingface.co/example/model"}
        ]

        with self.assertRaisesRegex(BrickManagerError, r"sources\[0\]\.ref"):
            validate_info_json(self.info)

    def test_model_entry_requires_sources(self) -> None:
        del self.model["sources"]

        with self.assertRaisesRegex(BrickManagerError, r"required field: .*\.sources"):
            validate_info_json(self.info)


if __name__ == "__main__":
    unittest.main()
