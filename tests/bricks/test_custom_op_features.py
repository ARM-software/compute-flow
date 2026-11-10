#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Checks for the versioned custom operator CPU feature catalog."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "compute_flow" / "python"))

from brick_manager.custom_ops.features import (  # noqa: E402
    BUILD_FEATURES,
    FEATURE_CATALOG_VERSION,
    HW_FEATURES,
    compiler_flags,
)


class CustomOpFeaturesTest(unittest.TestCase):
    def test_catalog_has_settings_for_each_supported_build_feature(self) -> None:
        self.assertEqual(FEATURE_CATALOG_VERSION, 1)
        expected = {
            "neon-dotprod": "-march=armv8.2-a+dotprod",
            "neon-fp16": "-march=armv8.2-a+fp16",
            "neon-i8mm": "-march=armv8.2-a+i8mm",
            "sme": "-march=armv9.2-a+sme",
            "sme2": "-march=armv9.2-a+sme2",
            "sve": "-march=armv8.2-a+sve",
            "sve2": "-march=armv8.2-a+sve2",
        }
        self.assertEqual(HW_FEATURES, {"arm": frozenset(expected)})
        self.assertEqual(set(BUILD_FEATURES), {"arm"})
        self.assertEqual(set(BUILD_FEATURES["arm"]), set(expected))
        for name, flag in expected.items():
            self.assertEqual(compiler_flags("arm", [name], "GNU"), (flag,))

    def test_combines_features_in_one_stable_architecture_flag(self) -> None:
        features = ["neon-i8mm", "neon-fp16", "neon-dotprod"]
        expected = ("-march=armv8.2-a+dotprod+fp16+i8mm",)
        self.assertEqual(compiler_flags("arm", features, "GNU"), expected)
        self.assertEqual(compiler_flags("arm", reversed(features), "Clang"), expected)
        self.assertEqual(compiler_flags("arm", features, "AppleClang"), expected)
        self.assertEqual(
            compiler_flags("arm", ["sve2", "neon-i8mm"], "Clang"),
            ("-march=armv8.2-a+i8mm+sve2",),
        )
        self.assertEqual(
            compiler_flags("arm", ["sve2", "sve"], "GNU"),
            ("-march=armv8.2-a+sve+sve2",),
        )
        self.assertEqual(
            compiler_flags("arm", ["sme2", "neon-i8mm"], "Clang"),
            ("-march=armv9.2-a+i8mm+sme2",),
        )

    def test_empty_build_features_need_no_extra_flag(self) -> None:
        self.assertEqual(compiler_flags("arm", [], "GNU"), ())
        self.assertEqual(compiler_flags("arm", [], "MSVC"), ())

    def test_rejects_unsupported_target_feature_and_compiler(self) -> None:
        with self.assertRaisesRegex(ValueError, "target: x86"):
            compiler_flags("x86", [], "GNU")
        with self.assertRaisesRegex(ValueError, "feature: unknown"):
            compiler_flags("arm", ["unknown"], "GNU")
        with self.assertRaisesRegex(ValueError, "feature: sme2-mopa"):
            compiler_flags("arm", ["sme2-mopa"], "Clang")
        with self.assertRaisesRegex(ValueError, "compiler: MSVC"):
            compiler_flags("arm", ["neon-i8mm"], "MSVC")


if __name__ == "__main__":
    unittest.main()
