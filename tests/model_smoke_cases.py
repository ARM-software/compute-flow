#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Canonical model package selected for each model-bearing brick variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODEL_SMOKE_CASES = (
    ("audio_detection", "Perch", "perch-v2-8bit"),
    ("audio_enhancement", "ULUNAS", "ulunas-frame-fp32-litert"),
    ("audiogen", "StableAudioOpenSmall", "stable-audio-open-small-fp32+int8"),
    ("image_edit", "InstructIR", "instructir-image-edit"),
    ("lm_text", "LMTextGeneric", "hy-mt2-1-8b-2bit-translation"),
    (
        "monocular_depth_estimation",
        "DepthAnythingV3",
        "da3-small-litert-fp16-compute",
    ),
    ("monocular_depth_estimation", "ZipDepth", "zipdepth-base-litert-f32"),
    ("object_detection", "Yolo", "yolo-v26n-litert-f16-int8w"),
    ("stt", "Moonshine", "moonshine-base-en"),
    ("stt", "ParakeetTDT", "parakeet-tdt-0-6b-v3-int8-dynamic"),
    ("tts", "NeuTTS", "neutts-nano-q4-0"),
    ("tts", "PocketTTS", "pocket-tts-en"),
    ("tts", "Qwen3TTS", "qwen3-tts-custom-voice"),
    ("vad", "SileroVAD", "silero-fp32"),
)


def validate_cases(repository_root: Path) -> None:
    declarations = {}
    for info_path in (repository_root / "bricks").glob("*/*/info.json"):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        declarations[(info["type"], info["name"])] = set(info.get("models", {}))

    for brick_type, brick_name, model_name in MODEL_SMOKE_CASES:
        declared_models = declarations.get((brick_type, brick_name))
        if declared_models is None:
            raise RuntimeError(f"Unknown smoke-test brick: {brick_type}:{brick_name}")
        if model_name not in declared_models:
            raise RuntimeError(
                f"Unknown smoke-test model for {brick_type}:{brick_name}: {model_name}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        validate_cases(arguments.repository_root)
        for model_case in MODEL_SMOKE_CASES:
            print("|".join(model_case))
        return 0
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Invalid model smoke cases: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
