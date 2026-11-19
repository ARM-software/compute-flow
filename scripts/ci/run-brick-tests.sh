#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail
export PYTHONPATH="$GITHUB_WORKSPACE/$COMPUTE_FLOW_CI_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
# TODO: Discover installed bricks with generated tests automatically.
bricks=(
  vad:silero-vad
  audio_detection:perch
  audio_tool_calling:neu-decide
  audio_to_audio:ulunas
  stt:moonshine
  audio_detection:nemotron3-diarization
  stt:parakeet-tdt
  stt:nemotron-asr
  lm_text:lm-text-generic
  monocular_depth_estimation:depth-anything-v3
  monocular_depth_estimation:zip-depth
  object_detection:yolo
  tts:qwen3-tts
  # TODO: Enable FLUX once its tests match the public model package;
  # flux2_klein_generation_4step_manifest.json is currently missing.
  # image_generation:flux2-klein-generation
  tts:pocket-tts
)
for brick in "${bricks[@]}"; do
  brick_path="$(uv run compute-flow brick path "$brick")"
  echo "::group::Test $brick"
  COMPUTE_FLOW_HOME="$COMPUTE_FLOW_TEST_MODEL_HOME" \
    uv run bash "$brick_path/test/run_tests.sh"
  echo "::endgroup::"
done
