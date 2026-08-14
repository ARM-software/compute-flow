#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Brick-type runners for model-backed CLI smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .audio_detection import run as run_audio_detection
from .audio_enhancement import run as run_audio_enhancement
from .audiogen import run as run_audiogen
from .image_edit import run as run_image_edit
from .lm_text import run as run_lm_text
from .monocular_depth_estimation import run as run_depth_estimation
from .object_detection import run as run_object_detection
from .stt import run as run_stt
from .tts import run as run_tts
from .vad import run as run_vad


Runner = Callable[[Path, Path, Path, Path], None]

RUNNERS: dict[str, Runner] = {
    "audio_detection": run_audio_detection,
    "audio_enhancement": run_audio_enhancement,
    "audiogen": run_audiogen,
    "image_edit": run_image_edit,
    "lm_text": run_lm_text,
    "monocular_depth_estimation": run_depth_estimation,
    "object_detection": run_object_detection,
    "stt": run_stt,
    "tts": run_tts,
    "vad": run_vad,
}
