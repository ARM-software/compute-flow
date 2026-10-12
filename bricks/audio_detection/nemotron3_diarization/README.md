<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3 Diarization

Identifies who speaks when in audio, including overlapping speakers.

## Configuration

The `threshold` config selects the speaker activity probability above which a frame is active
(default `0.5`, range `0` to `1`). Set it before processing a recording. `update_config()` applies
changes to this threshold.

Streaming is enabled by default. Set `constants.enable_streaming_mode.value` to `false` in the
manifest before `setup()` to use offline mode. Call `setup()` again when changing this constant;
`update_config()` only updates runtime configs.

## Speaker names

The manifest's `class_names` asset maps display names to speaker IDs. The supplied JSON uses
`speaker_0`, `speaker_1`, and so on. To customize the displayed names, edit the names while keeping
the IDs unchanged, then call `setup()` again. IDs are local to each recording; assigning a display
name does not make the model recognize that person's voice.
