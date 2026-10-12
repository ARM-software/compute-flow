<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-TTS

## Performance Optimization

Qwen3-TTS uses 16 active code prediction groups by default. To improve runtime performance, you can
try reducing `num_active_code_pred_groups` in the manifest file.

Reduce the active group count gradually and check the generated audio after each change. Lower
values can improve performance, but may reduce output quality. Choose the smallest value that meets
your performance target while keeping acceptable audio quality.
