<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Runner requirements

This document captures runner-related requirements. It will expand over time.

## set_input(): Shape Requirements

When calling the `set_input()` method on runners (or `lm_runner` adapters), users must provide a
fully static shape. Any dimension marked as `-1` is treated as dynamic and is not accepted for
user-provided inputs.

Why this matters:

- Runners validate input byte sizes against model expectations for static shapes.
- Dynamic shapes are only resolved internally by the runtime and cannot be verified safely at
  `set_input` time.

## MNN Runner

The MNN runner selects its backend from the first accelerator preference. CPU execution applies a
non-zero `num_threads` value to the MNN session. GPU execution is available on Android™ only, uses
OpenCL, and currently supports only device index `0`.

For either CPU or GPU execution, the runner selects MNN low-precision FP16 computation when the
model contains native FP16 weights. This includes FP16 constants and convolution weights produced by
MNNConvert's `--fp16` option. CPU callers can explicitly request FP16 computation with
`CpuOptions::enable_fp16`. Models without FP16 weights otherwise use MNN normal precision.

MNN input and output indices are derived from the lexicographically ordered tensor-name maps
returned by MNN. The same ordering must be used by the model manifest.
