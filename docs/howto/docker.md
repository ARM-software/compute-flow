<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

### Build the base Docker image

#### Build base image (local)

```bash
# Optional: build and cache the heavy dependency layer once
docker buildx bake deps

# Main flow: build the base artefacts image (no entrypoint)
docker buildx bake
```

This produces `compute-flow-base:latest`. Runtime images can inherit from this image and define
their own entrypoints.

#### Build (Raspberry Pi® / linux/arm64)

```bash
docker buildx bake base-pi
```

This produces `compute-flow-base-arm64:latest`.
