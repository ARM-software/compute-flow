<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

### Docker builds (base + bricks examples)

#### Build base image (local)

```bash
# Optional: build and cache the heavy dependency layer once
docker buildx bake deps

# Main flow: build the base artefacts image (no entrypoint)
docker buildx bake
```

This produces `compute-flow-base:latest`. Runtime images should inherit from
`compute-flow-base:latest` and define their own entrypoints.

#### Build runtime image (Minimal example)

```bash
docker buildx bake example
```

This produces a local image tagged `compute-flow-example:local`. The example Dockerfile runs the
Kokoro TTS CLI by default.

#### Build (Raspberry Pi® / linux/arm64)

```bash
docker buildx bake base-pi
docker buildx bake example-pi
```

This produces `compute-flow-base-arm64:latest` and `compute-flow-example-arm64:latest`.

#### Run (Bricks Example)

Mount your model pack directory as `/models`:

```bash
docker compose -f .docker/compose.yml up
```
