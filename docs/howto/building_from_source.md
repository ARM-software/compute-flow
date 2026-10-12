<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Building from source

Use the Compute Flow Brick Manager to build Compute Flow and its runtime dependencies from source
instead of downloading prebuilt release archives. The first build still needs network access because
the build downloads pinned upstream source repositories and archives directly.

## Prerequisites

- `uv` and the system dependencies reported by `compute-flow check`.
- At least 25 GiB of free disk space for downloaded source, intermediate objects, and libraries. A
  clean validation build used approximately 18 GiB after completion.
- A stable 100 Mbit/s or faster internet connection is recommended.

## Prepare the Python environment

From the Compute Flow project root, create the repository environment and check its system
dependencies:

```bash
uv sync
uv run compute-flow check
```

List the available build targets and select the one required for the build:

```bash
uv run compute-flow build list
```

## Build from source

Replace `TARGET` with one of the names reported by `compute-flow build list`:

```bash
uv run compute-flow build TARGET --from-source
```

The Brick Manager also enables this source workflow automatically when prebuilt release archives are
unavailable. Use `--from-source` to skip the release download and force the source workflow.

The command defaults to 4 parallel jobs. To set the value explicitly or use more jobs, pass `--jobs`
or `-j`:

```bash
uv run compute-flow build TARGET --from-source --jobs 4
```

The `--from-source` option makes the Brick Manager:

- disable prebuilt dependency downloads;
- build Protobuf, LiteRT, ExecuTorch, ONNX Runtime, MNN, and SentencePiece from their pinned
  sources;
- apply the selected job count to the main and nested dependency builds;
- configure and compile Compute Flow; and
- prepare the application SDK.

Other dependencies, including llama.cpp and the Hugging Face tokenizer, already use their source
fallback when an installed package is unavailable. Warnings from upstream projects are expected.

## Verify the build

A successful command exits with a zero status and reports the build directory and application SDK
directory. The SDK directory contains the public headers and built libraries needed by an
application.

Always include `--from-source` when continuing this workflow. Running the command without it
restores the default prebuilt dependency configuration in the same target build directory.

## Troubleshooting

If ExecuTorch reports that Python cannot find `torch`, sync the environment again and rerun the same
Brick Manager build command:

```bash
uv sync
uv run compute-flow build TARGET --from-source
```

If a direct source download is interrupted, rerun the same command:

```bash
uv run compute-flow build TARGET --from-source
```
