<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Run llama.cpp models with Compute Flow

## Goal

Demonstrate how to run llama.cpp-compatible models with Compute Flow.

> **Note:** `compute-flow model import` also supports models for other bricks and runners when they
> match the brick's
> [template manifest](../design/brick_manager.md#compute-flow-model-import-typename-template-model-new-model-name).

## Requirements

- A native Linux® or macOS® platform.
- `uv`, used to create the Python environment.
- `git`, used by the Compute Flow Brick Manager.

Run all commands from the Compute Flow project root.

## Set up Compute Flow

Create and activate the Python environment:

```bash
uv sync
source .venv/bin/activate
```

Check the required system dependencies and update the brick registry:

```bash
compute-flow check
compute-flow update
```

If `compute-flow check` reports a missing dependency, install it and run the check again.

## Model import

Run the interactive model import command. Replace `my-model` with a name for the imported model:

```bash
compute-flow model import \
  lm_text:lm-text-generic \
  hy-mt2-1-8b-2bit-translation \
  my-model
```

- At the model artifact prompt, enter a local GGUF file path or an HTTP/HTTPS URL.

  For example:

  ```text
  https://huggingface.co/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/resolve/main/Qwen__Qwen3-0.6B-Base_llamacpp_original.gguf
  ```

  This URL downloads the [Qwen3 0.6B Base GGUF model][qwen3-model].

- At the `prompt_template` config prompt, enter the template required by the model. It must contain
  `<<USER_TEXT>>`, which Compute Flow replaces with the input prompt. For the example model, enter
  `<<USER_TEXT>>`.

- At the other config prompts, press Enter to keep the existing values.

- Confirm the summary to install the model.

The GGUF file must be supported by llama.cpp and compatible with the settings copied from the
template model.

Check the imported model:

```bash
compute-flow model list lm_text:lm-text-generic
compute-flow model info lm_text:lm-text-generic my-model
```

## Build and run

Build Compute Flow and the installed brick for the current machine:

```bash
compute-flow build native
```

Run the Python CLI example:

```bash
python -m cli_examples.lm_text_cli \
  -m @manifest/lm-text-generic/my-model/my-model_manifest.json \
  -p "Write a short poem about the sea." \
  -n 128
```

Or run the C++ CLI example with the same model and prompt:

```bash
./build_native/lm_text_cli \
  -m @manifest/lm-text-generic/my-model/my-model_manifest.json \
  -p "Write a short poem about the sea." \
  -n 128
```

[qwen3-model]: https://huggingface.co/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/blob/main/Qwen__Qwen3-0.6B-Base_llamacpp_original.gguf
