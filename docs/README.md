<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Documentation

Compute Flow documentation is grouped by purpose.

## Examples

Guided applications that put multiple development steps together:

- [Minimal C++ application](examples/cpp/minimal_example.md)
- [Minimal Python application](examples/python/minimal_example.md)
- [Calling desktop tools from speech with Python](examples/python/audio_tool_calling.md)
- [Audio-to-audio translation with Python](examples/python/audio_to_audio_translation.md)
- [Audio transcription with speaker labels and timestamps using Python](examples/python/transcribing_audio_with_speaker_labels.md)

C++ examples should show how to build and run on both native platforms and Android™.

## How-to guides

Steps for using Compute Flow and for developing the project:

- [Getting started](howto/getting_started.md)
- [Adding a third-party brick](howto/adding_3p_bricks.md)
- [Building from source](howto/building_from_source.md)
- [Combining bricks](howto/combining_bricks.md)
- [Running llama.cpp models](howto/running_llamacpp_models.md)
- [Capturing Streamline traces](howto/capturing_streamline_traces.md)
- [Updating manifest config](howto/updating_manifest_config.md)
- [Application development](howto/developing_apps_with_compute_flow.md)
- [Developing a brick](howto/developing_bricks.md)
- [Building prebuilt libraries](howto/building_prebuilt_libraries.md)
- [Docker build commands](howto/docker.md)

## Design

Architecture, requirements, design contracts, and decisions behind Compute Flow:

- [Brick API](design/brick_api.md)
- [Brick CLI examples](design/brick_cli_examples.md)
- [Brick error handling](design/brick_error_handling.md)
- [Brick Manager](design/brick_manager.md)
- [Brick manifest](design/brick_manifest.md)
- [Brick runtime interface](design/brick_runtime_interface_v1.md)
- [Brick structure](design/brick_structure.md)
- [Prebuilt binaries](design/prebuilt_binaries.md)
- [Profiler](design/profiler_design.md)
- [Runner requirements](design/runner_requirements.md)

## Reference

Conventions and lookup material:

- [C/C++ code conventions](reference/cpp_code_conventions.md)
- [Python code conventions](reference/python_code_conventions.md)
- [Prebuilt binary registry](reference/prebuilt_binary_registry.md)
- [llama.cpp tensor mapping](reference/llama_cpp_tensor_mapping.md)
- [Language model runner usage](reference/lm_runner_usage.md)
- [Language model sampling policies](reference/lm_sampling_policies.md)
- [Tokenizers](reference/tokenizers.md)

Images used by the documentation are in `imgs/`.

The [product overview](PRODUCT.md) is kept at the root of `docs/`.
