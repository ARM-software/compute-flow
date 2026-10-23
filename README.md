<!--
    SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Compute Flow

Compute Flow is an SDK for building AI and GenAI applications from reusable building blocks called
**bricks**.

Compute Flow brings together inference runtimes such as LiteRT, ExecuTorch, ONNX Runtime, and
llama.cpp so new bricks can be created without depending directly on each runtime API. It is
designed for Arm® edge AI inference across platforms, including **Android™**, **Linux®**, and
**macOS®** targets. Microcontrollers are not yet supported.

## Project Status

Compute Flow is a prototype, so APIs and functionality may change. We welcome feedback and issue
reports.

## What can you do with Compute Flow?

Compute Flow helps you build on-device AI applications by:

1. [**Discovering, installing, and testing bricks**](docs/howto/getting_started.md)

   With Brick Manager, you can discover, install, and test bricks for speech recognition, language
   models, text-to-speech, and object detection.

2. [**Building AI pipelines**](docs/examples/python/audio_to_audio_translation.md)

   Combine multiple bricks into AI processing pipelines. For example, build an audio-to-audio
   translation pipeline in Python by connecting speech-to-text, translation, and text-to-speech
   bricks.

3. [**Creating reusable bricks**](docs/howto/developing_bricks.md)

   Build new bricks with models from different inference runtimes, and make them easy for the
   community to discover and reuse. You can use the Brick Manager CLI to **import bricks** from
   GitHub URLs.

## Getting started

Use `uv` to create the Python environment used by the Compute Flow tools:

```bash
uv sync && source .venv/bin/activate
```

If you do not have `uv` installed, follow the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/#pypi).

Check the local environment before installing or building bricks:

```bash
compute-flow check
```

If `compute-flow check` reports missing system dependencies, install them using the suggested action
and run the command again.

## Using bricks

The Compute Flow brick manager discovers bricks, installs the brick inference code, installs model
packages, and builds Compute Flow for the selected target. A typical native workflow is:

1. Update the local brick registry cache.
2. Install the brick and one of its models.
3. Build Compute Flow for the target platform.
4. Run the Python or C++ CLI example.

This example installs the pre-packaged Qwen3-TTS text-to-speech (TTS) brick in Compute Flow and runs
the native CLI examples.

```bash
compute-flow update
compute-flow model install tts:qwen3-tts qwen3-tts-0-6b-custom-voice
compute-flow build native
```

### Shell completion

Shell completion works with the Brick Manager and the CLI examples after activating `.venv`:
<img src="docs/imgs/compute_flow_tab_completion.gif"
alt="Terminal demonstration of Brick Manager and brick CLI completion"
width="800">

- The Compute Flow Brick Manager completes commands, brick types, brick names, and models.
- The Python and C++ CLI examples complete installed `@manifest/...` and `@sample/...` selectors.

Run the Python CLI example with completion support:

```bash
python -m cli_examples.tts_cli \
  -m @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_en_manifest.json \
  -p "Hello, I am Aiden, your personal AI assistant. How can I help?" \
  -t 1
```

Or run the C++ CLI example with the same arguments:

```bash
./build_native/tts_cli \
  -m @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_en_manifest.json \
  -p "Hello, I am Aiden, your personal AI assistant. How can I help?" \
  -t 1
```

`@manifest` selects a manifest from an installed model package.

For a full native and Android™ walkthrough, see the
[getting started guide](docs/howto/getting_started.md).

### Using bricks from the open source community

In addition to the pre-packaged bricks included with Compute Flow, you can install
community-developed bricks from Git repositories.

To add a community Git repository and install one of its brick models, run:

```bash
compute-flow registry add https://example.com/team/example-tts.git
compute-flow update
compute-flow model install tts:example-tts example-model
```

Replace the example URL, brick identifier, and model name with those provided by the brick author.
For further information, see the
[guide to adding a third-party brick](docs/howto/adding_3p_bricks.md).

### Supporting material

#### [How-to guides](docs/howto/)

| Document                                                               | Description                                           |
| ---------------------------------------------------------------------- | ----------------------------------------------------- |
| [**Getting started**](docs/howto/getting_started.md)                   | Build and run a brick on native and Android™ targets  |
| [**Adding a 3rd party brick**](docs/howto/adding_3p_bricks.md)         | Register and install a brick from a Git repository    |
| [**Developing a brick**](docs/howto/developing_bricks.md)              | Prepare a model package and implement a brick         |
| [**Building from source**](docs/howto/building_from_source.md)         | Build dependencies from upstream sources              |
| [**Running llama.cpp models**](docs/howto/running_llamacpp_models.md)  | Import and run a llama.cpp-compatible GGUF model      |
| [**Updating manifest config**](docs/howto/updating_manifest_config.md) | Change runtime config values without reloading models |

#### [Examples](docs/examples/)

| Document                                                                                                             | Description                                         |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| [**Minimal C++ application**](docs/examples/cpp/minimal_example.md)                                                  | Build a minimal application with the packaged SDK   |
| [**Python audio tool calling**](docs/examples/python/audio_tool_calling.md)                                          | Run desktop tools from spoken requests              |
| [**Python translation pipeline**](docs/examples/python/audio_to_audio_translation.md)                                | Audio-to-audio translation with Python              |
| [**Python podcast generation**](docs/examples/python/generating_a_podcast.md)                                        | Generate a two-speaker podcast from a text file     |
| [**Python audio transcription with speaker labels**](docs/examples/python/transcribing_audio_with_speaker_labels.md) | Transcribe audio with speaker labels and timestamps |

Other development, design, and reference documents are listed in the
[documentation index](docs/README.md).

## Support

For help, bug reports, or feature requests, please
[open an issue](https://github.com/ARM-software/compute-flow/issues).

## Contributing

We are not currently accepting contributions to the `compute_flow` core, but we welcome ideas,
feature requests, issue reports, and other feedback through the project's
[issue tracker](https://github.com/ARM-software/compute-flow/issues).

See the [contribution guidelines](CONTRIBUTING.md) for details.

## Acknowledgements

We thank everyone who has helped shape Compute Flow. See [Acknowledgements](ACKNOWLEDGEMENTS.md) for
recognition of their contributions.

## Trademarks

See the [trademark reference](TRADEMARKS/README.md) for the canonical body forms, legal notices, and
usage rules for names used throughout the project.

## License

Compute Flow is licensed under the [Apache License 2.0](LICENSES/Apache-2.0.txt).

Third-party components remain subject to their respective licenses. See
[third-party notices](THIRD_PARTY_NOTICES.md) for the direct dependency inventory.
