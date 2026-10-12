<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Brick CLI examples

The CLI examples in this document are intentionally simple and are not intended to demonstrate every
available feature. Their primary goal is to help users quickly learn how to run a brick, observe its
behavior, and evaluate its performance with minimal setup.

In addition to serving as learning tools, the examples are also designed to enable quick profiling
and benchmarking of brick executions. Common performance-related options, such as CPU thread
configuration and operator profiling, are exposed directly through the CLI so that users can easily
experiment with different runtime configurations and gather performance data.

To keep the examples easy to understand, some advanced or less commonly used options are omitted.
The focus is on providing a straightforward starting point for learning, validation, and performance
evaluation rather than a complete reference for all CLI capabilities.

All brick CLIs share a common set of arguments, such as the manifest path, CPU thread configuration,
and operator profiling options. In addition, each brick type provides its own brick-specific
arguments (for example, prompts for LLM and TTS bricks, audio inputs for STT and audio embedding
bricks, or image inputs for vision bricks).

The examples below illustrate common usage patterns and demonstrate the shared CLI arguments. Refer
to the documentation for the specific brick type for a complete description of the available options
and features. The CLI examples are model-agnostic and apply to any model that implements the
selected brick type.

The audio detection CLIs display up to three classes per detected event by default. Set `-k` to a
positive maximum to display more or fewer. If a brick reports fewer classes, the CLIs display all of
them.

## Examples of use

All brick CLIs support a manifest path, optional CPU thread configuration, and optional op
profiling. Add the required brick-specific flags for the selected brick type, such as prompt, audio,
or image input flags.

### Run multiple inputs without reloading the model

The C++ and Python CLI examples accept `-j <runs.json>`. The file must contain a non-empty JSON
array whose objects use the CLI's short argument names:

```json
[
  {
    "id": "first run",
    "<short-input-flag>": "<value>"
  },
  {
    "id": "second run",
    "<short-input-flag>": "<value>",
    "<short-optional-flag>": "<value>"
  }
]
```

Run the file with the C++ or Python CLI:

```sh
./<brick_type>_cli -m <manifest> -j runs.json
python -m cli_examples.<brick_type>_cli -m <manifest> -j runs.json
```

`id` is a required non-empty label printed before the entry. Each entry must also contain the CLI's
required input arguments and can contain optional run arguments. The CLI loads and sets up the
selected brick once, resets its state before each entry, and runs the entries in array order.
Console results and profiling information are displayed after each entry.

`-m`, `-t`, and `--enable-op-profiling` remain command-line arguments because they configure the
shared setup rather than an individual run.

Input paths in a runs file must be ordinary filesystem paths. The `@sample/...` selector is not
accepted; it remains available when the argument is passed directly on the command line.

### CLI manifest completion:

When models are installed with the Compute Flow Brick Manager, brick CLIs can use shell completion
to select an installed manifest without manually looking up the model directory.

Installed manifest selectors use the `@manifest/` namespace. Shell completion completes them
progressively, while the brick type is inferred from the typed CLI name:

```sh
./tts_cli -m <TAB>
# @manifest/

./tts_cli -m @manifest/<TAB>
# @manifest/qwen3-tts/

./tts_cli -m @manifest/qwen3-tts/<TAB>
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/

./tts_cli -m @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/<TAB>
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_en_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_es_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_fr_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_it_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_jp_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_ko_manifest.json
# @manifest/qwen3-tts/qwen3-tts-0-6b-custom-voice/qwen3_tts_custom_voice_zh_manifest.json
```

Installed selector completions include the manifest's `.json` extension. The CLI resolves the
selector internally through the brick type, so
`tts_cli -m @manifest/qwen3-tts/.../qwen3_tts_custom_voice_en_manifest.json` loads the installed TTS
manifest.

Model packages may include optional CLI input files under a `samples/` directory beside the
manifest. Read-only file arguments use the `@sample/` namespace to select these package files:

```sh
./vad_cli -m @manifest/silero-vad/silero-fp32/silero_manifest.json -a <TAB>
# @sample/

./vad_cli -m @manifest/silero-vad/silero-fp32/silero_manifest.json -a @sample/<TAB>
# @sample/beckett.wav
```

The same namespace applies to `-i` image inputs. Sample completion uses the package selected by
`-m`, regardless of argument order. Output arguments such as `-o` remain ordinary filesystem paths
and never resolve through `@sample/`.

Users can still provide ordinary input paths:

```sh
./vad_cli -m @manifest/silero-vad/silero-fp32/silero_manifest.json -a ./audio.wav
```

For Bash and zsh, the same manifest completion works whenever a previous command-line token is a
brick CLI, including Python module commands such as `python -m cli_examples.tts_cli -m <TAB>` and
C++ binaries such as `./build_native/tts_cli -m <TAB>`.

For local development, `uv sync` installs the completion script into the project virtual environment
and patches activation. Completion is enabled when the virtual environment is activated:

```sh
source .venv/bin/activate
```

The activation script loads the source completion script from a development checkout when available,
or the installed copy from the virtual environment:

```text
$VIRTUAL_ENV/../scripts/completion/brick_cli_completion.sh
$VIRTUAL_ENV/share/compute-flow/brick_cli_completion.sh
```

Completion reads installed manifests directly from `$COMPUTE_FLOW_HOME/models` or
`~/.compute-flow/models`. It does not use a separate completion registry and does not require a
refresh command after model installation.

Users can also provide the full path to the manifest:

```sh
./<brick_type>_cli -m /path/to/manifest.json ..
```

### Default number of threads to 1:

```sh
./<brick_type>_cli -m <manifest> ..
```

### All models with 2 threads:

```sh
./<brick_type>_cli -m <manifest> -t 2 ..
```

### All models with 1 thread, except `model_name0` and `model_name1`:

```sh
./<brick_type>_cli -m <manifest> -t 1,<model_name0>:2,<model_name1>:4 ..
```

### Enable op profiling:

```sh
./<brick_type>_cli -m <manifest> --enable-op-profiling ..
```

### Save generated text to a file:

The `lm_text_cli` example accepts an output path with any extension. When `-o` is present, only the
generated text is written to the file; prompts and profiling information remain on standard output.

```sh
./lm_text_cli -m <manifest> -p "Write a C hello-world program" -o hello.c
```
