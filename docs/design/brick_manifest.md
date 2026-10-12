<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Brick Manifest Design

## Table of Contents

- [Overview](#overview)
- [Design Goals](#design-goals)
- [The Brick Manifest - The JSON file](#the-brick-manifest---the-json-file)
- [Brick section](#brick-section)
  - [Structure](#structure)
  - [Fields](#fields)
- [Models section](#models-section)
  - [Structure](#structure-1)
  - [Fields](#fields-1)
- [Configs section](#configs-section)
  - [Structure](#structure-2)
  - [Template expressions](#template-expressions)
- [Constants section](#constants-section)
  - [Structure](#structure-3)
  - [Template expressions](#template-expressions-1)
- [Assets section](#assets-section)
  - [Structure](#structure-4)
- [Sample inputs](#sample-inputs)

## Overview

The manifest defines a **self-contained, executable inference pipeline** composed of multiple
models, assets, and a well-defined input/output contract.

It is designed to support multiple domains, including (but not limited to):

- Text-to-Speech (TTS)
- Speech-to-Text (STT)
- Audio Generation
- Object detection

The manifest is:

- **Runtime-agnostic** (supports multiple backends, such as LiteRT, ExecuTorch, ONNXRT, and
  llama.cpp)
- **Hardware-aware** (optional execution hints, particularly for AI accelerators)
- **Composable** (multiple models forming a pipeline)
- **Self-describing** (explicit I/O and configuration)

## Design Goals

To define the design goals of the brick manifest, it is important to consider Compute Flow’s vision:

> Compute Flow is designed for easy prototyping, rapid development, and education, making it an
> ideal platform to demonstrate and profile edge AI solutions on Arm®.

Based on this vision, the manifest should be designed to be:

- **easily modified**
- **easy to read**
- **easy to extend**

______________________________________________________________________

## The Brick Manifest - The JSON file

A brick manifest is a JSON file that describes the **brick package**, the directory or archive
containing everything needed to load and run a brick.

The brick manifest lists the implementation-specific resources needed to run the brick, such as
models, configuration values, constants, and assets.

At a high level, the manifest has the following structure:

```json
{
  "brick":  { ... },
  "models": { ... },
  "configs": { ... },
  "constants": { ... },
  "assets": { ... },
}
```

An example brick manifest is shown below:

Example:

```json
{
  "brick": {
    "type": "audio_detection",
    "name": "Perch",
    "version": "1.0"
  },
  "models": {
    "classifier": {
      "runner": "litert",
      "filename": "model.tflite",
      "inputs": {
        "audio": {
          "index": 0,
          "dtype": "float32",
          "shape": [-1, 160000]
        }
      },
      "outputs": {
        "label": {
          "index": 1,
          "dtype": "float32",
          "shape": [-1, 14795]
        }
      },
      "hw_hint": ["gpu"]
    }
  },
  "configs": {
    "frame_size_s": {
      "type": "int64",
      "value": 2
    }
  },
  "constants": {
    "sample_rate": {
      "type": "int64",
      "value": 32000
    }
  },
  "assets": {
    "class_names": "class_names.json"
  }
}
```

The following sections describe each manifest section in more detail.

## Brick section

- **Required:** yes
- **Mutable:** no
- **Description:** Defines the brick identity, including the brick type, the brick name, and the
  brick manifest/package version.

### Structure

```json
"brick": {
  "type": "<string>",
  "name": "<string>",
  "version": "<string>"
}
```

### Fields

#### Type (`type`)

Identifies the brick implementation type.

The manifest must not identify a brick API version. Brick APIs are properties of the brick
implementation, not the model package manifest. If a brick implementation changes or upgrades its
API, the manifest should not change for that reason.

For example:

```json
"type": "audio_detection"
```

#### Name (`name`)

Identifies the specific brick implementation of that brick type. This value must match the brick
`info.json` `name` exactly. For example, `Qwen3TTS` and `PocketTTS` are two text-to-speech (TTS)
brick names.

The runtime validates this field against the brick implementation metadata. A manifest whose
`brick.name` does not match the brick `info.json` `name` is invalid and must fail validation/load.

#### Version (`version`)

Identifies the version of the brick manifest/package. It is used by the brick implementation to know
whether the manifest is the correct one.

## Models section

- **Required:** yes
- **Mutable:** no
- **Description:** Defines the model files required by the brick and the information needed to
  execute them.

The models section lists all models used internally by the brick implementation. Each model entry
describes the execution backend, model file, model tensors, optional state tensors, and optional
hardware preferences.

The models listed in this section are implementation details and not necessarily they are part of
the public brick API.

### Structure

Each model is defined independently under a model name.

```json
"models": {
  "<model_name>": {
    "runner":   "<runtime>",
    "filename": "<model_file>",
    "inputs":   { ... },
    "outputs":  { ... },
    "states":   { ... },
    "runner_params": { ... },
    "hw_hint":  [ ... ]
  }
}
```

For example:

```json
"models": {
  "classifier": {
    "runner": "litert",
    "filename": "model.tflite",
    "inputs": {
      "audio": {
        "index": 0,
        "dtype": "float32",
        "shape": [-1, 160000]
      }
    },
    "outputs": {
      "label": {
        "index": 1,
        "dtype": "float32",
        "shape": [-1, 14795]
      }
    },
    "runner_params": {},
    "hw_hint": ["gpu"]
  }
}
```

Each model is connected by the brick implementation and runtime logic. The manifest does not
describe a computation graph or the data flow between models.

### Fields

#### Runners (`runner`)

Specifies the execution backend used to run the model.

Examples:

- `onnxrt`
- `llama_cpp`
- `litert`
- `executorch`
- `mnn`
- `hf_tokenizer`
- `sentencepiece_enc`
- `sentencepiece_dec`

Example:

```json
"runner": "litert"
```

#### Filename (`filename`)

Specifies the model file to load.

The value must refer to a model file included in the brick package. Relative paths are resolved from
the directory containing the manifest.

For example, if the manifest is located at:

```
<your-path>/<model-name>/manifest.json
```

then:

```json
"filename": "model.tflite"
```

resolves to:

```
<your-path>/<model-name>/model.tflite
```

______________________________________________________________________

#### Inputs (`inputs`)

Defines the input tensors expected by the model.

Model inputs use the same tensor-description format as brick API tensors:

```json
"<tensor_name>": {
  "index": 0,
  "dtype": "float32",
  "shape": [-1, 160000]
}
```

Attention should be paid to the tensor `index` field. The index must be the tensor index used by the
runner to access that tensor.

For example, when using the `litert` runner, the index should match the tensor index required to
retrieve or bind the tensor through LiteRT. This index may differ from the tensor name or from other
identifiers stored in the model file.

For the `mnn` runner, input indices follow the lexicographical order of the tensor names returned by
MNN. Index `0` selects the first input tensor name in that order. Use the same ordering when writing
an MNN model manifest.

#### Outputs (`outputs`)

Defines the output tensors produced by the model.

Model outputs use the same tensor-description format as brick API tensors:

```json
"<tensor_name>": {
  "index": 0,
  "dtype": "float32",
  "shape": [-1, 1536]
}
```

Again, attention should be paid to the tensor `index` field. The index must be the tensor index used
by the runner to access that tensor.

For example, when using the `litert` runner, the index should match the tensor index required to
retrieve or bind the tensor through LiteRT. This index may differ from the tensor name or from other
identifiers stored in the model file.

For the `mnn` runner, output indices follow the lexicographical order of the tensor names returned
by MNN. Index `0` selects the first output tensor name in that order.

#### States (`states`)

Defines optional state tensors used by stateful models.

State tensors represent values that must be carried across model invocations. They are commonly used
by streaming, autoregressive, recurrent, or cached models. A state tensor is produced as an output
by one invocation and passed back as an input to a later invocation.

- `index_in`: input tensor index used to feed the state into the model
- `index_out`: output tensor index used to read the updated state from the model

The indices must match the tensor indices used by the selected runner to access those tensors.

Example:

```json
"states": {
  "upsample_partial": {
    "index_in": 19,
    "index_out": 2,
    "dtype": "float32",
    "shape": [1, 512, 16]
  },
  "decoder_transformer_layer_0_offset": {
    "index_in": 3,
    "index_out": 15,
    "dtype": "int64",
    "shape": [1]
  },
  "decoder_transformer_layer_0_cache": {
    "index_in": 4,
    "index_out": 9,
    "dtype": "float32",
    "shape": [2, 1, 250, 8, 64]
  }
}
```

A full state definition may contain many entries, one for each state value that must be preserved
between runs.

During execution, the runtime should read the tensor at index_out after a model invocation and feed
it back into index_in before the next invocation.

LiteRT, ONNX Runtime, ExecuTorch, and MNN runners manage declared states automatically. They
initialize state inputs to zero, carry updated state outputs into the next invocation, and restore
zeroed state inputs when `reset_states()` is called.

Conceptually:

```cpp
next_input_state[index_in] = previous_output_state[index_out]
```

The states field is optional and should be omitted for stateless models.

**State tensors should be declared only in the states section, not repeated in the inputs or outputs
sections.**

```
                         ┌──────────────────────┐
                         │        MODEL         │
                         │                      │
inputs ────────────────▶ │                      │ ───────────────▶ outputs
                         │                      │
                         └──────────────────────┘
                              ▲          │
                              │          │
                              └── state ─┘
                         state output becomes
                         state input next run
```

#### Runner params (`runner_params`)

Defines runner-specific parameters required to configure the selected execution backend.

Every model entry must include a `runner_params` object. The object is used for settings that are
not common to all runners but are required or meaningful for a specific runner. The expected
structure depends on the value of `runner`.

For example, a LiteRT model may not require any runner-specific parameters, so it uses an empty
object:

```json
"classifier": {
  "runner": "litert",
  "filename": "model.tflite",
  "inputs": { ... },
  "outputs": { ... },
  "runner_params": {}
}
```

A `llama_cpp` model may require sampler configuration:

```json
"lm_core": {
  "runner": "llama_cpp",
  "filename": "model.gguf",
  "inputs": {
    "token_ids": {
      "index": 0,
      "dtype": "int32",
      "shape": [-1]
    }
  },
  "outputs": {
    "token_ids": {
      "index": 0,
      "dtype": "int32",
      "shape": [-1]
    }
  },
  "runner_params": {
    "embedding_position_mode": "single",
    "sampler": {
      "policy": "distribution",
      "distribution": {
        "top_k": 20,
        "top_k_strategy": "strict_k",
        "top_p": 0.8,
        "min_p": 0.0,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "seed": 12345,
        "rng_strategy": "mt19937"
      },
      "max_tokens_to_gen": 256,
      "append_generated_token": true
    }
  }
}
```

The `runner_params` field is required at the model schema level. Individual runners may require
specific fields inside it.

For `llama_cpp`, `runner_params.sampler.max_tokens_to_gen` optionally limits the number of non-EOG
tokens generated for one response. It must be `-1` or a non-negative integer and defaults to `-1`,
which means generation continues until EOG or another runtime constraint stops it. A value of `0`
evaluates the prompt without generating a token.

For direct embedding input, `llama_cpp` also accepts an optional
`runner_params.embedding_position_mode`. It defaults to `single`, which supplies one position per
embedding. Set it to `mrope_broadcast` to supply four identical position planes for llama.cpp models
that use M-RoPE/IMRoPE with embedding input.

#### Hardware Hints (`hw_hint`)

Provides optional execution preferences for hardware acceleration.

Example:

```json
"hw_hint": ["gpu"]
```

The value is an ordered list of preferred hardware targets. The runtime should attempt to execute
the model using the targets in the listed order. If the first target is unavailable, the runtime may
try the next one. If none are available, the runtime may fall back to CPU execution.

The field is optional. If omitted, CPU execution is assumed.

**`hw_hint` is a preference, not a strict requirement. The runtime may ignore it if the requested
target is unavailable or unsupported.**

## Configs section

- **Required:** no
- **Mutable:** yes
- **Description:** Defines user-configurable parameters that can be changed to adjust the brick
  behavior.

The `configs` section contains parameters that are intended to be modified by the user or
application. These values control runtime brick behavior.

Examples include:

- selecting the voice used by a Text-to-Speech brick
- changing the system prompt used by a language model
- adjusting generation parameters
- changing detection thresholds
- selecting optional postprocessing behavior

Configuration values may be overridden before calling `setup()`. They may also be updated at runtime
through the brick runtime contract, for example by calling `update_config()`.

The purpose of `update_config()` is to allow fast changes that do not require reloading the model.
Therefore, values in configs should be limited to parameters that the brick can apply without
rebuilding or reinitializing the underlying model.

### Structure

```json
"configs": {
  "<name>": {
    "type": "<datatype>",
    "value": <value>
  }
}
```

Example:

```json
"configs": {
  "system_prompt": {
    "type": "string",
    "value": "You are a helpful assistant."
  },
  "voice": {
    "type": "string",
    "value": "default"
  }
}
```

### Template expressions

`configs` values may contain template expressions. Template expressions are resolved by the manifest
loader before the values are passed to the brick implementation.

This is useful when one value is derived from other manifest values.

For example:

```json
"configs": {
  "prompt_template": {
    "type": "string",
    "value": "<｜hy_begin▁of▁sentence｜>{{ system_prompt }}<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Text:\n<<USER_TEXT>><｜hy_Assistant｜>{{ assistant_prefix }}"
  },
  "system_prompt": {
    "type": "string",
    "value": "Translate the text from {{ src_lang }} into {{ dst_lang }}. Output only the translated text."
  },
  "src_lang": {
    "type": "string",
    "value": "English"
  },
  "dst_lang": {
    "type": "string",
    "value": "Spanish"
  },
  "assistant_prefix": {
    "type": "string",
    "value": ""
  }
}
```

After template resolution, `system_prompt` becomes:

```
Translate the text from English into Spanish. Output only the translated text.
```

and `prompt_template` is resolved using the resolved `system_prompt` and `assistant_prefix` values.

Template references use the form:

```
{{ name }}
```

where `name` refers to another entry in the same `configs` section.

The manifest loader should resolve templates when calling get_config().

## Constants section

- **Required:** no
- **Mutable:** no
- **Description:** Defines fixed brick parameters that control runtime behavior but cannot be
  changed by the user.

The `constants` section contains implementation-specific values that are fixed for a given brick
variant. These values may be used by the brick runtime, preprocessing logic, postprocessing logic,
or model execution, but they are not part of the public brick API.

Unlike `configs`, constants cannot be overridden.

Examples include:

- model sample rate
- frame size
- number of classes
- fixed preprocessing parameters

### Structure

```json
"constants": {
  "<name>": {
    "type": "<datatype>",
    "value": <value>
  }
}
```

Example:

```json
"constants": {
  "sample_rate": {
    "type": "int64",
    "value": 32000
  },
  "model_version": {
    "type": "string",
    "value": "v2"
  }
}
```

### Template expressions

`constants` values may contain template expressions. Template expressions are resolved by the
manifest loader before the values are passed to the brick implementation.

This is useful when one value is derived from other manifest values.

For example:

```json
"constants": {
  "prompt_template": {
    "type": "string",
    "value": "<｜hy_begin▁of▁sentence｜>{{ system_prompt }}<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Text:\n<<USER_TEXT>><｜hy_Assistant｜>{{ assistant_prefix }}"
  },
  "system_prompt": {
    "type": "string",
    "value": "Translate the text from {{ src_lang }} into {{ dst_lang }}. Output only the translated text."
  },
  "src_lang": {
    "type": "string",
    "value": "English"
  },
  "dst_lang": {
    "type": "string",
    "value": "Spanish"
  },
  "assistant_prefix": {
    "type": "string",
    "value": ""
  }
}
```

After template resolution, `system_prompt` becomes:

```
Translate the text from English into Spanish. Output only the translated text.
```

and `prompt_template` is resolved using the resolved `system_prompt` and `assistant_prefix` values.

Template references use the form:

```
{{ name }}
```

where `name` refers to another entry in the same `constants` section.

The manifest loader should resolve templates when calling get_constant().

## Assets section

- **Required:** no
- **Mutable:** no
- **Description:** Additional files or folder locations required by the brick implementation.

The `assets` section lists resources used by the brick that are not model files. These may include
label files, tokenizer files, vocabulary files, normalization data, prompt templates, lookup tables,
or other implementation-specific resources.

Asset paths are always resolved relative to the directory containing the manifest.

For example, if the manifest is located at:

```
<your-path>/<brick-name>/manifest.json
```

then the following asset:

```json
"assets": {
  "class_names": "class_names.json"
}
```

resolves to:

```
<your-path>/<brick-name>/class_names.json
```

### Structure

```json
"assets": {
  "<name0>": "<relative_path0>",
  "<name1>": "<relative_path1>"
}
```

Example:

```json
"assets": {
  "class_names": "class_names.json",
  "tokenizer": "tokenizer/tokenizer.json",
  "normalization_stats": "preprocessing/stats.json"
}
```

## Sample inputs

A model package may include optional user-facing input samples under a `samples/` directory beside
the manifest file. Samples are used to exercise brick CLIs and are not required runtime assets, so
they must not be declared in the manifest `assets` section.

For example:

```text
<model-package>/
  <name>_manifest.json
  samples/
    example.wav
    example.jpg
```

Typed brick CLIs resolve `@sample/<path>` relative to this directory. For example,
`@sample/example.wav` resolves to `<model-package>/samples/example.wav` after the CLI resolves its
selected manifest.

The `@sample/` namespace is valid only for read-only input arguments such as `-a`, `-i`, and `-c`.
Output arguments remain ordinary filesystem paths and must not write into an installed model
package.

Distributed sample files must be listed in the model package `SHA256SUMS` file.
