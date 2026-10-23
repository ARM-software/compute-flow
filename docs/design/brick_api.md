<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Brick API

A brick exposes two separate contracts:

1. The **brick runtime interface**, which defines how the runtime drives execution through functions
   such as `setup()` and `run()`.

2. The **brick API**, which defines the public input and output tensors exchanged with the brick.

This document focuses on the brick API, which defines what tensor data the brick consumes and
produces.

## 1. Overview

A brick API defines the public contract exposed by a brick. It defines what tensor data the brick
consumes and produces.

Brick APIs follow the Compute Flow [compatibility principle](../PRODUCT.md#compatibility-principle).
New API versions may introduce breaking changes for new bricks, but bricks built against previous
API versions must continue to work with newer releases of Compute Flow.

The API is independent from the model package manifest. Multiple brick variants can implement the
same API even if they use different models, runners, preprocessing steps, postprocessing logic, or
assets.

For example, different Text-to-Speech bricks may use different models internally, but they can all
expose the same public API:

The brick API is a property of the brick implementation, not of the manifest. The manifest
identifies the brick type and name only:

```json
  "brick": {
    "type": "tts",
    "name": "Qwen3TTS",
    "version": "1.0.0"
  }
```

The API definition, which describes the brick’s input and output tensors, is not stored in the brick
manifest and is not referenced from it. If a brick implementation changes API version, the manifest
should not change for that reason.

Brick APIs are defined in separate JSON files under the `brick_apis/` directory in Compute Flow.
Each file is named after the implementation API contract it defines.

For example, the `tts.v1` API used by Qwen3-TTS is defined in:

```
brick_apis/tts.v1.json
```

while the `object_detection.v1.json` API is defined in:

```
brick_apis/object_detection.v1.json
```

The brick API JSON file consists of the following fields:

```json
{
  "name": "brick_name_api",
  "description": "Brick description.",
  "inputs": {},
  "outputs": {}
}
```

Each entry in inputs and outputs must describe all public tensors exposed by the API. At minimum,
each tensor definition should include:

```json
{
  "index": 0,
  "dtype": "data type",
  "shape": ["#meaning_dim0", "#meaning_dim1", "#meaning_dim2"],
  "description": "Description."
}
```

The fields have the following meaning:

| Field         | Meaning                                                                   |
| ------------- | ------------------------------------------------------------------------- |
| `index`       | Tensor index in the public brick API.                                     |
| `dtype`       | Tensor data type.                                                         |
| `shape`       | Tensor shape using concrete, symbolic, or derived dimensions.             |
| `description` | Human-readable description of the tensor.                                 |
| `json_schema` | Optional JSON Schema for the decoded value of a UTF-8 JSON string tensor. |

`json_schema` documents the expected value; Compute Flow does not validate it at runtime. For a
string tensor, `shape` counts bytes, so `[-1]` represents variable-length JSON text.

For example, an audio generation API may define the left audio channel output as:

```json
"left_channel": {
  "index": 0,
  "dtype": "float32",
  "shape": ["#num_left_samples"],
  "description": "Number of samples for the left channel."
}
```

## 2. Guidelines for Designing a Brick API

A brick API should represent the common contract shared by all variants of a brick type, rather than
the capabilities of a specific implementation.

A useful rule of thumb is:

> The brick API should represent the minimum common denominator across all brick variants of the
> same type.

### 2.1 Avoid Variant-Specific APIs

APIs should not expose features that are only available in a subset of variants.

For example, Text-to-Speech bricks universally accept text as input and produce audio as output.

Voice cloning, however, is not supported by every TTS model. Therefore, voice cloning should not be
part of the mandatory API. Instead, it should be exposed through brick-specific configuration
mechanisms.

Good API-level contract:

```json
{
  "inputs": {
    "text": {
      "index": 0,
      "dtype": "string",
      "shape": [-1],
      "description": "Text to synthesize."
    }
  },
  "outputs": {
    "left_channel": {
      "index": 0,
      "dtype": "float32",
      "shape": ["#num_samples"],
      "description": "Generated left audio channel."
    },
    "right_channel": {
      "index": 1,
      "dtype": "float32",
      "shape": ["#num_samples"],
      "description": "Generated right audio channel."
    },
    "sample_rate": {
      "index": 2,
      "dtype": "int64",
      "shape": [1],
      "description": "Sample rate of the generated audio in Hz."
    }
  }
}
```

The API remains stable, while specific implementations can expose additional capabilities through
the manifest's config field.

### 2.2 Expose All Required Outputs

Applications should not need to inspect manifest's constants to correctly consume outputs.

For example, a Text-to-Speech brick should return:

- Left audio channel
- Right audio channel
- Sample rate

Returning only the audio channels would make it impossible for the audio playback system to
determine how the samples should be interpreted. For example, without the sample rate, the player
cannot know the correct playback speed or duration of the generated audio.

The API should therefore expose all information required to process the outputs correctly.

Good:

```json
"outputs": {
  "left_channel": {
    "index": 0,
    "dtype": "float32",
    "shape": ["#num_samples"],
    "description": "Generated left audio channel."
  },
  "right_channel": {
    "index": 1,
    "dtype": "float32",
    "shape": ["#num_samples"],
    "description": "Generated right audio channel."
  },
  "sample_rate": {
    "index": 2,
    "dtype": "int64",
    "shape": [1],
    "description": "Sample rate of the generated audio in Hz."
  }
}
```

Avoid:

```json
"outputs": {
  "left_channel": {
    "index": 0,
    "dtype": "float32",
    "shape": ["#num_samples"]
  },
  "right_channel": {
    "index": 1,
    "dtype": "float32",
    "shape": ["#num_samples"]
  }
}
```

### 2.3 Design Outputs for Streaming and Non-Streaming

Output tensors must let the same brick type work in both streaming and non-streaming modes.

For example, an audio detection brick may produce one frame for the entire input, or multiple frames
with finer temporal granularity. In non-streaming mode, a single `run()` call can return every
frame. In streaming mode, each `run()` call may return only the frames currently available, and that
number can be fewer than the total number of frames for the full input.

The output API must therefore include enough information for the application to locate each event in
the input audio.

For example, an audio detection API may include an explicit temporal frame for this reason:

```json
"outputs": {
  "id": {
    "index": 0,
    "dtype": "int64",
    "shape": ["#num_detected_events", "#num_classes_reported"],
    "description": "Class IDs for each detected event, ordered by descending probability."
  },
  "score": {
    "index": 1,
    "dtype": "float32",
    "shape": ["#num_detected_events", "#num_classes_reported"],
    "description": "Confidence scores for each class per detected event, ordered by descending probability."
  },
  "name": {
    "index": 2,
    "dtype": "uint8",
    "shape": ["#num_detected_events", "#num_classes_reported", "#max_name_length"],
    "description": "Class names encoded as UTF-8 bytes per detected event, ordered by descending probability."
  },
  "frame": {
    "index": 3,
    "dtype": "float32",
    "shape": ["#num_detected_events", 2],
    "description": "Temporal bounding frame in seconds for each detected event (t, t_end)."
  }
}
```

Without `frame`, a streaming consumer that receives fewer events than the full input cannot place
those events on the input timeline.

Returning explicit bounds with each event makes the timestamp independent of the implementation's
window size and streaming behavior:

```text
frame[event_index] = [start_time_seconds, end_time_seconds]
```

A binary audio classifier does not need to return class IDs or names for every frame. Its positive
condition is defined by the brick type, so it can use the smaller `audio_binary_classification.v1`
output contract:

```json
"outputs": {
  "score": {
    "index": 0,
    "dtype": "float32",
    "shape": ["#num_frames"],
    "description": "Confidence score in [0, 1] that the positive condition is present in each frame."
  },
  "frame": {
    "index": 1,
    "dtype": "float32",
    "shape": ["#num_frames", 2],
    "description": "Temporal bounding frame in seconds for each classification frame (t, t_end)."
  }
}
```

### 2.4 Keep the API Independent from the Model

The public API should not expose model-specific tensors unless they are part of the common contract
for the brick type.

For example, a Qwen3-TTS model may internally expose tensors such as:

```json
{
  "inputs_embeds": {
    "index": 0,
    "dtype": "float32",
    "shape": [1, "#num_tokens", "#embedding_size"]
  },
  "hidden_state": {
    "index": 0,
    "dtype": "float32",
    "shape": [1, 1, "#embedding_size"]
  },
  "token_id": {
    "index": 1,
    "dtype": "int32",
    "shape": [1]
  }
}
```

These are implementation details. They should be described in the brick manifest under `models`, not
in the public API, unless every variant of the brick type is expected to expose them.

The API should describe what applications consume, not how a specific model computes it.

### 2.5 Use clear, stable Names

API tensor names should describe semantic meaning, not implementation details.

Prefer:

```text
raw_wav
sample_rate
id
score
name
frame
```

Avoid names tied to a specific model or implementation:

```text
qwen3_tts_logits
model_output_0
tflite_tensor_1
```

The API should remain valid if the implementation changes.

### 2.6 Version APIs Explicitly

API names should include a version suffix.

Example:

```text
audio_binary_classification.v1
audio_detection.v1
audio_generation.v1
audio_to_audio.v1
image_to_image.v1
object_detection.v1
```

A breaking change should create a new API version.

Breaking changes include:

- Removing an input or output
- Renaming an input or output
- Changing tensor indices
- Changing dtype
- Changing shape semantics
- Changing the meaning or ordering of output values

Non-breaking changes may include:

- Adding descriptions
- Clarifying documentation
