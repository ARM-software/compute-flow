<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Running a minimal Compute Flow application with Python

## Dependencies

- A Mac® with Apple silicon, or a Linux® system with an Arm®-based CPU, such as a Raspberry Pi®.
- `uv`, `git`, and a native C and C++ compiler.
- An audio output device.

## Goal

This guide installs the Qwen3-TTS 0.6B Custom Voice brick and model, builds the native Compute Flow
Python module, and runs one text-to-speech prompt. Like the
[minimal C++ application](../cpp/minimal_example.md), it reports the number of generated audio
samples and their sample rate. It also plays the generated audio.

Run all commands from the Compute Flow project root.

## Install the brick and model

### Step 1: Set up Compute Flow

Create and activate the project environment:

```bash
uv sync && source .venv/bin/activate
```

Check the required system dependencies:

```bash
compute-flow check
```

If the check reports a missing dependency, follow the suggested action and run the check again.

### Step 2: Download the brick and model

Update the local brick registry, then install the Qwen3-TTS 0.6B Custom Voice model:

```bash
compute-flow update
compute-flow model install tts:qwen3-tts qwen3-tts-0-6b-custom-voice
```

The model command installs both the brick source and model package. Inspect the installation and
save the model package path:

```bash
compute-flow model list tts:qwen3-tts
compute-flow model info tts:qwen3-tts qwen3-tts-0-6b-custom-voice
PACKAGE_DIR="$(compute-flow model path tts:qwen3-tts qwen3-tts-0-6b-custom-voice)"
```

## Build the native Python module

### Step 3: Build Compute Flow

Build Compute Flow and the installed brick for the current machine:

```bash
compute-flow build native
```

The build installs the `compute_flow_py` module and the brick's Python binding into the activated
project environment.

## Create the application

### Step 4: Add the source file

Create an application directory:

```bash
mkdir minimal_example_python
```

Save the following as `minimal_example_python/main.py`:

```python
"""Run one text-to-speech request with Compute Flow."""

import argparse

import compute_flow_py as cf
import numpy as np
import sounddevice as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to the model manifest")
    args = parser.parse_args()

    # Load the model configuration and installed brick implementation.
    manifest = cf.load_manifest(args.manifest)
    brick = cf.load_brick(manifest)

    # Prepare the brick and warm it up before the first request.
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    brick.setup(manifest, options)

    input_tensors = {"text_prompt": "Hello, I am Aiden, your personal AI assistant."}
    audio_chunks = []
    num_samples = 0
    sample_rate = 0

    # Consume each audio chunk until generation is complete.
    while not brick.is_generation_complete():
        output = brick.run(input_tensors)
        left = output["left_channel"]
        right = output["right_channel"]
        num_samples += left.size
        sample_rate = int(output["sample_rate"].item())
        if left.size:
            audio_chunks.append(np.column_stack((left, right)))

    print(f"Generated {num_samples} samples at {sample_rate} Hz.")

    if audio_chunks:
        sd.play(np.concatenate(audio_chunks), sample_rate)
        sd.wait()

    # Clear request state while keeping the loaded models ready for reuse.
    brick.reset_states()


if __name__ == "__main__":
    main()
```

The input dictionary stays alive for the full generation loop. The application counts samples from
the left audio channel and copies each stereo chunk before the brick can reuse its output buffers.

## Run the application

### Step 5: Generate and play audio

Run the application with the English Qwen3-TTS 0.6B Custom Voice manifest:

```bash
python minimal_example_python/main.py "$PACKAGE_DIR/qwen3_tts_custom_voice_en_manifest.json"
```

The application prints the generated sample count per channel and the sample rate, then plays the
audio through the default output device:

```text
Generated <number of samples> samples at <sample rate> Hz.
```
