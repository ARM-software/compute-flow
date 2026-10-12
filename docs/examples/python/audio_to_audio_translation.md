<!--
    SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Audio-to-audio translation with Python

## Dependencies

- A Mac® with Apple silicon, or a Linux® system with an Arm®-based CPU, such as a Raspberry Pi® 5.
- `uv`, `git`, and a native C and C++ compiler.

## Goal

This guide demonstrates how to combine bricks with a practical example. The example connects the
following bricks to build an audio-to-audio translation pipeline:

1. Parakeet TDT transcribes audio in a supported European language.
2. Hy-MT2 translates the transcription into another language.
3. Qwen3-TTS synthesizes the translation as a WAV file.

The application accepts the source and destination languages as command-line arguments.

Run all commands from the Compute Flow project root.

## Install the bricks and models

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

### Step 2: Download the models

Update the local brick registry, then install the three model packages:

```bash
compute-flow update
compute-flow model install stt:parakeet-tdt parakeet-tdt-0-6b-v3-int8-dynamic
compute-flow model install lm_text:lm-text-generic hy-mt2-1-8b-2bit-translation
compute-flow model install tts:qwen3-tts qwen3-tts-0-6b-custom-voice
```

## Build the native Python module

### Step 3: Build Compute Flow

Build Compute Flow with all three installed bricks:

```bash
compute-flow build native
```

## Create the application

### Step 4: Connect the bricks

Create an application directory:

```bash
mkdir audio_translation_example
```

Save the following as `audio_translation_example/audio_translation.py`:

```python
"""Translate speech between languages with three Compute Flow bricks."""

from __future__ import annotations

import argparse
from pathlib import Path

import compute_flow_py as cf
import numpy as np
import soundfile as sf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stt-manifest", required=True)
    parser.add_argument("--translation-manifest", required=True)
    parser.add_argument("--tts-manifest", required=True)
    parser.add_argument("--input", required=True, dest="input_audio")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--destination-language", required=True)
    parser.add_argument(
        "--output", default="translated_french.wav", dest="output_audio"
    )
    args = parser.parse_args(argv)

    # Load the input as a batch containing one mono float32 waveform.
    input_audio, input_sample_rate = sf.read(
        args.input_audio, dtype="float32", always_2d=True
    )
    mono_audio = np.mean(input_audio, axis=1, dtype=np.float32)
    raw_wav = np.ascontiguousarray(mono_audio.reshape(1, -1), dtype=np.float32)
    sample_rate = np.asarray([input_sample_rate], dtype=np.int64)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True

    # Load the manifests.
    stt_manifest = cf.load_manifest(args.stt_manifest)
    translation_manifest = cf.load_manifest(args.translation_manifest)
    tts_manifest = cf.load_manifest(args.tts_manifest)

    # Configure the requested translation and text-to-speech languages.
    translation_manifest.set_config("src_lang", args.source_language)
    translation_manifest.set_config("dst_lang", args.destination_language)
    tts_manifest.set_config("language", args.destination_language.lower())

    # Load the bricks.
    stt_brick = cf.load_brick(stt_manifest)
    translation_brick = cf.load_brick(translation_manifest)
    tts_brick = cf.load_brick(tts_manifest)

    # Set up the bricks.
    stt_brick.setup(stt_manifest, options)
    translation_brick.setup(translation_manifest, options)
    tts_brick.setup(tts_manifest, options)

    #################################
    # Run speech-to-text brick
    #################################
    stt_brick.reset_states()

    stt_input = {"raw_wav": raw_wav, "sample_rate": sample_rate}
    transcription_chunks = []
    while not stt_brick.is_generation_complete():
        stt_output = stt_brick.run(stt_input)
        transcription_chunks.append(str(np.asarray(stt_output["text"]).item()))

    #################################
    # Run translation brick
    #################################
    translation_brick.reset_states()

    translation_input = {"text_prompt": "".join(transcription_chunks)}
    translation_chunks = []
    while not translation_brick.is_generation_complete():
        translation_output = translation_brick.run(translation_input)
        translation_text = np.asarray(translation_output["text"]).item()
        translation_chunks.append(str(translation_text))

    tts_input = {"text_prompt": "".join(translation_chunks)}
    left_chunks = []
    right_chunks = []
    output_sample_rate = 0

    #################################
    # Run text-to-speech brick
    #################################
    tts_brick.reset_states()

    while not tts_brick.is_generation_complete():
        tts_output = tts_brick.run(tts_input)
        left = np.asarray(tts_output["left_channel"], dtype=np.float32).reshape(-1)
        right = np.asarray(tts_output["right_channel"], dtype=np.float32).reshape(-1)
        sample_rate_tensor = np.asarray(tts_output["sample_rate"])
        output_sample_rate = int(sample_rate_tensor.reshape(-1)[0])

        if left.size:
            left_chunks.append(left.copy())
            right_chunks.append(right.copy())

    tts_brick.reset_states()

    if not left_chunks or output_sample_rate == 0:
        raise RuntimeError("Qwen3-TTS did not generate audio")

    left_audio = np.concatenate(left_chunks)
    right_audio = np.concatenate(right_chunks)
    output_audio = np.column_stack((left_audio, right_audio))

    sf.write(
        args.output_audio,
        output_audio,
        output_sample_rate,
        subtype="FLOAT",
    )

    print(f"Transcription: {''.join(transcription_chunks)}")
    print(f"Translation: {''.join(translation_chunks)}")
    print(f"Saved: {Path(args.output_audio).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Each generation loop keeps its input alive until the brick completes. Text chunks are joined before
they become the next brick's input, while audio chunks are copied because the TTS brick can reuse
its output buffers on the next call.

## Run the pipeline

### Step 5: Translate the sample audio

Resolve the installed package paths, then run the application with the English Parakeet sample. The
application configures the English Qwen3-TTS manifest for French before setup:

```bash
PARAKEET_PACKAGE="$(compute-flow model path stt:parakeet-tdt parakeet-tdt-0-6b-v3-int8-dynamic)"
HY_MT2_PACKAGE="$(compute-flow model path lm_text:lm-text-generic hy-mt2-1-8b-2bit-translation)"
QWEN3_TTS_PACKAGE="$(compute-flow model path tts:qwen3-tts qwen3-tts-0-6b-custom-voice)"

python audio_translation_example/audio_translation.py \
  --stt-manifest "$PARAKEET_PACKAGE/parakeet_tdt_0_6b_v3_manifest.json" \
  --translation-manifest "$HY_MT2_PACKAGE/hy_mt2_1_8b_manifest.json" \
  --tts-manifest "$QWEN3_TTS_PACKAGE/qwen3_tts_custom_voice_en_manifest.json" \
  --input "$PARAKEET_PACKAGE/samples/beckett.wav" \
  --source-language English \
  --destination-language French \
  --output translated_french.wav
```

The expected text output is:

```text
Transcription: Ever tried, ever failed. No matter. Try again. Fail again. Fail better.
Translation: Avez-vous déjà essayé, avez-vous déjà échoué. Peu importe. Essayez à nouveau. Échouez encore. Échouez mieux.
```

## Use a different brick variant

The application code does not need to change when you use a different variant of the same brick
type. Download the new variant and its model, rebuild Compute Flow, and pass the new manifest to the
application.

For example, replace Parakeet TDT with the English Moonshine STT variant:

```bash
compute-flow model install stt:moonshine moonshine-base-en
compute-flow build native

MOONSHINE_PACKAGE="$(compute-flow model path stt:moonshine moonshine-base-en)"
HY_MT2_PACKAGE="$(compute-flow model path lm_text:lm-text-generic hy-mt2-1-8b-2bit-translation)"
QWEN3_TTS_PACKAGE="$(compute-flow model path tts:qwen3-tts qwen3-tts-0-6b-custom-voice)"

python audio_translation_example/audio_translation.py \
  --stt-manifest "$MOONSHINE_PACKAGE/moonshine_manifest_base_en.json" \
  --translation-manifest "$HY_MT2_PACKAGE/hy_mt2_1_8b_manifest.json" \
  --tts-manifest "$QWEN3_TTS_PACKAGE/qwen3_tts_custom_voice_en_manifest.json" \
  --input "$MOONSHINE_PACKAGE/samples/beckett.wav" \
  --source-language English \
  --destination-language French \
  --output translated_french.wav
```

## Changing the LM params

Translation wording can vary slightly. The application writes the generated French speech to
`translated_french.wav`.

For example, you can adjust LM parameters such as `top_p` and `temperature` to see how they affect
the translation. Get the path to the installed Hy-MT2 manifest:

```bash
HY_MT2_PACKAGE="$(compute-flow model path lm_text:lm-text-generic hy-mt2-1-8b-2bit-translation)"
echo "$HY_MT2_PACKAGE/hy_mt2_1_8b_manifest.json"
```

In that manifest, set the fields in `models.lm_core.runner_params.sampler.distribution` before
running the application:

```json
"top_p": 0.7,
"temperature": 0.9
```

To use other languages, change `--source-language` and `--destination-language`. The application
applies the destination language to Qwen3-TTS before setup.
