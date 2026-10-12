<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Transcribing audio with speaker labels and timestamps with Python

## Dependencies

- A Mac® with Apple silicon, or a Linux® system with an Arm®-based CPU, such as a Raspberry Pi® 5.
- `uv`, `git`, and a native C and C++ compiler.

## Goal

Use Nemotron 3 Diarization to identify speaker turns in a recording. Transcribe each turn with
Parakeet TDT, then write a transcript with speaker labels and time ranges.

Run all commands from the Compute Flow project root.

## Install the bricks and models

### Step 1: Set up Compute Flow

Create and activate the project environment:

```bash
uv sync
source .venv/bin/activate
```

Check the required system dependencies:

```bash
compute-flow check
```

If the check reports a missing dependency, follow the suggested action and run the check again.

### Step 2: Download the models

Update the local brick registry, then install both model packages:

```bash
compute-flow update
compute-flow model install audio_detection:nemotron3-diarization nemotron-3-diarization-litert-fp32
compute-flow model install stt:parakeet-tdt parakeet-tdt-0-6b-v3-int8-dynamic
```

## Build the native Python module

### Step 3: Build Compute Flow

Build Compute Flow with both installed bricks:

```bash
compute-flow build native
```

## Create the application

### Step 4: Connect the bricks

Create a directory for the script and transcript:

```bash
mkdir -p speaker_transcription_example
```

Save the following as `speaker_transcription_example/transcribe_audio.py`:

```python
"""Transcribe audio with speaker labels and timestamps using Compute Flow."""

from __future__ import annotations

import argparse
from pathlib import Path

import compute_flow_py as cf
import numpy as np
import soundfile as sf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diarization-manifest", required=True)
    parser.add_argument("--stt-manifest", required=True)
    parser.add_argument("--input", required=True, dest="input_audio")
    parser.add_argument("--output", default="transcript.txt")
    args = parser.parse_args(argv)

    # Load the recording once as contiguous mono float32 audio.
    audio, input_sample_rate = sf.read(
        args.input_audio, dtype="float32", always_2d=True
    )
    mono_audio = np.ascontiguousarray(
        np.mean(audio, axis=1, dtype=np.float32), dtype=np.float32
    )
    sample_rate = np.asarray([input_sample_rate], dtype=np.int64)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True

    # Load the manifests.
    diarization_manifest = cf.load_manifest(args.diarization_manifest)
    stt_manifest = cf.load_manifest(args.stt_manifest)

    # Load the bricks.
    diarization_brick = cf.load_brick(diarization_manifest)
    stt_brick = cf.load_brick(stt_manifest)

    # Set up the bricks.
    diarization_brick.setup(diarization_manifest, options)
    stt_brick.setup(stt_manifest, options)

    # Identify the speaker turns.
    diarization_brick.reset_states()
    diarization_input = cf.TensorDict()
    diarization_input["raw_wav"] = mono_audio
    diarization_input["sample_rate"] = sample_rate
    speaker_turns = []

    while not diarization_brick.is_generation_complete():
        output = diarization_brick.run(diarization_input)
        names = np.asarray(output["name"], dtype=np.uint8)
        frames = np.asarray(output["frame"], dtype=np.float32).reshape(-1, 2)

        for index, (start, end) in enumerate(frames):
            speaker = (
                bytes(names[index, 0].tolist()).split(b"\x00", 1)[0].decode("utf-8")
            )
            speaker_turns.append((speaker, float(start), float(end)))

    # Transcribe each detected speaker interval.
    transcript_sections = []
    for speaker, start, end in speaker_turns:
        start_sample = max(0, round(start * input_sample_rate))
        end_sample = min(mono_audio.size, round(end * input_sample_rate))
        if end_sample <= start_sample:
            continue

        stt_brick.reset_states()
        stt_input = cf.TensorDict()
        stt_input["raw_wav"] = mono_audio[start_sample:end_sample].reshape(1, -1)
        stt_input["sample_rate"] = sample_rate
        transcription_chunks = []

        while not stt_brick.is_generation_complete():
            output = stt_brick.run(stt_input)
            transcription_chunks.append(str(np.asarray(output["text"]).item()))

        transcription = "".join(transcription_chunks).strip()
        if transcription:
            transcript_sections.append(
                f"{speaker.replace('speaker_', 'speaker ', 1)} "
                f"[{start:.2f}, {end:.2f}]s\n{transcription}"
            )

    transcript = "\n\n".join(transcript_sections)

    Path(args.output).write_text(f"{transcript}\n", encoding="utf-8")
    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The application reshapes each detected speaker interval for the STT brick. It keeps each input alive
until generation completes and omits intervals with no recognized text. The time range beside each
speaker is the diarization interval in seconds, rounded to two decimal places.

## Run the pipeline

### Step 5: Produce the transcript

Resolve the installed manifests, then run both bricks on the sample included with the Nemotron
model:

```bash
NEMOTRON_PACKAGE="$(compute-flow model path audio_detection:nemotron3-diarization nemotron-3-diarization-litert-fp32)"
PARAKEET_PACKAGE="$(compute-flow model path stt:parakeet-tdt parakeet-tdt-0-6b-v3-int8-dynamic)"

python speaker_transcription_example/transcribe_audio.py \
  --diarization-manifest "$NEMOTRON_PACKAGE/nemotron3_diarization_manifest.json" \
  --stt-manifest "$PARAKEET_PACKAGE/parakeet_tdt_0_6b_v3_manifest.json" \
  --input "$NEMOTRON_PACKAGE/samples/diarization_example.wav" \
  --output speaker_transcription_example/transcript.txt
```

The application displays the transcript on screen and saves the same content to the output file. A
shortened output looks like this:

```text
speaker 0 [0.35, 9.31]s
To celebrate the release of Nemotron 3 diarization, we invited seven more open text to speech voices to the party. I'm Chris, the NemoTron.

speaker 1 [9.15, 13.81]s
Hi everybody. Thanks for having me.
```

The application transcribes each speaker interval independently. An interval containing overlapping
speech can include words from more than one voice, so exact attribution of every word is not
guaranteed.
