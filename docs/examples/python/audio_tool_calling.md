<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Calling desktop tools from speech with Python

## Dependencies

- A Mac® with Apple silicon.
- `uv`, `git`, and a native C and C++ compiler.
- A microphone, Python Tk support, and a text editor associated with `.txt` files.

## Goal

Use Silero VAD to detect a spoken request, then pass the recorded speech to NeuDecide. The model
selects a tool that opens a countdown or adds a task in a text editor. The Python application runs
on macOS®.

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

Update the local brick registry, then install the VAD and audio tool calling model packages:

```bash
compute-flow update
compute-flow model install vad:silero-vad silero-fp32
compute-flow model install audio_tool_calling:neu-decide neudecide-en
```

The model commands also install the two bricks.

## Build the native Python module

### Step 3: Build Compute Flow

Build Compute Flow with both installed bricks:

```bash
compute-flow build native
```

## Create the application

### Step 4: Connect the bricks

Create an application directory:

```bash
mkdir -p audio_tool_calling_example
```

Save the following as `audio_tool_calling_example/audio_tool_calling.py`:

```python
"""Listen for one request and run a desktop tool selected by NeuDecide."""

import argparse
import json
import subprocess
import sys
import tkinter as tk
from collections import deque
from pathlib import Path

import compute_flow_py as cf
import numpy as np
import sounddevice as sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vad_manifest")
    parser.add_argument("tool_manifest")
    args = parser.parse_args()

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True

    # Prepare both bricks once.
    vad_manifest = cf.load_manifest(args.vad_manifest)
    tool_manifest = cf.load_manifest(args.tool_manifest)

    vad = cf.load_brick(vad_manifest)
    tool_model = cf.load_brick(tool_manifest)

    vad.setup(vad_manifest, options)
    tool_model.setup(tool_manifest, options)
    vad.reset_states()

    # Keep a short lead-in so the first syllable is included.
    sample_rate = np.asarray([16000], dtype=np.int64)
    recent_audio = deque(maxlen=10)
    speech_audio = []
    quiet_frames = 0

    print("Listening...", flush=True)
    with sd.InputStream(
        samplerate=16000, channels=1, blocksize=512, dtype="float32"
    ) as mic:
        while True:
            block, _ = mic.read(512)
            frame = block[:, 0].copy()
            score = np.asarray(
                vad.run({"raw_wav": frame, "sample_rate": sample_rate})["score"]
            ).item()

            if not speech_audio:
                recent_audio.append(frame)
                if score > 0.5:
                    speech_audio = list(recent_audio)
                    print("Speech detected", flush=True)
            else:
                speech_audio.append(frame)
                quiet_frames = quiet_frames + 1 if score < 0.5 else 0
                if quiet_frames == 25:
                    break

    # Send the complete utterance to the audio tool calling brick.
    tool_model.reset_states()
    inputs = {"raw_wav": np.concatenate(speech_audio), "sample_rate": sample_rate}
    chunks = []
    while not tool_model.is_generation_complete():
        output = tool_model.run(inputs)
        chunk = np.asarray(output["calls"]).item()
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

    calls = json.loads("".join(chunks))
    print(f"Tool calls: {calls}")

    for call in calls:
        name = call["name"]
        arguments = call["arguments"]
        print(f"{name}: {arguments}")

        if name == "setTimer":
            window = tk.Tk()
            window.title("Timer")
            label = tk.Label(window, font=("Arial", 48))
            label.pack(padx=40, pady=30)

            def update_timer(remaining: int) -> None:
                label["text"] = f"{remaining // 60:02d}:{remaining % 60:02d}"
                if remaining:
                    window.after(1000, update_timer, remaining - 1)
                else:
                    window.bell()

            update_timer(max(0, round(arguments["minutes"] * 60)))
            window.mainloop()

        elif name == "addTask":
            task_file = Path(__file__).with_name("tasks.txt").resolve()
            with task_file.open("a", encoding="utf-8") as tasks:
                tasks.write(arguments["task"] + "\n")

            if sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "TextEdit", str(task_file)])
            else:
                subprocess.Popen(["xdg-open", str(task_file)])


if __name__ == "__main__":
    main()
```

Silero VAD scores each 32 ms microphone frame. After speech begins, 25 quiet frames end the request.
The recorded waveform stays alive while NeuDecide generates its JSON tool calls. The code handles
only the two tools demonstrated here; it prints other model calls without running them.

## Run the pipeline

### Step 5: Speak a command

Resolve the installed manifests and start listening:

```bash
VAD_PACKAGE="$(compute-flow model path vad:silero-vad silero-fp32)"
TOOL_PACKAGE="$(compute-flow model path audio_tool_calling:neu-decide neudecide-en)"

python audio_tool_calling_example/audio_tool_calling.py \
  "$VAD_PACKAGE/silero_manifest.json" \
  "$TOOL_PACKAGE/neudecide_manifest.json"
```

Try either English request, then run the command again to try the other:

- “Set a timer for one minute.” A countdown window opens and rings at zero; close it when done.
- “Add buy milk to my task list.” The task is saved in `tasks.txt` and opened in a text editor.

For example, the second request should print a call similar to:

```text
Listening...
Speech detected
Tool calls: [{'name': 'addTask', 'arguments': {'task': 'buy milk'}}]
addTask: {'task': 'buy milk'}
```

Model output can vary with the spoken request. Use Ctrl+C to stop listening before a request is
detected.
