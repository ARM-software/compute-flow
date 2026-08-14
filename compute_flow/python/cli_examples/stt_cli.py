#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python CLI for the STT (Speech-To-Text) brick — mirrors the C++ tool one-to-one.

Usage:
  python -m cli_examples.stt_cli -m <manifest_path> -a <mono_audio_path> [-t <thread_spec>] [-o <output_file>]
"""

import argparse
import numpy as np
import os
import sys
import soundfile as sf
import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def load_as_wav(path: str) -> np.ndarray:
    """
    Load float32 mono WAV using soundfile and return a NumPy 2D array (1, N).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load as float32, always_2d gives shape (N, C)
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)

    # Ensure mono
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)

    # Convert to 2-D matrix of shape (1, N)
    wav = np.expand_dims(mono.astype(np.float32, copy=False), axis=0)

    return wav, np.asarray(file_sr)


def text_output_to_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    array = np.asarray(value)
    if array.size == 0:
        return ""
    if array.dtype.kind == "S":
        return b"".join(array.reshape(-1).tolist()).decode("utf-8", errors="replace")
    if array.dtype.kind == "U":
        return "".join(array.reshape(-1).tolist())
    if array.dtype == np.uint8:
        return bytes(array.reshape(-1).tolist()).decode("utf-8", errors="replace")
    if array.shape == ():
        return str(array.item())
    return "".join(str(item) for item in array.reshape(-1).tolist())


def record_audio(duration: float, sr: int = 16000) -> np.ndarray:
    """
    Record audio from the default microphone for `duration` seconds.
    Returns a NumPy 2D array (1, N) of float32 samples at `sr`.
    """
    if duration <= 0:
        raise ValueError(f"Recording duration must be > 0 seconds (got {duration}).")

    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for recording audio. "
            "Install it with `pip install sounddevice`."
        ) from e

    num_frames = int(duration * sr)

    print(f"Recording {duration:.2f} seconds of audio at {sr} Hz...")
    recording = sd.rec(
        num_frames,
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocking=False,
    )
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # recording shape: (N, 1)
    mono = recording[:, 0]

    # Convert to shape (1, N)
    wav = np.expand_dims(mono.astype(np.float32, copy=False), axis=0)
    return wav


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="STT brick CLI (Python version of the C++ tool)",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser.add_argument(
        "-a", dest="audio_path", help="Input audio file or @sample/... selector"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record audio from default microphone instead of reading from a file",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (used with --record, default: 5.0)",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off)",
    )

    args = parser.parse_args(argv)

    if not args.manifest_path:
        parser.print_help()
        sys.exit("ERROR: Missing required arguments.")

    if not args.audio_path and not args.record:
        parser.print_help()
        sys.exit("ERROR: Either an audio file or recording must be set")

    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(args.thread_spec)

    manifest_path = get_manifest_path(args.manifest_path, "stt")
    audio_path = args.audio_path
    if audio_path and not args.record:
        audio_path = get_sample_path(audio_path, manifest_path)

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)

    # Configure default and per-model CPU thread counts
    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = args.enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    input = cf.TensorDict()

    input["sample_rate"] = np.asarray(16000)
    if args.record:
        input["raw_wav"] = record_audio(args.duration, sr=16000)
    else:
        input["raw_wav"], input["sample_rate"] = load_as_wav(audio_path)

    # Run the model inference
    sys.stdout.write("[TRANSCRIPTION]\n")
    while not brick.is_generation_complete():
        output = brick.run(input)
        text = text_output_to_str(output["text"])
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
    sys.stdout.write("\n")

    # Display profiler run
    print(brick.write_profile(), end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
