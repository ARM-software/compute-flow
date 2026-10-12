#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an STT brick to transcribe audio or microphone input."""

from __future__ import annotations

import argparse
import os
import sys

import compute_flow_py as cf
import numpy as np
import soundfile as sf

from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def load_as_wav(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load float32 mono WAV samples and their sample rate."""
    if not os.path.exists(path):
        raise ValueError(f"Audio file not found: {path}")

    # Load as float32, always_2d gives shape (N, C)
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)

    # Ensure mono
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)

    # Convert to 2-D matrix of shape (1, N)
    wav = np.expand_dims(mono.astype(np.float32, copy=False), axis=0)

    return wav, np.asarray([file_sr], dtype=np.int64)


def text_output_to_str(value: object) -> str:
    """Convert an STT output tensor value to text."""
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
    """Record float32 mono audio from the default microphone."""
    if duration <= 0:
        raise ValueError(f"Recording duration must be > 0 seconds (got {duration}).")

    try:
        import sounddevice as sd
    except ImportError as error:
        raise ImportError(
            "sounddevice is required for recording audio. "
            "Install it with `pip install sounddevice`."
        ) from error

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


def transcribe(brick, input_data: cf.TensorDict) -> None:
    """Run STT inference and write each generated text chunk."""
    sys.stdout.write("----------------\n")
    sys.stdout.write("[TRANSCRIPTION]\n")

    while not brick.is_generation_complete():
        output = brick.run(input_data)
        text = text_output_to_str(output["text"])

        if text:
            sys.stdout.write(text)
            sys.stdout.flush()

    sys.stdout.write("\n----------------\n")


def run_single(
    manifest_path: str,
    audio_path: str | None,
    thread_spec: str,
    enable_op_profiling: bool,
    record: bool,
    duration: float,
) -> None:
    """Run one STT request from a file or microphone recording."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "stt")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)

    # Configure default and per-model CPU thread counts
    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    brick.reset_states()

    input_data = cf.TensorDict()

    if record:
        input_data["sample_rate"] = np.asarray([16000], dtype=np.int64)
        input_data["raw_wav"] = record_audio(duration, sr=16000)
    else:
        resolved_audio_path = get_sample_path(audio_path, manifest_path)
        input_data["raw_wav"], input_data["sample_rate"] = load_as_wav(
            resolved_audio_path
        )

    # Run the model inference
    transcribe(brick, input_data)

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run STT requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "stt")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)

    # Configure default and per-model CPU thread counts
    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    run_configs = load_run_configs(runs_file, {"a"}, {"a"})

    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        audio_path = get_run_value(config, "a", "", run_index)
        if not audio_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'a' must be a non-empty string"
            )

        # Read the audio file and set the input
        input_data = cf.TensorDict()
        input_data["raw_wav"], input_data["sample_rate"] = load_as_wav(audio_path)

        # Run the model inference
        transcribe(brick, input_data)

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv: list[str] | None = None) -> int:
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description="STT brick CLI (Python version of the C++ tool)",
        usage=(
            "%(prog)s -m <manifest_path> "
            "(-a <audio_file> | -j <runs.json> | --record [--duration <seconds>]) "
            "[-t <thread_spec>]"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        metavar="<manifest_path>",
        help="Manifest path or @manifest/... selector.",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        metavar="<thread_spec>",
        help=(
            "Number of threads to use. Specify a default for all models and optionally "
            "override individual models, e.g. "
            "<default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)."
        ),
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    parser.add_argument(
        "-a",
        dest="audio_path",
        metavar="<audio_file>",
        help="Input WAV path or @sample/... selector.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON file path for multiple run settings.",
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
        metavar="<seconds>",
        help="Recording duration in seconds (used with --record, default: 10.0).",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off)",
    )

    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = args.runs_file is not None
    if multiple_runs_mode and (args.audio_path is not None or args.record):
        parser.error("use either direct run arguments or -j")
    if not multiple_runs_mode and args.audio_path is None and not args.record:
        parser.error("either an audio file or recording must be set")

    try:
        if multiple_runs_mode:
            run_multiple(
                args.manifest_path,
                args.runs_file,
                args.thread_spec,
                args.enable_op_profiling,
            )
        else:
            run_single(
                args.manifest_path,
                args.audio_path,
                args.thread_spec,
                args.enable_op_profiling,
                args.record,
                args.duration,
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
