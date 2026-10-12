#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Python CLI for the audio detection brick.

Usage:
  python -m cli_examples.audio_detection_cli -m <manifest_path> (-a <mono_audio_path> | -j <runs.json> | --record [--duration <seconds>]) [-t <thread_spec>] [-k <top_k>]
"""

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
    """Load float32 mono audio and return shape (N) plus int64 sample rate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)

    return mono.astype(np.float32, copy=False), np.asarray([file_sr], dtype=np.int64)


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

    # The audio detection API expects shape (N).
    return recording[:, 0].astype(np.float32, copy=False)


def run_single(
    manifest_path: str,
    audio_path: str | None,
    top_k: int,
    thread_spec: str,
    enable_op_profiling: bool,
    record: bool,
    duration: float,
) -> None:
    """Run one audio detection request from a file or microphone recording."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)

    manifest_path = get_manifest_path(manifest_path, "audio_detection")

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

    if top_k <= 0:
        raise ValueError("-k must be greater than 0")

    if record:
        raw_wav = record_audio(duration, sr=16000)
        sample_rate = np.asarray([16000], dtype=np.int64)
    else:
        resolved_audio_path = get_sample_path(audio_path, manifest_path)

        # Read audio file
        raw_wav, sample_rate = load_as_wav(resolved_audio_path)

    # Set the input
    input = cf.TensorDict()
    input["raw_wav"] = raw_wav
    input["sample_rate"] = sample_rate

    while not brick.is_generation_complete():
        # Run the model inference
        output = brick.run(input)

        scores = np.asarray(output["score"], dtype=np.float32)
        ids = np.asarray(output["id"], dtype=np.int64)
        names = np.asarray(output["name"], dtype=np.uint8)
        frame = np.asarray(output["frame"], dtype=np.float32).reshape(
            scores.shape[0], 2
        )

        num_classes_to_display = min(top_k, scores.shape[1])

        for frame_idx in range(scores.shape[0]):
            time_s = float(frame[frame_idx, 0])
            time_e = float(frame[frame_idx, 1])

            print(f"Timestamp [{time_s:.3f}:{time_e:.3f}]s:")
            for idx in range(num_classes_to_display):
                class_name = (
                    bytes(names[frame_idx, idx].tolist())
                    .split(b"\x00", 1)[0]
                    .decode("utf-8")
                )
                print(
                    f"    ({class_name} ({int(ids[frame_idx, idx])}), "
                    f"{float(scores[frame_idx, idx]):f})"
                )

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)

    manifest_path = get_manifest_path(manifest_path, "audio_detection")

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

    run_configs = load_run_configs(runs_file, {"a", "k"}, {"a"})
    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        audio_path = get_run_value(config, "a", "", run_index)
        top_k = get_run_value(config, "k", 3, run_index)
        if not audio_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'a' must be a non-empty string"
            )
        if top_k <= 0:
            raise ValueError(f"Run {run_index + 1} argument 'k' must be greater than 0")

        # Read audio file
        raw_wav, sample_rate = load_as_wav(audio_path)

        # Set the input
        input = cf.TensorDict()
        input["raw_wav"] = raw_wav
        input["sample_rate"] = sample_rate

        while not brick.is_generation_complete():
            # Run the model inference
            output = brick.run(input)

            scores = np.asarray(output["score"], dtype=np.float32)
            ids = np.asarray(output["id"], dtype=np.int64)
            names = np.asarray(output["name"], dtype=np.uint8)
            frame = np.asarray(output["frame"], dtype=np.float32).reshape(
                scores.shape[0], 2
            )

            num_classes_to_display = min(top_k, scores.shape[1])

            for frame_idx in range(scores.shape[0]):
                time_s = float(frame[frame_idx, 0])
                time_e = float(frame[frame_idx, 1])

                print(f"Timestamp [{time_s:.3f}:{time_e:.3f}]s:")
                for idx in range(num_classes_to_display):
                    class_name = (
                        bytes(names[frame_idx, idx].tolist())
                        .split(b"\x00", 1)[0]
                        .decode("utf-8")
                    )
                    print(
                        f"    ({class_name} ({int(ids[frame_idx, idx])}), "
                        f"{float(scores[frame_idx, idx]):f})"
                    )

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv=None) -> int:
    # ----- Parse the cmd line arguments
    # ----------------------------------
    # Required arguments
    parser = argparse.ArgumentParser(
        description="Audio detection brick CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument(
        "-a", dest="audio_path", help="Input audio file or @sample/... selector"
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        help="Run multiple entries without reloading the model",
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
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-k",
        dest="top_k",
        type=int,
        default=None,
        help="Maximum number of classes to display (default: 3)",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off)",
    )
    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = args.runs_file is not None
    if multiple_runs_mode and (
        args.audio_path is not None or args.record or args.top_k is not None
    ):
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
                args.top_k if args.top_k is not None else 3,
                args.thread_spec,
                args.enable_op_profiling,
                args.record,
                args.duration,
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    sys.exit(main())
