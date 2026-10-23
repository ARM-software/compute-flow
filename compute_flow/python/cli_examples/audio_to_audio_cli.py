#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Python CLI for the audio-to-audio brick."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import compute_flow_py as cf
import numpy as np
import soundfile as sf
from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def default_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}_enhanced.wav"))


def load_as_wav(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mono float32 audio and return shape (N) plus int64 sample rate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)

    return mono.astype(np.float32, copy=False), np.asarray([file_sr], dtype=np.int64)


def run_single(
    manifest_path: str,
    audio_path: str,
    output_file: str | None,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_to_audio")
    resolved_audio_path = get_sample_path(audio_path, manifest_path)
    manifest = cf.load_manifest(manifest_path)
    set_cpu_threads(manifest, default_threads, model_threads)
    brick = cf.load_brick(manifest)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling
    brick.setup(manifest, options)
    brick.reset_states()

    # Read audio file
    raw_wav, sample_rate = load_as_wav(resolved_audio_path)

    input_data = cf.TensorDict()

    # Set the input
    input_data["raw_wav"] = raw_wav
    input_data["sample_rate"] = sample_rate

    enhanced_chunks = []

    # Run the model inference
    while not brick.is_generation_complete():
        output = brick.run(input_data)
        enhanced_chunks.append(
            np.asarray(output["enhanced_wav"], dtype=np.float32).reshape(-1).copy()
        )
        output_sample_rate = int(
            np.asarray(output["sample_rate"], dtype=np.int64).reshape(-1)[0]
        )

    enhanced_wav = np.concatenate(enhanced_chunks)
    output_file = output_file or default_output_path(resolved_audio_path)

    sf.write(output_file, enhanced_wav, output_sample_rate, subtype="FLOAT")
    print(f"Wrote enhanced audio to {output_file}")

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_to_audio")
    manifest = cf.load_manifest(manifest_path)
    set_cpu_threads(manifest, default_threads, model_threads)
    brick = cf.load_brick(manifest)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling
    brick.setup(manifest, options)

    run_configs = load_run_configs(runs_file, {"a", "o"}, {"a"})
    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        audio_path = get_run_value(config, "a", "", run_index)
        if not audio_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'a' must be a non-empty string"
            )

        # Read audio file
        raw_wav, sample_rate = load_as_wav(audio_path)

        input_data = cf.TensorDict()

        # Set the input
        input_data["raw_wav"] = raw_wav
        input_data["sample_rate"] = sample_rate

        enhanced_chunks = []

        # Run the model inference
        while not brick.is_generation_complete():
            output = brick.run(input_data)
            enhanced_chunks.append(
                np.asarray(output["enhanced_wav"], dtype=np.float32).reshape(-1).copy()
            )
            output_sample_rate = int(
                np.asarray(output["sample_rate"], dtype=np.int64).reshape(-1)[0]
            )

        output_file = get_run_value(config, "o", "", run_index)
        if not output_file:
            output_file = default_output_path(audio_path)

        enhanced_wav = np.concatenate(enhanced_chunks)

        sf.write(output_file, enhanced_wav, output_sample_rate, subtype="FLOAT")
        print(f"Wrote enhanced audio to {output_file}")

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audio-to-audio brick CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument(
        "-a",
        dest="audio_path",
        help="Input audio file or @sample/... selector",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        help="Run multiple entries without reloading the model",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        help="Output enhanced mono float32 WAV file",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off)",
    )
    args = parser.parse_args(argv)

    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and args.audio_path is None) or (
        multiple_runs_mode
        and (args.audio_path is not None or args.output_file is not None)
    ):
        parser.error("use either -a [-o] or -j")

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
                args.output_file,
                args.thread_spec,
                args.enable_op_profiling,
            )
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
