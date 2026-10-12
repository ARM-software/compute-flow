#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Python CLI for the audio embedding brick."""

from __future__ import annotations

import argparse
import os

import compute_flow_py as cf
import numpy as np
import soundfile as sf
from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)

_NUM_EMBEDDING_EDGE_VALUES = 3


def _format_embedding(embedding: np.ndarray) -> str:
    return np.array2string(
        embedding,
        precision=2,
        threshold=2 * _NUM_EMBEDDING_EDGE_VALUES,
        edgeitems=_NUM_EMBEDDING_EDGE_VALUES,
        separator=", ",
        floatmode="fixed",
    )


def load_as_wav(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load float32 mono audio and return its waveform and sample rate."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    data, file_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)

    waveform = mono.astype(np.float32, copy=False)
    sample_rate = np.asarray([file_sample_rate], dtype=np.int64)

    return waveform, sample_rate


def run_single(
    manifest_path: str,
    audio_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Extract audio embeddings from one audio file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_embedding")

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

    resolved_audio_path = get_sample_path(audio_path, manifest_path)

    # Read the audio file and set the input
    waveform, sample_rate = load_as_wav(resolved_audio_path)
    input_data = cf.TensorDict()
    input_data["raw_wav"] = waveform
    input_data["sample_rate"] = sample_rate

    # Run the model inference
    while not brick.is_generation_complete():
        output = brick.run(input_data)

        embeddings = np.asarray(output["embedding"], dtype=np.float32)
        frames = np.asarray(output["frame"], dtype=np.float32)

        for embedding_index in range(embeddings.shape[0]):
            start_time = float(frames[embedding_index, 0])
            end_time = float(frames[embedding_index, 1])

            print(f"Timestamp [{start_time:.3f}:{end_time:.3f}]s:")
            print(f"Embedding: {_format_embedding(embeddings[embedding_index])}")

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Extract audio embeddings for each entry in a JSON runs file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_embedding")

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
        waveform, sample_rate = load_as_wav(audio_path)
        input_data = cf.TensorDict()
        input_data["raw_wav"] = waveform
        input_data["sample_rate"] = sample_rate

        # Run the model inference
        while not brick.is_generation_complete():
            output = brick.run(input_data)

            embeddings = np.asarray(output["embedding"], dtype=np.float32)
            frames = np.asarray(output["frame"], dtype=np.float32)

            for embedding_index in range(embeddings.shape[0]):
                start_time = float(frames[embedding_index, 0])
                end_time = float(frames[embedding_index, 1])

                print(f"Timestamp [{start_time:.3f}:{end_time:.3f}]s:")
                print(f"Embedding: {_format_embedding(embeddings[embedding_index])}")

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the audio embedding CLI."""
    parser = argparse.ArgumentParser(
        description="Audio embedding brick CLI",
        usage=(
            "%(prog)s -m <manifest_path> "
            "(-a <audio.wav> | -j <runs.json>) [-t <thread_spec>]"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        metavar="<manifest_path>",
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument(
        "-a",
        dest="audio_path",
        metavar="<audio.wav>",
        help="Input audio path or @sample/... selector.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON file path for multiple run settings.",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        metavar="<thread_spec>",
        help=(
            "Number of threads to use. Specify a default for all models and optionally "
            "override individual models, e.g. "
            "<default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)"
        ),
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off).",
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )

    args = parser.parse_args(argv)

    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and args.audio_path is None) or (
        multiple_runs_mode and args.audio_path is not None
    ):
        parser.error("use either -a or -j")

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
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
