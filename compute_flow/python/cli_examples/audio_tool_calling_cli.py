#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an audio tool calling brick and display its structured tool calls."""

from __future__ import annotations

import argparse
import json

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
    """Load a mono float32 waveform and its sample rate."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)

    if data.shape[1] != 1:
        raise ValueError("Input WAV must be mono")

    return data[:, 0], np.asarray([sample_rate], dtype=np.int64)


def get_calls_chunk(output: cf.TensorDict) -> str:
    """Read one UTF-8 output chunk from the brick."""
    chunk = np.asarray(output["calls"]).item()

    if isinstance(chunk, bytes):
        return chunk.decode("utf-8")
    if not isinstance(chunk, str):
        raise ValueError("Tool calls output must be a UTF-8 string")

    return chunk


def validate_calls_v1(calls: object) -> None:
    """Validate the output against audio_tool_calling.v1's JSON schema."""
    if not isinstance(calls, list):
        raise ValueError("Tool calls output must be a JSON array")

    for index, call in enumerate(calls):
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str)
            or not isinstance(call["arguments"], dict)
        ):
            raise ValueError(
                f"Tool call at index {index} must contain only a string 'name' "
                "and an object 'arguments'"
            )


def generate_calls(brick, audio_path: str) -> None:
    """Generate tool calls from one WAV file using an initialized brick."""
    brick.reset_states()

    # Read audio file
    raw_wav, sample_rate = load_as_wav(audio_path)

    inputs = cf.TensorDict()

    # Set the input
    inputs["raw_wav"] = raw_wav
    inputs["sample_rate"] = sample_rate

    # Run the model inference; JSON may be incomplete until generation finishes.
    chunks: list[str] = []

    print("Tool calls:")

    while not brick.is_generation_complete():
        output = brick.run(inputs)
        chunk = get_calls_chunk(output)

        chunks.append(chunk)
        print(chunk, end="", flush=True)

    print()

    calls = json.loads("".join(chunks))

    validate_calls_v1(calls)

    # Display profiler run
    print(brick.write_profile(), end="")


def run_single(
    manifest_path: str,
    audio_value: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run one audio tool calling request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_tool_calling")

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

    audio_path = get_sample_path(audio_value, manifest_path)

    generate_calls(brick, audio_path)


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run audio tool calling requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_tool_calling")

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
        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        audio_path = get_run_value(config, "a", "", run_index)

        if not audio_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'a' must be a non-empty string"
            )

        generate_calls(brick, audio_path)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch one or more audio tool calling requests."""
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s -m <manifest.json> "
            "(-a <audio.wav> | -j <runs.json>) [-t <thread_spec>]"
        ),
    )
    parser.add_argument(
        "-m",
        "--manifest",
        dest="manifest_path",
        required=True,
        metavar="<manifest.json>",
        help="Manifest path or @manifest/... selector.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-a",
        "--input",
        dest="audio_value",
        metavar="<audio.wav>",
        help="Mono WAV path or @sample/... selector.",
    )
    input_group.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON for multiple runs; each entry needs id and a.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="thread_spec",
        default="1",
        metavar="<thread_spec>",
        help="Default threads and optional model overrides (default: 1).",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-operation profiling.",
    )
    args = parser.parse_args(argv)

    try:
        if args.runs_file is not None:
            run_multiple(
                args.manifest_path,
                args.runs_file,
                args.thread_spec,
                args.enable_op_profiling,
            )
        else:
            run_single(
                args.manifest_path,
                args.audio_value,
                args.thread_spec,
                args.enable_op_profiling,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
