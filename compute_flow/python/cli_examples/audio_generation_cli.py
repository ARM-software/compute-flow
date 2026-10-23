#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Python CLI for audio generation bricks."""

import argparse
import os
import sys

import compute_flow_py as cf
import numpy as np

from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def get_filename(prompt: str, seed: int) -> str:
    name = prompt.replace(" ", "_").lower()
    return f"{name}_{seed}.wav"


def save_as_wav(
    path: str,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int = 44100,
) -> None:
    """Save float32 stereo audio to a WAV file."""
    try:
        import soundfile as sf
    except ImportError:
        print(
            "Error: `soundfile` is required to save WAVs. Try: pip install soundfile",
            file=sys.stderr,
        )
        raise

    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)

    if left.shape[0] != right.shape[0]:
        raise ValueError("left_channel and right_channel must have the same length")

    stereo = np.stack([left, right], axis=1)

    sf.write(path, stereo, sample_rate, subtype="FLOAT")


def run_single(
    manifest_path: str,
    prompt: str,
    output_file: str | None,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "audio_generation")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)

    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    input_data = cf.TensorDict()

    # Set the input
    # Text prompt - shape[number of chars, 1]
    input_data["text_prompt"] = prompt

    # Run the model inference
    output = brick.run(input_data)

    # filename = <prompt>_<seed>.wav
    output_file = output_file or get_filename(prompt, manifest.get_constant("seed"))
    left = output["left_channel"]
    right = output["right_channel"]
    sample_rate = int(np.asarray(output["sample_rate"], dtype=np.int64).reshape(-1)[0])

    save_as_wav(output_file, left, right, sample_rate=sample_rate)
    print(f"Saved: {os.path.abspath(output_file)}")

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
    manifest_path = get_manifest_path(manifest_path, "audio_generation")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)

    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    run_configs = load_run_configs(runs_file, {"p", "o"}, {"p"})

    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        prompt = get_run_value(config, "p", "", run_index)
        if not prompt:
            raise ValueError(
                f"Run {run_index + 1} argument 'p' must be a non-empty string"
            )

        input_data = cf.TensorDict()

        # Set the input
        # Text prompt - shape[number of chars, 1]
        input_data["text_prompt"] = prompt

        # Run the model inference
        output = brick.run(input_data)

        if "o" in config:
            output_file = get_run_value(config, "o", "", run_index)
            if not output_file:
                raise ValueError(
                    f"Run {run_index + 1} argument 'o' must be a non-empty string"
                )

            # filename = <prompt>_<seed>.wav
            left = output["left_channel"]
            right = output["right_channel"]
            sample_rate = int(
                np.asarray(output["sample_rate"], dtype=np.int64).reshape(-1)[0]
            )

            save_as_wav(output_file, left, right, sample_rate=sample_rate)
            print(f"Saved: {os.path.abspath(output_file)}")

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audio generation brick CLI (Python version of the C++ tool)",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument("-p", dest="prompt", help="Input prompt text")
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
        default=None,
        help="Output WAV file name (default: <prompt>_<seed>.wav)",
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-op profiling (default: off)",
    )
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )

    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and args.prompt is None) or (
        multiple_runs_mode and (args.prompt is not None or args.output_file is not None)
    ):
        parser.error("use either -p [-o] or -j")

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
                args.prompt,
                args.output_file,
                args.thread_spec,
                args.enable_op_profiling,
            )
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
