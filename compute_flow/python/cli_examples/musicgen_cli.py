#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python CLI for the MusicGen brick — mirrors the C++ tool one-to-one.

Usage:
  python -m cli_examples.musicgen_cli -m <manifest_path> -p <prompt> [-t <thread_spec>] [-s <seed> -l <audio_len> -n <num_steps> -o <output_file>]
"""

import argparse
import numpy as np
import os
import sys
from typing import Dict
import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


# ---------------------------------------------------------------------
# Default constants (same as your C++)
# ---------------------------------------------------------------------
K_SEED_DEFAULT = 99
K_AUDIO_LEN_SEC_DEFAULT = 180.0
K_NUM_STEPS_DEFAULT = 8


def get_filename(prompt: str, seed: int) -> str:
    name = prompt.replace(" ", "_").lower()
    return f"{name}_{seed}.wav"


def save_as_wav(
    path: str, left: np.ndarray, right: np.ndarray, sr: int = 44100
) -> None:
    """
    Save float32 stereo WAV using soundfile. Matches your C++ (IEEE float).
    """
    try:
        import soundfile as sf  # pip install soundfile
    except Exception as e:
        print(
            "Error: `soundfile` is required to save WAVs. Try: pip install soundfile",
            file=sys.stderr,
        )
        raise

    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)

    if left.shape[0] != right.shape[0]:
        raise ValueError("left_channel and right_channel must have the same length")

    stereo = np.stack([left, right], axis=1)  # (T, 2)
    sf.write(path, stereo, sr, subtype="FLOAT")  # 32-bit float WAV


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="MusicGen brick CLI (Python version of the C++ tool)",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument("-p", dest="prompt", required=True, help="Input prompt text")
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-s",
        dest="seed",
        type=int,
        default=K_SEED_DEFAULT,
        help=f"Random seed (default: {K_SEED_DEFAULT})",
    )
    parser.add_argument(
        "-l",
        dest="audio_len_sec",
        type=float,
        default=K_AUDIO_LEN_SEC_DEFAULT,
        help=f"Audio length in seconds (default: {K_AUDIO_LEN_SEC_DEFAULT})",
    )
    parser.add_argument(
        "-n",
        dest="num_steps",
        type=int,
        default=K_NUM_STEPS_DEFAULT,
        help=f"Number of steps (default: {K_NUM_STEPS_DEFAULT})",
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

    if not args.manifest_path or not args.prompt:
        parser.print_help()
        sys.exit("ERROR: Missing required arguments (-m, -p).")

    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(args.thread_spec)

    manifest_path = get_manifest_path(args.manifest_path, "musicgen")

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
    input["text_prompt"] = args.prompt
    input["seed"] = np.array([[np.uint64(args.seed)]], dtype=np.uint64)
    input["audio_length_sec"] = np.array(
        [np.float32(args.audio_len_sec)], dtype=np.float32
    )

    # Run the model inference
    output = brick.run(input)

    # Display profiler run
    print(brick.write_profile(), end="")

    # filename = <prompt>_<seed>.wav
    output_file = args.output_file or get_filename(
        args.prompt, manifest.get_constant("seed")
    )

    left = output["left_channel"]
    right = output["right_channel"]

    save_as_wav(output_file, left, right, sr=44100)
    print(f"Saved: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    sys.exit(main())
