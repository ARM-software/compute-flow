#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python CLI for the NeuralCodec brick — mirrors the C++ tool one-to-one.

Usage:
  python -m cli_examples.neural_codec_cli -m <manifest_path> -c <codes> [-t <thread_spec>] [-o <output_file>]
"""

import argparse
import sys

import numpy as np
import soundfile as sf

import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def save_as_wav(
    path: str, left: np.ndarray, right: np.ndarray, sample_rate: int
) -> None:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Left and right channel sizes do not match")
    stereo = np.stack([left, right], axis=1)
    sf.write(path, stereo, sample_rate, subtype="FLOAT")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="NeuralCodec brick CLI (Python version of the C++ tool)",
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
        "-c",
        dest="codes_path",
        required=True,
        help="Neural codec codes path or @sample/... selector (.npy)",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-o", dest="output_file", default="output.wav", help="Output wav file"
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

    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(args.thread_spec)

    manifest_path = get_manifest_path(args.manifest_path, "neural_codec")
    codes_path = get_sample_path(args.codes_path, manifest_path)

    try:
        codes = np.load(codes_path)
        if codes.size == 0:
            raise ValueError(f"Codes file is empty: {codes_path}")
        codes = codes.astype(np.int32, copy=False)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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

    # Set the input
    codes = codes.reshape(1, 1, codes.size)  # batch, mono, time
    input_td = {"codes": codes}

    # Run the model inference
    output = brick.run(input_td)

    # Display profiler run
    print(brick.write_profile(), end="")

    sample_rate = manifest.get_constant("output_sample_rate")
    left = output["left_channel"]
    right = output["right_channel"]
    save_as_wav(args.output_file, left, right, sample_rate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
