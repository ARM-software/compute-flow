#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an image classification brick and display its highest scoring classes."""

from __future__ import annotations

import argparse

import compute_flow_py as cf
import numpy as np
from PIL import Image

from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)

_DEFAULT_TOP_K = 5


def load_image(path: str) -> np.ndarray:
    """Load an RGB image as normalized planar float32 data."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def display_result(output: cf.TensorDict, top_k: int) -> None:
    """Print the highest scoring classes in the brick's ranked output."""
    ids = np.asarray(output["id"])
    scores = np.asarray(output["score"])
    names = np.asarray(output["name"])

    if ids.dtype != np.int64 or ids.ndim != 1 or len(ids) == 0:
        raise ValueError("Class IDs must be a non-empty int64 vector")
    if scores.dtype != np.float32 or scores.shape != ids.shape:
        raise ValueError("Scores must be a float32 vector aligned with class IDs")
    if (
        names.dtype != np.uint8
        or names.ndim != 2
        or names.shape[0] != len(ids)
        or names.shape[1] == 0
    ):
        raise ValueError("Names must be a uint8 matrix aligned with class IDs")

    for index in range(min(top_k, len(ids))):
        name = bytes(names[index]).split(b"\0", 1)[0].decode("utf-8")
        print(
            f"{index + 1}. {name} (id {int(ids[index])}, score {float(scores[index]):.4f})"
        )


def run_single(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    image_value: str,
    top_k: int,
) -> None:
    """Run one image classification request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_classification")

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

    image_path = get_sample_path(image_value, manifest_path)

    # Load the image
    image = load_image(image_path)

    inputs = cf.TensorDict()

    # Set the input
    inputs["image"] = image

    # Run the model inference
    while not brick.is_generation_complete():
        output = brick.run(inputs)

    display_result(output, top_k)

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    runs_file: str,
) -> None:
    """Run image classification requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_classification")

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

    run_configs = load_run_configs(runs_file, {"i", "k"}, {"i"})

    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        image_path = get_run_value(config, "i", "", run_index)
        top_k = get_run_value(config, "k", _DEFAULT_TOP_K, run_index)

        if not image_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'i' must be a non-empty string"
            )
        if top_k <= 0:
            raise ValueError(f"Run {run_index + 1} argument 'k' must be greater than 0")

        # Load the image
        image = load_image(image_path)

        inputs = cf.TensorDict()

        # Set the input
        inputs["image"] = image

        # Run the model inference
        while not brick.is_generation_complete():
            output = brick.run(inputs)

        display_result(output, top_k)

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch one or more classification requests."""
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s -m <manifest.json> "
            "(-i <input_image> [-k <top_k>] | -j <runs.json>) [-t <thread_spec>]"
        ),
        add_help=False,
    )
    parser.add_argument(
        "-m",
        "--manifest",
        dest="manifest_path",
        required=True,
        metavar="<manifest.json>",
        help="Manifest path or @manifest/... selector.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="image_value",
        metavar="<input_image>",
        help="Image path or @sample/... selector.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON for multiple runs; each entry needs id and i, and may set k.",
    )
    parser.add_argument(
        "-k",
        dest="top_k",
        type=int,
        metavar="<top_k>",
        help="Maximum number of classes to display (default: 5).",
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
    parser.add_argument("-h", "--help", action="help", help="Show this help message.")

    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and args.image_value is None) or (
        multiple_runs_mode and (args.image_value is not None or args.top_k is not None)
    ):
        parser.error("use either -i [-k] or -j")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("-k must be greater than 0")

    try:
        if multiple_runs_mode:
            run_multiple(
                args.manifest_path,
                args.thread_spec,
                args.enable_op_profiling,
                args.runs_file,
            )
        else:
            run_single(
                args.manifest_path,
                args.thread_spec,
                args.enable_op_profiling,
                args.image_value,
                args.top_k if args.top_k is not None else _DEFAULT_TOP_K,
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
