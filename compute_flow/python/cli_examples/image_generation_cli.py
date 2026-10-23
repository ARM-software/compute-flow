#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an image-generation brick and save the generated image as PNG."""

from __future__ import annotations

import argparse

import compute_flow_py as cf
import numpy as np
from PIL import Image

from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)

_NUM_CHANNELS = 3


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a normalized planar RGB image as PNG."""
    if image.ndim != 3 or image.shape[0] != _NUM_CHANNELS:
        raise ValueError("Image-generation brick returned an unsupported output format")
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("Image-generation brick returned an empty image")

    interleaved = np.transpose(image, (1, 2, 0))
    output = np.rint(np.clip(interleaved, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(output).save(output_path, format="PNG")

    _, height, width = image.shape
    print(f"Output resolution: {width} x {height}")
    print(f"Wrote generated image to {output_path}")


def run_single(
    manifest_path: str,
    prompt: str,
    output_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run one image-generation request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_generation")

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

    # Set the input
    inputs = cf.TensorDict()
    inputs["prompt"] = prompt

    output = cf.TensorDict()

    # Run the model inference
    while not brick.is_generation_complete():
        output = brick.run(inputs)

    # Save the output image
    save_image(np.asarray(output["image"], dtype=np.float32), output_path)

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run image-generation requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_generation")

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

    run_configs = load_run_configs(runs_file, {"p", "o"}, {"p", "o"})

    for run_index, config in enumerate(run_configs, start=1):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index), run_index)
        prompt = get_run_value(config, "p", "", run_index)
        output_path = get_run_value(config, "o", "", run_index)
        if not prompt:
            raise ValueError(f"Run {run_index} argument 'p' must be a non-empty string")
        if not output_path:
            raise ValueError(f"Run {run_index} argument 'o' must be a non-empty string")

        print(f"Run {run_id}")

        # Set the input
        inputs = cf.TensorDict()
        inputs["prompt"] = prompt

        output = cf.TensorDict()

        # Run the model inference
        while not brick.is_generation_complete():
            output = brick.run(inputs)

        # Save the output image
        save_image(np.asarray(output["image"], dtype=np.float32), output_path)

        # Display profiler run
        print(brick.write_profile(), end="")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the image-generation example."""
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s -m <manifest.json> "
            "(-p <prompt> -o <output.png> | -j <runs.json>) [-t <thread_spec>]"
        ),
        add_help=False,
    )
    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        metavar="<manifest.json>",
        help="Manifest path or @manifest/... selector.",
    )
    parser.add_argument(
        "-p",
        dest="prompt",
        metavar="<prompt>",
        help="Text describing the image to generate.",
    )
    parser.add_argument(
        "-o",
        dest="output_path",
        metavar="<output.png>",
        help="Output PNG path.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="JSON file with prompt and output path for each run.",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        metavar="<thread_spec>",
        help=(
            "Default threads and optional model overrides, e.g. "
            "4,text_encoder:2 (default: 1)."
        ),
    )
    parser.add_argument(
        "--enable-op-profiling",
        action="store_true",
        help="Enable per-operation profiling.",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message.")

    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = bool(args.runs_file)
    direct_arguments_present = args.prompt is not None or args.output_path is not None
    if (
        not args.manifest_path
        or (not multiple_runs_mode and (not args.prompt or not args.output_path))
        or (multiple_runs_mode and direct_arguments_present)
    ):
        parser.error("use either -p -o or -j with a non-empty manifest path")

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
                args.output_path,
                args.thread_spec,
                args.enable_op_profiling,
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
