#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run a monocular depth-estimation brick and save a relative-depth visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import compute_flow_py as cf
import numpy as np
from PIL import Image

from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def load_image(path: str) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def default_output_path(input_path: str) -> Path:
    path = Path(input_path)
    return Path(f"{path.stem}_relative_depth.png")


def colorize_relative_depth(relative_depth: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [48, 18, 59],
            [50, 100, 180],
            [45, 200, 170],
            [245, 220, 70],
            [180, 4, 38],
        ],
        dtype=np.float32,
    )
    visualization = np.zeros((*relative_depth.shape, 3), dtype=np.uint8)
    valid = np.isfinite(relative_depth) & (relative_depth != 0.0)
    if not np.any(valid):
        return visualization

    minimum = float(relative_depth[valid].min())
    maximum = float(relative_depth[valid].max())
    normalized = np.full(relative_depth.shape, 0.5, dtype=np.float32)
    if maximum > minimum:
        normalized[valid] = (relative_depth[valid] - minimum) / (maximum - minimum)

    scaled = np.clip(normalized, 0.0, 1.0) * (len(palette) - 1)
    lower = np.minimum(np.floor(scaled).astype(np.int64), len(palette) - 2)
    weight = (scaled - lower)[..., None]
    colors = palette[lower] + ((palette[lower + 1] - palette[lower]) * weight)
    visualization[valid] = np.rint(colors[valid]).astype(np.uint8)
    return visualization


def run_single(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    input_value: str,
    output_path: Path | None,
) -> None:
    """Run one monocular depth-estimation request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "monocular_depth_estimation")

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

    input_path = get_sample_path(input_value, manifest_path)

    # Load the image
    image = load_image(input_path)
    _, input_height, input_width = image.shape

    inputs = cf.TensorDict()

    # Set the input
    inputs["image"] = image

    # Run the model inference
    while not brick.is_generation_complete():
        output = brick.run(inputs)

    relative_depth = np.asarray(output["relative_depth"], dtype=np.float32)
    output_height, output_width = relative_depth.shape
    print(f"Input image resolution: {input_width}x{input_height}")
    print(f"Output image resolution: {output_width}x{output_height}")

    # Display profiler run
    print(brick.write_profile(), end="")

    # Save the output image
    default_input = input_path if input_value == input_path else Path(input_path).name
    path = output_path or default_output_path(default_input)
    visualization = colorize_relative_depth(relative_depth)

    Image.fromarray(visualization).save(path)

    print(f"Wrote relative-depth visualization to {path}")


def run_multiple(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    runs_file: str,
) -> None:
    """Run monocular depth-estimation requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "monocular_depth_estimation")

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

    run_configs = load_run_configs(runs_file, {"i", "o"}, {"i"})

    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        output_path = get_run_value(config, "o", "", run_index)
        input_path = get_run_value(config, "i", "", run_index)

        if not input_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'i' must be a non-empty string"
            )

        if "o" in config and not output_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'o' must be a non-empty string"
            )

        # Load the image
        image = load_image(input_path)
        _, input_height, input_width = image.shape

        inputs = cf.TensorDict()

        # Set the input
        inputs["image"] = image

        # Run the model inference
        while not brick.is_generation_complete():
            output = brick.run(inputs)

        relative_depth = np.asarray(output["relative_depth"], dtype=np.float32)
        output_height, output_width = relative_depth.shape
        print(f"Input image resolution: {input_width}x{input_height}")
        print(f"Output image resolution: {output_width}x{output_height}")

        # Display profiler run
        print(brick.write_profile(), end="")

        if "o" not in config:
            continue

        # Save the output image
        visualization = colorize_relative_depth(relative_depth)

        Image.fromarray(visualization).save(output_path)

        print(f"Wrote relative-depth visualization to {output_path}")


def main(argv: list[str] | None = None) -> int:
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s -m <manifest.json> "
            "(-i <input_image> | -j <runs.json>) "
            "[-o <output.png>] [-t <thread_spec>]"
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
        dest="input_value",
        metavar="<input_image>",
        help="Image path or @sample/... selector.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON file path for multiple run settings.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        metavar="<output.png>",
        help="Relative-depth visualization path (default: ./<input-stem>_relative_depth.png).",
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
    if (not multiple_runs_mode and args.input_value is None) or (
        multiple_runs_mode
        and (args.input_value is not None or args.output_path is not None)
    ):
        parser.error("use either -i [-o] or -j")

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
                args.input_value,
                args.output_path,
            )
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
