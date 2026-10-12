#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an image segmentation brick and save a colorized class mask."""

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
    """Load an RGB image as normalized planar float32 data."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def default_output_path(input_path: str, run_number: int = 0) -> Path:
    """Choose a visualization path in the current directory."""
    path = Path(input_path)
    suffix = f"_{run_number}" if run_number else ""
    return Path(f"{path.stem}_segmentation{suffix}.png")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map each class ID to a stable display color."""
    values = mask.astype(np.uint32) * np.uint32(0x9E3779B1)
    visualization = np.empty((*mask.shape, 3), dtype=np.uint8)
    visualization[..., 0] = (values >> 16).astype(np.uint8)
    visualization[..., 1] = (values >> 8).astype(np.uint8)
    visualization[..., 2] = values.astype(np.uint8)
    return visualization


def display_result(
    output: cf.TensorDict, width: int, height: int, output_path: Path
) -> None:
    """Print the class table and save a colorized mask."""
    mask = np.asarray(output["mask"])
    class_names = np.asarray(output["class_name"])

    if mask.dtype != np.int32 or mask.shape != (height, width):
        raise ValueError("The segmentation mask must be int32 at input resolution")
    if (
        class_names.dtype != np.uint8
        or class_names.ndim != 2
        or class_names.shape[0] == 0
        or class_names.shape[1] == 0
    ):
        raise ValueError("Class names must be a non-empty uint8 matrix")
    if mask.min() < 0 or mask.max() >= len(class_names):
        raise ValueError("Mask IDs must index class-name rows")

    print(f"Segmentation resolution: {width}x{height}")
    print("Classes:")

    for class_id, name_bytes in enumerate(class_names):
        name = bytes(name_bytes).split(b"\0", 1)[0].decode("utf-8")
        print(f"{class_id} {name}")

    # Save the visualization
    visualization = colorize_mask(mask)

    Image.fromarray(visualization).save(output_path)

    print(f"Wrote segmentation visualization to {output_path}")


def run_single(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    input_value: str,
    output_path: str | None,
) -> None:
    """Run one image segmentation request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_segmentation")

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

    default_input = input_path if input_value == input_path else Path(input_path).name
    path = Path(output_path) if output_path else default_output_path(default_input)

    display_result(output, input_width, input_height, path)

    # Display profiler run
    print(brick.write_profile(), end="")


def run_multiple(
    manifest_path: str,
    thread_spec: str,
    enable_op_profiling: bool,
    runs_file: str,
) -> None:
    """Run image segmentation requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_segmentation")

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

        path = (
            Path(output_path)
            if "o" in config
            else default_output_path(input_path, run_index + 1)
        )

        display_result(output, input_width, input_height, path)

        # Display profiler run
        print(brick.write_profile(), end="")


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
        help="Input JSON for multiple runs; each run saves a PNG.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        metavar="<output.png>",
        help="Segmentation visualization path (single-run default: ./<input-stem>_segmentation.png; multi-run defaults add the run number, and JSON 'o' overrides it).",
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
