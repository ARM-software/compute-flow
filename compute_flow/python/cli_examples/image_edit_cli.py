#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Run an image-edit brick and save the edited image."""

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

_NUM_CHANNELS = 3
_DIMENSION_MULTIPLE = 16


def load_image(path: str) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def pad_image(image: np.ndarray) -> np.ndarray:
    _, height, width = image.shape
    padded_height = (
        (height + _DIMENSION_MULTIPLE - 1) // _DIMENSION_MULTIPLE
    ) * _DIMENSION_MULTIPLE
    padded_width = (
        (width + _DIMENSION_MULTIPLE - 1) // _DIMENSION_MULTIPLE
    ) * _DIMENSION_MULTIPLE

    return np.pad(
        image,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="constant",
    )


def default_output_path(input_path: str) -> Path:
    path = Path(input_path)

    return path.with_name(f"{path.stem}_edited.png")


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    if image.ndim != 3 or image.shape[0] != _NUM_CHANNELS:
        raise RuntimeError("Image-edit brick returned an unsupported output format")

    interleaved = np.transpose(image, (1, 2, 0))
    output = np.rint(np.clip(interleaved, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(output).save(output_path, format="PNG")


def run_single(
    manifest_path: str,
    input_value: str,
    prompt: str,
    output_path: Path | None,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run one image-edit request."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_edit")

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
    padded_image = pad_image(image)

    inputs = cf.TensorDict()

    # Set the input
    inputs["image"] = padded_image
    inputs["prompt"] = prompt

    # Run the model inference
    output = brick.run(inputs)

    output_image = np.asarray(output["image"], dtype=np.float32)
    _, source_height, source_width = image.shape
    _, output_height, output_width = output_image.shape
    print(f"Original input resolution: {source_width} x {source_height}")
    print(f"Output resolution: {output_width} x {output_height}")

    # Display profiler run
    print(brick.write_profile(), end="")

    # Save the output image
    default_input = input_path if input_value == input_path else Path(input_path).name
    path = output_path or default_output_path(default_input)

    save_image(output_image, path)

    print(f"Wrote edited image to {path}")


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> None:
    """Run image-edit requests from a JSON file."""
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "image_edit")

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

    run_configs = load_run_configs(runs_file, {"i", "p", "o"}, {"i", "p"})

    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        output_path = get_run_value(config, "o", "", run_index)
        input_path = get_run_value(config, "i", "", run_index)
        prompt = get_run_value(config, "p", "", run_index)

        if not input_path:
            raise ValueError(
                f"Run {run_index + 1} argument 'i' must be a non-empty string"
            )

        if not prompt:
            raise ValueError(
                f"Run {run_index + 1} argument 'p' must be a non-empty string"
            )

        # Load the image
        image = load_image(input_path)
        padded_image = pad_image(image)

        inputs = cf.TensorDict()

        # Set the input
        inputs["image"] = padded_image
        inputs["prompt"] = prompt

        # Run the model inference
        output = brick.run(inputs)

        output_image = np.asarray(output["image"], dtype=np.float32)
        _, source_height, source_width = image.shape
        _, output_height, output_width = output_image.shape
        print(f"Original input resolution: {source_width} x {source_height}")
        print(f"Output resolution: {output_width} x {output_height}")

        # Display profiler run
        print(brick.write_profile(), end="")

        # Save the output image
        path = output_path or default_output_path(input_path)

        save_image(output_image, path)

        print(f"Wrote edited image to {path}")


def main(argv: list[str] | None = None) -> int:
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s -m <manifest.json> "
            "(-i <input_image> -p <prompt> | -j <runs.json>) "
            "[-o <output.png>] [-t <thread_spec>]"
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
        "-i",
        dest="input_path",
        metavar="<input_image>",
        help="Image path or @sample/... selector.",
    )
    parser.add_argument(
        "-p",
        dest="prompt",
        metavar="<prompt>",
        help="Natural-language editing instruction.",
    )
    parser.add_argument(
        "-j",
        dest="runs_file",
        metavar="<runs.json>",
        help="Input JSON file path for multiple run settings.",
    )
    parser.add_argument(
        "-o",
        dest="output_path",
        type=Path,
        metavar="<output.png>",
        help="Output PNG path (default: <input>_edited.png).",
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
    parser.add_argument("-h", action="help", help="Show this help message.")

    args = parser.parse_args(argv)

    # Check the mandatory arguments
    multiple_runs_mode = args.runs_file is not None
    direct_arguments_present = any(
        value is not None for value in (args.input_path, args.prompt, args.output_path)
    )
    if (
        not multiple_runs_mode and (args.input_path is None or args.prompt is None)
    ) or (multiple_runs_mode and direct_arguments_present):
        parser.error("use either -i -p [-o] or -j")
    if not multiple_runs_mode and not args.prompt:
        parser.error("argument -p: must not be empty")

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
                args.input_path,
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
