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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--manifest", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Relative-depth visualization path "
            "(default: ./<input-stem>_relative_depth.png)"
        ),
    )
    parser.add_argument("-t", "--threads", default="1")
    parser.add_argument("--enable-op-profiling", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = get_manifest_path(args.manifest, "monocular_depth_estimation")
    input_path = get_sample_path(args.input, manifest_path)
    output_path = args.output or default_output_path(input_path)
    image = load_image(input_path)
    _, input_height, input_width = image.shape
    default_threads, model_threads = parse_hint_cpu_threads(args.threads)
    manifest = cf.load_manifest(manifest_path)
    # Configure default and per-model CPU thread counts
    set_cpu_threads(manifest, default_threads, model_threads)

    brick = cf.load_brick(manifest)
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = args.enable_op_profiling
    brick.setup(manifest, options)

    inputs = cf.TensorDict()
    inputs["image"] = image
    output = None
    while not brick.is_generation_complete():
        output = brick.run(inputs)
    if output is None:
        raise RuntimeError("Monocular depth-estimation brick did not produce output")
    print(brick.write_profile(), end="")

    relative_depth = np.asarray(output["relative_depth"], dtype=np.float32)
    output_height, output_width = relative_depth.shape
    visualization = colorize_relative_depth(relative_depth)
    Image.fromarray(visualization).save(output_path)
    print(f"Input image resolution: {input_width}x{input_height}")
    print(f"Output image resolution: {output_width}x{output_height}")
    print(f"Wrote relative-depth visualization to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
