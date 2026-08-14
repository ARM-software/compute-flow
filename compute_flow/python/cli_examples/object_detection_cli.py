#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python CLI for the object detection brick — mirrors the C++ tool one-to-one.

Usage:
  python -m cli_examples.object_detection_cli -m <manifest_path> -i <image> [-t <thread_spec>]
"""

import argparse
import numpy as np
import os
import sys
from typing import Iterable, Sequence
import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path, get_sample_path
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)

from PIL import Image, ImageDraw, ImageFont


def image_to_npy(image_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load an image file and convert it to a numpy tensor suitable for the object detection brick.

    Returns:
        np.ndarray of shape (1, 3, H, W), dtype=float32
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Convert to numpy (H, W, C)
    img_np = np.asarray(img)

    # Convert to float32
    img_np = img_np.astype(np.float32)

    # Normalize to [0, 1]
    if normalize:
        img_np /= 255.0

    # HWC -> CHW
    img_np = np.transpose(img_np, (2, 0, 1))

    # Add batch dimension: (1, 3, H, W)
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


def decode_name_row(row: Sequence[int]) -> str:
    """Decode one row (numpy uint8 row or bytes-like) into utf-8 string, stripping trailing NULs."""
    if isinstance(row, (bytes, bytearray)):
        b = bytes(row)
    else:
        # Convert to bytes from iterable of ints
        b = bytes(np.asarray(row).reshape(-1).tolist())
    return b.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def load_font(font_size):
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux®
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS®
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "DejaVuSans.ttf",
    ]

    for path in possible_fonts:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    print("Warning: using default font (size not adjustable)")
    return ImageFont.load_default()


def draw_detections_and_save(
    image_path_or_array,
    boxes: np.ndarray,
    class_ids: Sequence[int],
    scores: Sequence[float],
    class_names_rows: Iterable[Sequence[int]],
    save_path: str,
    score_fmt: str = "{:.2f}",
    linewidth: int = 2,
    font_size: int = 14,
) -> None:
    """
    Draw detections onto the ORIGINAL image and save.

    - image_path_or_array: path to original image or numpy HxWxC uint8 array
    - boxes: (N,4) array-like, each box [x1,y1,x2,y2] in original image coordinates.
    """
    # Load original image (we draw onto the original)
    if isinstance(image_path_or_array, str):
        orig_img = Image.open(image_path_or_array).convert("RGB")
        orig_w, orig_h = orig_img.size
        orig_arr = None
    else:
        orig_arr = np.asarray(image_path_or_array)
        if orig_arr.dtype != np.uint8:
            orig_arr = (
                255 * (orig_arr.astype(np.float32) / max(1.0, float(orig_arr.max())))
            ).astype(np.uint8)
        if orig_arr.ndim == 2:
            orig_arr = np.stack([orig_arr] * 3, axis=-1)
        orig_h, orig_w = orig_arr.shape[:2]
        orig_img = Image.fromarray(orig_arr.astype(np.uint8)).convert("RGB")

    boxes = np.asarray(boxes, dtype=np.float32)
    class_ids = np.asarray(class_ids).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    class_names_rows = list(class_names_rows)
    decoded_names = [decode_name_row(r) for r in class_names_rows]

    if boxes.size == 0:
        print("No boxes to draw.")
        orig_img.save(save_path)
        print(f"Saved: {save_path}")
        return

    # Normalize shapes
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes.reshape(1, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be shape (N,4)")

    boxes_orig = boxes.copy()
    boxes_orig[:, 0] = np.clip(boxes_orig[:, 0], 0, orig_w - 1)
    boxes_orig[:, 1] = np.clip(boxes_orig[:, 1], 0, orig_h - 1)
    boxes_orig[:, 2] = np.clip(boxes_orig[:, 2], 0, orig_w - 1)
    boxes_orig[:, 3] = np.clip(boxes_orig[:, 3], 0, orig_h - 1)

    draw = ImageDraw.Draw(orig_img)

    # Try to load a truetype font; fall back to default
    try:
        font = load_font(font_size)
    except Exception:
        font = ImageFont.load_default()

    def color_for_id(cid: int):
        # deterministic RGB from id
        r = (int(cid) * 37 + 13) % 255
        g = (int(cid) * 97 + 29) % 255
        b = (int(cid) * 61 + 47) % 255
        return (r, g, b)

    N = boxes_orig.shape[0]
    for i in range(N):
        x1, y1, x2, y2 = boxes_orig[i]
        xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))

        cid = int(class_ids[i]) if i < len(class_ids) else -1
        score = float(scores[i]) if i < len(scores) else None
        name = decoded_names[i] if i < len(decoded_names) else "unknown"

        label = name
        if score is not None:
            try:
                label = f"{name} {score_fmt.format(score)}"
            except Exception:
                label = f"{name} {score:.2f}"

        col = color_for_id(cid) if cid >= 0 else (255, 0, 0)

        # draw rectangle (thickness)
        for t in range(linewidth):
            draw.rectangle([xi1 - t, yi1 - t, xi2 + t, yi2 + t], outline=tuple(col))

        # compute text size and background rectangle
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = xi1
        text_y = yi1 - text_h - 4
        if text_y < 0:
            text_y = yi1 + 4

        rect_bg = [text_x, text_y, text_x + text_w + 4, text_y + text_h + 2]
        draw.rectangle(rect_bg, fill=tuple(col))

        brightness = col[0] * 0.299 + col[1] * 0.587 + col[2] * 0.114
        txt_color = (0, 0, 0) if brightness > 186 else (255, 255, 255)
        draw.text((text_x + 2, text_y + 1), label, fill=txt_color, font=font)

    # save annotated image
    orig_img.save(save_path)
    print(f"Saved annotated image to: {save_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Object detection brick CLI (Python version of the C++ tool)",
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
        "-i",
        dest="image",
        required=True,
        help="Input image path or @sample/... selector",
    )
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
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

    if not args.manifest_path or not args.image:
        parser.print_help()
        sys.exit("ERROR: Missing required arguments (-m, -i).")

    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(args.thread_spec)

    manifest_path = get_manifest_path(args.manifest_path, "object_detection")
    image_path = get_sample_path(args.image, manifest_path)

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

    input["img"] = image_to_npy(image_path)

    # Run the model inference
    output = brick.run(input)

    # Display profiler run
    print(brick.write_profile(), end="")

    boxes = output["box"]
    class_ids = output["id"]
    class_scores = output["score"]
    class_names = output["name"]

    for row in class_names:
        print("Detected:", decode_name_row(row))

    # Generate an image with the bounding boxes
    try:
        # Save into current working directory
        input_filename = os.path.basename(image_path)
        base, _ = os.path.splitext(input_filename)

        out_image = os.path.join(os.getcwd(), f"{base}_detected.png")

        draw_detections_and_save(
            image_path,
            boxes=boxes,
            class_ids=class_ids,
            scores=class_scores,
            class_names_rows=class_names,
            save_path=out_image,
            score_fmt="{:.2f}",
            linewidth=6,
            font_size=20,
        )
    except Exception as e:
        print(f"Warning: failed to draw/save detections: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
