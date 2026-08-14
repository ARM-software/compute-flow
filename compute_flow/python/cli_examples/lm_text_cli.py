#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python CLI for the lm_text brick — mirrors the C++ tool one-to-one.

Usage:
  python -m cli_examples.lm_text_cli -m <manifest_path> -p <prompt> [-t <thread_spec>] [-o <output_file>]
"""

import argparse
import sys

import numpy as np

import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LM text only brick CLI (Python version of the C++ tool)",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument("-p", dest="text_prompt", required=True, help="Text prompt")
    parser.add_argument(
        "-t",
        dest="thread_spec",
        default="1",
        help="Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)",
    )
    parser.add_argument(
        "-n",
        dest="max_generated_tokens",
        type=int,
        default=(2**63 - 1),
        help="Maximum number of generated tokens (default=max int64)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        help="Write generated text to this file instead of stdout",
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

    generated_text_stream = sys.stdout
    if args.output_file:
        try:
            generated_text_stream = open(
                args.output_file, "w", encoding="utf-8", newline=""
            )
        except OSError as error:
            parser.error(f"Failed to open output file: {error}")

    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(args.thread_spec)

    manifest_path = get_manifest_path(args.manifest_path, "lm_text")

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

    input_data = {"text_prompt": args.text_prompt}

    pending_text = ""

    num_generated_tokens = 0

    sys.stdout.write("\n")
    sys.stdout.write("Input prompt: " + args.text_prompt + "\n")
    if args.output_file:
        sys.stdout.write("Output file: " + args.output_file + "\n")
    else:
        sys.stdout.write("Output text: ")

    while (
        not brick.is_generation_complete()
        and num_generated_tokens < args.max_generated_tokens
    ):
        output = brick.run(input_data)

        chunk = output["text"]

        if chunk:
            pending_text += chunk

            last_ws = max(
                pending_text.rfind(" "),
                pending_text.rfind("\t"),
                pending_text.rfind("\r"),
                pending_text.rfind("\n"),
            )
            if last_ws != -1:
                to_print = pending_text[: last_ws + 1]
                generated_text_stream.write(to_print)
                generated_text_stream.flush()
                pending_text = pending_text[last_ws + 1 :]

        num_generated_tokens += 1

    # Display the left-over text
    if pending_text:
        generated_text_stream.write(pending_text)
    generated_text_stream.flush()
    if args.output_file:
        generated_text_stream.close()
    else:
        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Display profiler run
    print(brick.write_profile(), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
