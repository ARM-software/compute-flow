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
import time

import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def write_generated_text(
    brick,
    text_prompt: str,
    max_generated_tokens: int,
    output_file: str | None,
) -> tuple[float, float, int]:
    generated_text_stream = sys.stdout
    if output_file:
        try:
            generated_text_stream = open(output_file, "w", encoding="utf-8", newline="")
        except OSError as error:
            raise ValueError(f"Failed to open output file: {error}") from error

    # Set the input
    input_data = {"text_prompt": text_prompt}
    pending_text = ""
    num_generated_tokens = 0
    prefill_time_s = 0.0
    decode_time_s = 0.0
    num_decode_tokens = 0
    sys.stdout.write(f"Input prompt: {text_prompt}\n")
    if output_file:
        sys.stdout.write(f"Output file: {output_file}\n")
    else:
        sys.stdout.write("Output text: ")

    # Run the model inference
    while (
        not brick.is_generation_complete()
        and num_generated_tokens < max_generated_tokens
    ):
        run_start = time.perf_counter()
        output = brick.run(input_data)
        run_duration_s = time.perf_counter() - run_start
        # The first call includes prompt prefill; subsequent calls decode one token each.
        if num_generated_tokens == 0:
            prefill_time_s = run_duration_s
        else:
            decode_time_s += run_duration_s
            num_decode_tokens += 1
        chunk = output["text"]
        if chunk:
            pending_text += chunk
            last_whitespace = max(
                pending_text.rfind(" "),
                pending_text.rfind("\t"),
                pending_text.rfind("\r"),
                pending_text.rfind("\n"),
            )
            if last_whitespace != -1:
                generated_text_stream.write(pending_text[: last_whitespace + 1])
                generated_text_stream.flush()
                pending_text = pending_text[last_whitespace + 1 :]
        num_generated_tokens += 1

    # Display the left-over text
    if pending_text:
        generated_text_stream.write(pending_text)
    generated_text_stream.flush()
    if output_file:
        generated_text_stream.close()
    else:
        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

    return prefill_time_s, decode_time_s, num_decode_tokens


def run_single(
    manifest_path: str,
    text_prompt: str,
    max_generated_tokens: int,
    output_file: str | None,
    thread_spec: str,
    enable_op_profiling: bool,
) -> tuple[float, float, int]:
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "lm_text")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)
    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    decode_stats = write_generated_text(
        brick, text_prompt, max_generated_tokens, output_file
    )

    # Display profiler run
    print(brick.write_profile(), end="")

    return decode_stats


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
) -> tuple[float, float, int]:
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "lm_text")

    # Load the manifest
    manifest = cf.load_manifest(manifest_path)
    set_cpu_threads(manifest, default_threads, model_threads)

    # Load the brick
    brick = cf.load_brick(manifest)

    # Initialize the brick
    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling

    # Setup the brick
    brick.setup(manifest, options)

    run_configs = load_run_configs(runs_file, {"p", "n", "o"}, {"p"})
    total_prefill_time_s = 0.0
    total_decode_time_s = 0.0
    total_num_decode_tokens = 0
    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        text_prompt = get_run_value(config, "p", "", run_index)
        max_generated_tokens = get_run_value(config, "n", 2**63 - 1, run_index)
        output_file = get_run_value(config, "o", "", run_index)
        if not text_prompt:
            raise ValueError(
                f"Run {run_index + 1} argument 'p' must be a non-empty string"
            )

        prefill_time_s, decode_time_s, num_decode_tokens = write_generated_text(
            brick,
            text_prompt,
            max_generated_tokens,
            output_file or None,
        )
        total_prefill_time_s += prefill_time_s
        total_decode_time_s += decode_time_s
        total_num_decode_tokens += num_decode_tokens

        # Display profiler run
        print(brick.write_profile(), end="")

    return total_prefill_time_s, total_decode_time_s, total_num_decode_tokens


def main(argv=None) -> int:
    # Parse the command-line arguments
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
    parser.add_argument("-p", dest="text_prompt", help="Text prompt")
    parser.add_argument(
        "-j",
        dest="runs_file",
        help="Run multiple entries without reloading the model",
    )
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
        default=None,
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

    # Check the mandatory arguments
    direct_arguments_present = any(
        value is not None
        for value in (
            args.text_prompt,
            args.max_generated_tokens,
            args.output_file,
        )
    )
    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and args.text_prompt is None) or (
        multiple_runs_mode and direct_arguments_present
    ):
        parser.error("use either direct run arguments or -j")

    try:
        if multiple_runs_mode:
            prefill_time_s, decode_time_s, num_decode_tokens = run_multiple(
                args.manifest_path,
                args.runs_file,
                args.thread_spec,
                args.enable_op_profiling,
            )
        else:
            prefill_time_s, decode_time_s, num_decode_tokens = run_single(
                args.manifest_path,
                args.text_prompt,
                args.max_generated_tokens
                if args.max_generated_tokens is not None
                else 2**63 - 1,
                args.output_file,
                args.thread_spec,
                args.enable_op_profiling,
            )

        if prefill_time_s > 0.0:
            print(f"Prompt prefill time: {prefill_time_s:.3f}s")
        else:
            print("Prompt prefill time: unavailable")

        if decode_time_s > 0.0:
            print(f"Decode tokens per second: {num_decode_tokens / decode_time_s:.3f}")
        else:
            print("Decode tokens per second: unavailable")
    except ValueError as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
