#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Python CLI for the TTS brick with optional audio playback.

Usage:
  python -m cli_examples.tts_cli -m <manifest_path> (-p <prompt> | -j <runs.json>) [-t <thread_spec>] [-o <output_file>] [--playback]
"""

import argparse
import importlib
import os
import sys
import time
import multiprocessing as mp
from multiprocessing.queues import Queue

try:
    import numpy as np
except ImportError:
    print(
        "Error: `numpy` is required. Install dependencies from the repo root with "
        "`python -m pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    sys.exit(1)

import compute_flow_py as cf
from cli_examples.internal.manifest_selector import get_manifest_path
from cli_examples.internal.run_config import get_run_value, load_run_configs
from cli_examples.internal.thread_config import (
    parse_hint_cpu_threads,
    set_cpu_threads,
)


def save_as_wav(
    path: str, left: np.ndarray, right: np.ndarray, sr: int = 44100
) -> None:
    """
    Save float32 stereo WAV using soundfile. Matches your C++ (IEEE float).
    """
    try:
        import soundfile as sf  # pip install soundfile
    except Exception:
        print(
            "Error: `soundfile` is required to save WAVs. Try: pip install soundfile",
            file=sys.stderr,
        )
        raise

    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)

    if left.shape[0] != right.shape[0]:
        raise ValueError("left_channel and right_channel must have the same length")

    stereo = np.stack([left, right], axis=1)  # (T, 2)
    sf.write(path, stereo, sr, subtype="FLOAT")  # 32-bit float WAV


def _tts_process(
    manifest_path: str,
    prompt: str,
    thread_spec: str,
    enable_op_profiling: bool,
    output_file: str | None,
    audio_queue: Queue | None,
    stats_queue: Queue,
) -> None:
    # Parse the number of CPU threads to use
    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)

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

    input = cf.TensorDict()
    input["text_prompt"] = prompt

    left_full = []
    right_full = []
    total_samples = 0
    run_time_s = 0.0
    first_run_time_s = None

    # Run the model inference
    while not brick.is_generation_complete():
        run_start = time.perf_counter()
        output = brick.run(input)
        run_duration_s = time.perf_counter() - run_start
        run_time_s += run_duration_s
        if first_run_time_s is None:
            first_run_time_s = run_duration_s

        left = output["left_channel"]
        right = output["right_channel"]
        sample_rate = int(np.asarray(output["sample_rate"]).reshape(-1)[0])

        if left.size:
            # Copy before queueing to avoid races with reused output buffers.
            left_copy = np.asarray(left, dtype=np.float32).copy()
            right_copy = np.asarray(right, dtype=np.float32).copy()

            if audio_queue is not None:
                audio_queue.put((left_copy, right_copy, sample_rate))

            total_samples += left_copy.size
            left_full.append(left_copy)
            right_full.append(right_copy)

    if audio_queue is not None:
        audio_queue.put(None)

    # Display profiler run
    print(brick.write_profile(), end="")

    if sample_rate:
        audio_len_s = total_samples / float(sample_rate)
    else:
        audio_len_s = 0.0

    output_file = output_file or "output.wav"

    result0 = np.concatenate(left_full) if left_full else np.array([], dtype=np.float32)
    result1 = (
        np.concatenate(right_full) if right_full else np.array([], dtype=np.float32)
    )

    save_as_wav(output_file, result0, result1, sr=sample_rate)
    saved_path = os.path.abspath(output_file)

    stats_queue.put(
        {
            "run_time_s": run_time_s,
            "first_run_time_s": first_run_time_s,
            "audio_len_s": audio_len_s,
            "total_samples": total_samples,
            "sample_rate": sample_rate,
            "saved_path": saved_path,
        }
    )


def _playback_process(audio_queue: Queue, stats_queue: Queue) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        stats_queue.put(
            {
                "stream_write_s": 0.0,
                "time_to_first_chunk_s": None,
                "error": "sounddevice not available",
            }
        )
        return

    stream_write_s = 0.0
    first_item = audio_queue.get()
    if first_item is None:
        stats_queue.put(
            {
                "stream_write_s": stream_write_s,
            }
        )
        return

    left, right, sample_rate = first_item
    if sample_rate is None:
        stats_queue.put(
            {
                "stream_write_s": 0.0,
                "error": "sample_rate missing from tts output",
            }
        )
        return

    with sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        dtype="float32",
        blocksize=0,
    ) as stream:
        write_start = time.perf_counter()
        stream.write(np.column_stack((left, right)))
        stream_write_s += time.perf_counter() - write_start

        while True:
            item = audio_queue.get()
            if item is None:
                break

            left, right, chunk_sample_rate = item
            if chunk_sample_rate != sample_rate:
                stats_queue.put(
                    {
                        "stream_write_s": stream_write_s,
                        "error": "sample_rate changed during playback",
                    }
                )
                return

            write_start = time.perf_counter()
            stream.write(np.column_stack((left, right)))
            stream_write_s += time.perf_counter() - write_start

    stats_queue.put(
        {
            "stream_write_s": stream_write_s,
        }
    )


def run_single(
    manifest_path: str,
    prompt: str,
    output_file: str | None,
    thread_spec: str,
    enable_op_profiling: bool,
    playback: bool,
) -> int:
    manifest_path = get_manifest_path(manifest_path, "tts")

    if playback:
        try:
            importlib.import_module("sounddevice")
        except ImportError:
            print(
                "Error: `sounddevice` is required for audio playback. Install it with "
                "`python -m pip install sounddevice`.",
                file=sys.stderr,
            )
            return 1

    ctx = mp.get_context("spawn")
    audio_queue = ctx.Queue() if playback else None
    tts_stats_queue = ctx.Queue()
    playback_stats_queue = ctx.Queue() if playback else None

    tts = ctx.Process(
        target=_tts_process,
        args=(
            manifest_path,
            prompt,
            thread_spec,
            enable_op_profiling,
            output_file,
            audio_queue,
            tts_stats_queue,
        ),
    )

    if playback:
        player = ctx.Process(
            target=_playback_process,
            args=(audio_queue, playback_stats_queue),
        )
        player.start()
    tts.start()

    tts_stats = tts_stats_queue.get()
    if playback:
        playback_stats = playback_stats_queue.get()

    tts.join()
    if playback:
        player.join()

    print(f"Run time (sum of brick.run): {tts_stats['run_time_s']:.3f}s")
    print(f"Audio length: {tts_stats['audio_len_s']:.3f}s")
    if playback:
        print(f"Blocking stream.write time: {playback_stats['stream_write_s']:.3f}s")
    if tts_stats["audio_len_s"] > 0:
        rtf = tts_stats["run_time_s"] / tts_stats["audio_len_s"]
        print(f"Real-time factor (RTF): {rtf:.3f} ( < 1 means realtime )")
    else:
        print("Real-time factor (RTF): unavailable")

    first_run_time_s = tts_stats.get("first_run_time_s")
    if first_run_time_s is None:
        print("Time to first chunk generated: unavailable")
    else:
        print(f"Time to first chunk generated: {first_run_time_s:.3f}s")

    print(f"Saved: {tts_stats['saved_path']}")
    return 0


def run_multiple(
    manifest_path: str,
    runs_file: str,
    thread_spec: str,
    enable_op_profiling: bool,
    playback: bool,
) -> int:
    if playback:
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "Error: `sounddevice` is required for audio playback. Install it with "
                "`python -m pip install sounddevice`.",
                file=sys.stderr,
            )
            return 1

    default_threads, model_threads = parse_hint_cpu_threads(thread_spec)
    manifest_path = get_manifest_path(manifest_path, "tts")
    manifest = cf.load_manifest(manifest_path)
    # Configure default and per-model CPU thread counts
    set_cpu_threads(manifest, default_threads, model_threads)
    brick = cf.load_brick(manifest)

    options = cf.BrickOptions()
    options.runtime.enable_warmup = True
    options.profiling.enable_op_profiling = enable_op_profiling
    brick.setup(manifest, options)

    run_configs = load_run_configs(runs_file, {"p", "o"}, {"p"})
    for run_index, config in enumerate(run_configs):
        brick.reset_states()

        run_id = get_run_value(config, "id", str(run_index + 1), run_index)
        print(f"Run {run_id}")
        output_file = (
            get_run_value(config, "o", "", run_index) if "o" in config else None
        )
        prompt = get_run_value(config, "p", "", run_index)
        if not prompt:
            raise ValueError(
                f"Run {run_index + 1} argument 'p' must be a non-empty string"
            )

        input = cf.TensorDict()
        input["text_prompt"] = prompt

        left_full = []
        right_full = []
        total_samples = 0
        run_time_s = 0.0
        first_run_time_s = None
        sample_rate = 0

        while not brick.is_generation_complete():
            run_start = time.perf_counter()
            output = brick.run(input)
            run_duration_s = time.perf_counter() - run_start
            run_time_s += run_duration_s
            if first_run_time_s is None:
                first_run_time_s = run_duration_s

            left = np.asarray(output["left_channel"], dtype=np.float32)
            right = np.asarray(output["right_channel"], dtype=np.float32)
            sample_rate = int(np.asarray(output["sample_rate"]).reshape(-1)[0])
            total_samples += left.size

            left_full.append(left.copy())
            right_full.append(right.copy())

        print(brick.write_profile(), end="")

        audio_len_s = total_samples / float(sample_rate) if sample_rate else 0.0
        print(f"Run time (sum of brick.run): {run_time_s:.3f}s")
        print(f"Audio length: {audio_len_s:.3f}s")
        if audio_len_s > 0:
            print(
                f"Real-time factor (RTF): {run_time_s / audio_len_s:.3f} "
                "( < 1 means realtime )"
            )
        else:
            print("Real-time factor (RTF): unavailable")
        if first_run_time_s is None:
            print("Time to first chunk generated: unavailable")
        else:
            print(f"Time to first chunk generated: {first_run_time_s:.3f}s")

        if output_file is not None or playback:
            left_result = (
                np.concatenate(left_full)
                if left_full
                else np.array([], dtype=np.float32)
            )
            right_result = (
                np.concatenate(right_full)
                if right_full
                else np.array([], dtype=np.float32)
            )
            if output_file is not None:
                output_file = output_file or "output.wav"
                save_as_wav(output_file, left_result, right_result, sr=sample_rate)
                print(f"Saved: {os.path.abspath(output_file)}")
            if playback and total_samples:
                with sd.OutputStream(
                    samplerate=sample_rate, channels=2, dtype="float32", blocksize=0
                ) as stream:
                    stream.write(np.column_stack((left_result, right_result)))

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="TTS brick CLI (Python version of the C++ tool)",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-m",
        dest="manifest_path",
        required=True,
        help="Manifest path or @manifest/... selector",
    )
    parser.add_argument("-p", dest="prompt", help="Input prompt text")
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
        "-o",
        dest="output_file",
        default=None,
        help="(Optional) Output audio file name (Default: output.wav)",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="Play generated audio (default: off)",
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

    multiple_runs_mode = args.runs_file is not None
    if (not multiple_runs_mode and not args.prompt) or (
        multiple_runs_mode and (args.prompt is not None or args.output_file is not None)
    ):
        parser.error("use either -p [-o] or -j")

    if multiple_runs_mode:
        try:
            return run_multiple(
                args.manifest_path,
                args.runs_file,
                args.thread_spec,
                args.enable_op_profiling,
                args.playback,
            )
        except ValueError as error:
            parser.error(str(error))

    return run_single(
        args.manifest_path,
        args.prompt,
        args.output_file,
        args.thread_spec,
        args.enable_op_profiling,
        args.playback,
    )


if __name__ == "__main__":
    sys.exit(main())
