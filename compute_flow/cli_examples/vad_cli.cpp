//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"

#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace compute_flow::bricks::vad {

void print_usage(const char* exec) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -a <audio.wav> [-t <thread_spec>] [-o <output.json>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -a <audio.wav>            Input audio path or @sample/... selector.\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -o <output.json>          Optional path to dump JSON summary (probabilities only).\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off).\n"
        "  -h                        Show this help message and exit.\n",
        exec);
}

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string audio_path;
    std::string thread_spec = "1";
    bool enable_op_profiling = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt = 0;
    while ((opt = ::getopt_long(argc, argv, "m:a:t:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path = optarg; break;
            case 'a': audio_path    = optarg; break;
            case 't': thread_spec = optarg; break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (manifest_path.empty() || audio_path.empty()) {
        fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "vad");
    audio_path = compute_flow::bricks::internal::get_sample_path(audio_path, manifest_path);

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    TensorDict input;

    std::vector<float> mono_raw_audio;

    // Read audio file
    const int64_t sample_rate = internal::read_mono_wav(audio_path, mono_raw_audio);
    const int64_t num_samples = static_cast<int64_t>(mono_raw_audio.size());

    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");
    input["raw_wav"].data         = mono_raw_audio.data();
    input["raw_wav"].format.dt    = DataType::F32;
    input["raw_wav"].format.shape = {num_samples};
    input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    // Run the model inference
    TensorDict output = brick->run(input);

    // Display profiler run
    brick->write_profile(std::cout);

    const int64_t num_frames = output["score"].format.shape[0];
    const auto* scores = reinterpret_cast<const float*>(output["score"].data);
    const auto* frames = reinterpret_cast<const float*>(output["frame"].data);

    for(int64_t i = 0; i < num_frames; ++i) {
        const char* result = scores[i] > 0.5F ? "Speech" : "No Speech";
        printf(
            "Timestamp [%.3f:%.3f]s: %s, (prob: %f)\n",
            frames[i * 2],
            frames[i * 2 + 1],
            result,
            scores[i]);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::vad

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::vad::run_cli(argc, argv);
}
