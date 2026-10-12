//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::vad {
namespace {

void print_usage(const char* name) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest_path> (-a <audio.wav> | -j <runs.json>) [-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
                 "  -a <audio.wav>            Input audio path or @sample/... selector.\n"
                 "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
                 "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally "
                 "override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
                 "  --enable-op-profiling     Enable per-op profiling (default: off).\n"
                 "  -h                        Show this help message and exit.\n",
                 name);
}

void run_single(std::string manifest_path, const std::string& audio_value, const std::string& thread_spec,
                bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "vad");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    brick->reset_states();

    const std::string audio_file = internal::get_sample_path(audio_value, manifest_path);
    std::vector<float> mono_raw_audio;

    // Read audio file
    const int64_t sample_rate = internal::read_mono_wav(audio_file, mono_raw_audio);

    TensorDict input;

    // Set the input
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");
    input["raw_wav"].data             = mono_raw_audio.data();
    input["raw_wav"].format.dt        = DataType::F32;
    input["raw_wav"].format.shape     = {static_cast<int64_t>(mono_raw_audio.size())};
    input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    TensorDict output;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }

    // Display profiler run
    brick->write_profile(std::cout);

    const int64_t num_frames = output["score"].format.shape[0];
    const auto* scores       = reinterpret_cast<const float*>(output["score"].data);
    const auto* frames       = reinterpret_cast<const float*>(output["frame"].data);

    for (int64_t frame_index = 0; frame_index < num_frames; ++frame_index) {
        const char* result = scores[frame_index] > 0.5F ? "Speech" : "No Speech";

        std::printf("Timestamp [%.3f:%.3f]s: %s, (prob: %f)\n", frames[frame_index * 2],
                    frames[frame_index * 2 + 1], result, scores[frame_index]);
    }
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "vad");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    const auto run_configs = internal::load_run_configs(runs_file, {"a"}, {"a"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string audio_file = internal::get_run_value<std::string>(config, "a", "", run_index);

        COMPUTE_FLOW_CHECK(!audio_file.empty());

        std::vector<float> mono_raw_audio;

        // Read audio file
        const int64_t sample_rate = internal::read_mono_wav(audio_file, mono_raw_audio);

        TensorDict input;

        // Set the input
        input.declare_tensor("raw_wav");
        input.declare_tensor("sample_rate");
        input["raw_wav"].data             = mono_raw_audio.data();
        input["raw_wav"].format.dt        = DataType::F32;
        input["raw_wav"].format.shape     = {static_cast<int64_t>(mono_raw_audio.size())};
        input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
        input["sample_rate"].format.dt    = DataType::I64;
        input["sample_rate"].format.shape = {1};

        TensorDict output;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            output = brick->run(input);
        }

        // Display profiler run
        brick->write_profile(std::cout);

        const int64_t num_frames = output["score"].format.shape[0];
        const auto* scores       = reinterpret_cast<const float*>(output["score"].data);
        const auto* frames       = reinterpret_cast<const float*>(output["frame"].data);

        for (int64_t frame_index = 0; frame_index < num_frames; ++frame_index) {
            const char* result = scores[frame_index] > 0.5F ? "Speech" : "No Speech";

            std::printf("Timestamp [%.3f:%.3f]s: %s, (prob: %f)\n", frames[frame_index * 2],
                        frames[frame_index * 2 + 1], result, scores[frame_index]);
        }
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string audio_path;
    std::string thread_spec = "1";
    std::string json_file;
    bool enable_op_profiling            = false;
    static struct option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int32_t option_value = 0;

    while ((option_value = ::getopt_long(argc, argv, "m:a:t:j:h", long_options, nullptr)) != -1) {
        switch (option_value) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'a':
                audio_path = optarg;
                break;
            case 't':
                thread_spec = optarg;
                break;
            case 'j':
                json_file = optarg;
                break;
            case 1001:
                enable_op_profiling = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (manifest_path.empty() || (json_file.empty() && audio_path.empty()) ||
        (!json_file.empty() && !audio_path.empty())) {
        std::fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(manifest_path, json_file, thread_spec, enable_op_profiling);
    } else {
        run_single(manifest_path, audio_path, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::vad

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::vad::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
