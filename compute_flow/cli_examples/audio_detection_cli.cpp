//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::audio_detection {
namespace {

void print_usage(const char* exec) {
    fprintf(stderr,
            "Usage: %s -m <manifest_path> (-a <audio.wav> | -j <runs.json>) [-t <thread_spec>] [-k <top_k>]\n\n"
            "Options:\n"
            "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
            "  -a <audio.wav>            Input WAV path or @sample/... selector.\n"
            "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
            "  -t <thread_spec>          Default threads and optional per-model overrides, e.g. 2,classifier:4 "
            "(default: 1).\n"
            "  -k <top_k>                Number of highest-probability classes to display (default: 3).\n"
            "  --enable-op-profiling     Enable per-op profiling output.\n"
            "  -h                        Show this help message and exit.\n",
            exec);
}

void run_single(std::string manifest_path, const std::string& audio_path, int64_t top_k, const std::string& thread_spec,
                bool enable_op_profiling) {
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = compute_flow::bricks::internal::get_manifest_path(manifest_path, "audio_detection");

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    COMPUTE_FLOW_CHECK(top_k > 0);
    const std::string resolved_audio_path = internal::get_sample_path(audio_path, manifest_path);
    std::vector<float> mono_raw_audio;

    // Read audio file
    const int64_t sample_rate = internal::read_mono_wav(resolved_audio_path, mono_raw_audio);

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

    while (!brick->is_generation_complete()) {
        // Run the model inference
        TensorDict output            = brick->run(input);
        const int64_t num_frames     = output["score"].format.shape[0];
        const int64_t num_classes    = output["score"].format.shape[1];
        const int64_t max_name_bytes = output["name"].format.shape[2];
        COMPUTE_FLOW_CHECK(top_k <= num_classes);

        const float* scores  = reinterpret_cast<float*>(output["score"].data);
        const int64_t* ids   = reinterpret_cast<int64_t*>(output["id"].data);
        const uint8_t* names = reinterpret_cast<uint8_t*>(output["name"].data);
        const float* frame   = reinterpret_cast<float*>(output["frame"].data);

        for (int64_t frame_index = 0; frame_index < num_frames; ++frame_index) {
            const float* frame_scores  = scores + frame_index * num_classes;
            const int64_t* frame_ids   = ids + frame_index * num_classes;
            const uint8_t* frame_names = names + frame_index * num_classes * max_name_bytes;
            const float* frame_bounds  = frame + frame_index * 2;

            printf("Timestamp [%.3f:%.3f]s: \n", frame_bounds[0], frame_bounds[1]);

            for (int64_t index = 0; index < top_k; ++index) {
                const std::string class_name(reinterpret_cast<const char*>(frame_names) + index * max_name_bytes);
                printf("    (%s (%lld), %f)\n", class_name.c_str(), static_cast<long long>(frame_ids[index]),
                       frame_scores[index]);
            }
        }
    }

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = compute_flow::bricks::internal::get_manifest_path(manifest_path, "audio_detection");

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    const auto run_configs = internal::load_run_configs(runs_file, {"a", "k"}, {"a"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string audio_path = internal::get_run_value<std::string>(config, "a", "", run_index);
        const int64_t top_k          = internal::get_run_value<int64_t>(config, "k", 3, run_index);
        COMPUTE_FLOW_CHECK(!audio_path.empty());
        COMPUTE_FLOW_CHECK(audio_path.rfind("@sample", 0) != 0);
        COMPUTE_FLOW_CHECK(top_k > 0);

        std::vector<float> mono_raw_audio;

        // Read audio file
        const int64_t sample_rate = internal::read_mono_wav(audio_path, mono_raw_audio);

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

        while (!brick->is_generation_complete()) {
            // Run the model inference
            TensorDict output            = brick->run(input);
            const int64_t num_frames     = output["score"].format.shape[0];
            const int64_t num_classes    = output["score"].format.shape[1];
            const int64_t max_name_bytes = output["name"].format.shape[2];
            COMPUTE_FLOW_CHECK(top_k <= num_classes);

            const float* scores  = reinterpret_cast<float*>(output["score"].data);
            const int64_t* ids   = reinterpret_cast<int64_t*>(output["id"].data);
            const uint8_t* names = reinterpret_cast<uint8_t*>(output["name"].data);
            const float* frame   = reinterpret_cast<float*>(output["frame"].data);

            for (int64_t frame_index = 0; frame_index < num_frames; ++frame_index) {
                const float* frame_scores  = scores + frame_index * num_classes;
                const int64_t* frame_ids   = ids + frame_index * num_classes;
                const uint8_t* frame_names = names + frame_index * num_classes * max_name_bytes;
                const float* frame_bounds  = frame + frame_index * 2;

                printf("Timestamp [%.3f:%.3f]s: \n", frame_bounds[0], frame_bounds[1]);

                for (int64_t index = 0; index < top_k; ++index) {
                    const std::string class_name(reinterpret_cast<const char*>(frame_names) + index * max_name_bytes);
                    printf("    (%s (%lld), %f)\n", class_name.c_str(), static_cast<long long>(frame_ids[index]),
                           frame_scores[index]);
                }
            }
        }

        // Display profiler run
        brick->write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path;
    std::string audio_path;
    std::string thread_spec = "1";
    int64_t top_k           = 3;
    std::string json_file;
    bool has_top_k_argument  = false;
    bool enable_op_profiling = false;

    static struct option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int opt = 0;
    while ((opt = ::getopt_long(argc, argv, "m:a:t:k:j:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'a':
                audio_path = optarg;
                break;
            case 't':
                thread_spec = optarg;
                break;
            case 'k':
                top_k              = std::stoll(optarg);
                has_top_k_argument = true;
                break;
            case 'j':
                json_file = optarg;
                break;
            case 1001:
                enable_op_profiling = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || (json_file.empty() && audio_path.empty()) ||
        (!json_file.empty() && (!audio_path.empty() || has_top_k_argument))) {
        fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    COMPUTE_FLOW_CHECK(top_k > 0);

    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(manifest_path, json_file, thread_spec, enable_op_profiling);
    } else {
        run_single(manifest_path, audio_path, top_k, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::audio_detection

int main(int argc, char** argv) { return compute_flow::bricks::audio_detection::run_cli(argc, argv); }
