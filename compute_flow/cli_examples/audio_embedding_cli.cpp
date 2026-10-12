//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <algorithm>
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

namespace compute_flow::bricks::audio_embedding {
namespace {

constexpr int64_t k_num_embedding_edge_values = 3;

void print_usage(const char* name) {
    std::fprintf(
        stderr,
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

void print_embedding(const float* embedding, int64_t embedding_size) {
    COMPUTE_FLOW_ASSERT(embedding != nullptr);
    COMPUTE_FLOW_ASSERT(embedding_size > 0);

    std::printf("Embedding: [");

    const int64_t num_leading_values = std::min(embedding_size, k_num_embedding_edge_values);

    for (int64_t value_index = 0; value_index < num_leading_values; ++value_index) {
        if (value_index > 0) {
            std::printf(", ");
        }

        std::printf("%.2f", embedding[value_index]);
    }

    const bool is_truncated = embedding_size > 2 * k_num_embedding_edge_values;
    const int64_t trailing_start = is_truncated ? embedding_size - k_num_embedding_edge_values
                                                 : num_leading_values;

    if (is_truncated) {
        std::printf(", ...");
    }

    for (int64_t value_index = trailing_start; value_index < embedding_size; ++value_index) {
        std::printf(", %.2f", embedding[value_index]);
    }

    std::printf("]\n");
}

void run_single(std::string manifest_path, const std::string& audio_value, const std::string& thread_spec,
                bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "audio_embedding");

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

    // Run the model inference
    while (!brick->is_generation_complete()) {
        TensorDict output = brick->run(input);

        const int64_t num_embeddings = output["embedding"].format.shape[0];
        const int64_t embedding_size = output["embedding"].format.shape[1];
        const auto* embeddings       = reinterpret_cast<const float*>(output["embedding"].data);
        const auto* frames           = reinterpret_cast<const float*>(output["frame"].data);

        for (int64_t embedding_index = 0; embedding_index < num_embeddings; ++embedding_index) {
            const int64_t embedding_offset = embedding_index * embedding_size;

            std::printf("Timestamp [%.3f:%.3f]s:\n", frames[embedding_index * 2],
                        frames[embedding_index * 2 + 1]);

            print_embedding(embeddings + embedding_offset, embedding_size);
        }
    }

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "audio_embedding");

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

        // Run the model inference
        while (!brick->is_generation_complete()) {
            TensorDict output = brick->run(input);

            const int64_t num_embeddings = output["embedding"].format.shape[0];
            const int64_t embedding_size = output["embedding"].format.shape[1];
            const auto* embeddings       = reinterpret_cast<const float*>(output["embedding"].data);
            const auto* frames           = reinterpret_cast<const float*>(output["frame"].data);

            for (int64_t embedding_index = 0; embedding_index < num_embeddings; ++embedding_index) {
                const int64_t embedding_offset = embedding_index * embedding_size;

                std::printf("Timestamp [%.3f:%.3f]s:\n", frames[embedding_index * 2],
                            frames[embedding_index * 2 + 1]);

                print_embedding(embeddings + embedding_offset, embedding_size);
            }
        }

        // Display profiler run
        brick->write_profile(std::cout);
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
}  // namespace compute_flow::bricks::audio_embedding

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::audio_embedding::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
