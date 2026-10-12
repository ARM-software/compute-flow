//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::image_classification {
namespace {

constexpr int64_t k_num_channels = 3;
constexpr int64_t k_default_top_k = 5;

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-i <input_image> [-k <top_k>] | -j <runs.json>) "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -i <input_image>       Image path or @sample/... selector.\n"
                 "  -j <runs.json>         Input JSON for multiple runs; each entry needs id and i, and may set k.\n"
                 "  -k <top_k>             Maximum number of classes to display (default: 5).\n"
                 "  -t <thread_spec>       Default threads and optional model overrides (default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n",
                 executable);
}

void display_result(const TensorDict& output, int64_t top_k) {
    const TensorData& ids = output["id"];
    const TensorData& scores = output["score"];
    const TensorData& names = output["name"];

    COMPUTE_FLOW_CHECK(ids.data != nullptr && ids.format.dt == DataType::I64 && ids.format.shape.size() == 1 &&
                       ids.format.shape[0] > 0);
    COMPUTE_FLOW_CHECK(scores.data != nullptr && scores.format.dt == DataType::F32 &&
                       scores.format.shape == ids.format.shape);
    COMPUTE_FLOW_CHECK(names.data != nullptr && names.format.dt == DataType::U8 && names.format.shape.size() == 2 &&
                       names.format.shape[0] == ids.format.shape[0] && names.format.shape[1] > 0);

    const int64_t num_classes = ids.format.shape[0];
    const int64_t max_name_length = names.format.shape[1];
    const int64_t num_classes_to_display = std::min(top_k, num_classes);
    const int64_t* class_ids = static_cast<const int64_t*>(ids.data);
    const float* class_scores = static_cast<const float*>(scores.data);
    const char* class_names = static_cast<const char*>(names.data);

    for (int64_t index = 0; index < num_classes_to_display; ++index) {
        const char* name_start = class_names + index * max_name_length;
        const char* name_end = std::find(name_start, name_start + max_name_length, '\0');
        const std::string name(name_start, name_end);

        std::printf("%lld. %s (id %lld, score %.4f)\n", static_cast<long long>(index + 1), name.c_str(),
                    static_cast<long long>(class_ids[index]), class_scores[index]);
    }
}

void run_single(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                const std::string& image_value, int64_t top_k) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path = internal::get_manifest_path(manifest_path, "image_classification");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);
    brick->reset_states();

    const std::string image_path = internal::get_sample_path(image_value, manifest_path);
    int32_t image_width = 0;
    int32_t image_height = 0;

    // Load the image
    std::vector<float> image =
        internal::load_rgb_image_chw(image_path, image_width, image_height, internal::ImageDecodeMode::NormalizedU8);

    TensorDict input;

    // Set the input
    input.declare_tensor("image").format = {DataType::F32, {k_num_channels, image_height, image_width}};
    input["image"].data = image.data();

    TensorDict output;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }

    display_result(output, top_k);

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                  const std::string& runs_file) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path = internal::get_manifest_path(manifest_path, "image_classification");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    const auto run_configs = internal::load_run_configs(runs_file, {"i", "k"}, {"i"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<std::size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string image_path = internal::get_run_value<std::string>(config, "i", "", run_index);
        const int64_t top_k = internal::get_run_value<int64_t>(config, "k", k_default_top_k, run_index);

        COMPUTE_FLOW_CHECK(!image_path.empty());
        COMPUTE_FLOW_CHECK(top_k > 0);

        int32_t image_width = 0;
        int32_t image_height = 0;

        // Load the image
        std::vector<float> image =
            internal::load_rgb_image_chw(image_path, image_width, image_height, internal::ImageDecodeMode::NormalizedU8);

        TensorDict input;

        // Set the input
        input.declare_tensor("image").format = {DataType::F32, {k_num_channels, image_height, image_width}};
        input["image"].data = image.data();

        TensorDict output;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            output = brick->run(input);
        }

        display_result(output, top_k);

        // Display profiler run
        brick->write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string image_path;
    std::string json_file;
    std::string thread_spec = "1";
    int64_t top_k = k_default_top_k;
    bool has_top_k_argument = false;
    bool enable_op_profiling = false;

    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option_value = 0;
    while ((option_value = ::getopt_long(argc, argv, "m:i:j:k:t:h", long_options, nullptr)) != -1) {
        switch (option_value) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'i':
                image_path = optarg;
                break;
            case 'j':
                json_file = optarg;
                break;
            case 'k': {
                const char* end = optarg + std::char_traits<char>::length(optarg);
                const auto result = std::from_chars(optarg, end, top_k);

                if (result.ec != std::errc{} || result.ptr != end || top_k <= 0) {
                    std::fprintf(stderr, "-k must be a positive integer.\n");
                    return EXIT_FAILURE;
                }

                has_top_k_argument = true;
                break;
            }
            case 't':
                thread_spec = optarg;
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

    if (manifest_path.empty() || (json_file.empty() && image_path.empty()) ||
        (!json_file.empty() && (!image_path.empty() || has_top_k_argument))) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!json_file.empty()) {
        run_multiple(manifest_path, thread_spec, enable_op_profiling, json_file);
    } else {
        run_single(manifest_path, thread_spec, enable_op_profiling, image_path, top_k);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::image_classification

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::image_classification::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
