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

#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::object_detection {

namespace {

constexpr int32_t k_num_channels = 3;

void print_usage(const char* name) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest_path> (-i <image> | -j <runs.json>) [-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
                 "  -i <image>                Input image path or @sample/... selector\n"
                 "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
                 "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally "
                 "override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
                 "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
                 "  -h                        Show this help message\n",
                 name);
}

void run_single(std::string manifest_path, const std::string& image_value, const std::string& thread_spec,
                bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "object_detection");

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

    const std::string image_path = internal::get_sample_path(image_value, manifest_path);
    int32_t image_width          = 0;
    int32_t image_height         = 0;

    // Load the image
    auto image =
        internal::load_rgb_image_chw(image_path, image_width, image_height, internal::ImageDecodeMode::StbFloat);

    TensorDict input;

    // Set the input
    input.declare_tensor("img");
    input["img"].data         = image.data();
    input["img"].format.dt    = DataType::F32;
    input["img"].format.shape = {1, k_num_channels, image_height, image_width};

    TensorDict output;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }

    const auto class_names       = output["name"];
    const int64_t num_classes    = class_names.format.shape[1];
    const int64_t max_name_bytes = class_names.format.shape[2];
    const char* names            = reinterpret_cast<const char*>(class_names.data);

    for (int64_t index = 0; index < class_names.format.shape[0]; ++index) {
        const char* name       = names + index * num_classes * max_name_bytes;
        const char* name_end   = name + max_name_bytes;
        const char* terminator = std::find(name, name_end, '\0');
        const std::string class_name(name, terminator);
        std::printf("Detected: %s\n", class_name.c_str());
    }

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "object_detection");

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

    const auto run_configs = internal::load_run_configs(runs_file, {"i"}, {"i"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string image_path = internal::get_run_value<std::string>(config, "i", "", run_index);

        COMPUTE_FLOW_CHECK(!image_path.empty());

        int32_t image_width  = 0;
        int32_t image_height = 0;

        // Load the image
        auto image =
            internal::load_rgb_image_chw(image_path, image_width, image_height, internal::ImageDecodeMode::StbFloat);

        TensorDict input;

        // Set the input
        input.declare_tensor("img");
        input["img"].data         = image.data();
        input["img"].format.dt    = DataType::F32;
        input["img"].format.shape = {1, k_num_channels, image_height, image_width};

        TensorDict output;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            output = brick->run(input);
        }

        const auto class_names       = output["name"];
        const int64_t num_classes    = class_names.format.shape[1];
        const int64_t max_name_bytes = class_names.format.shape[2];
        const char* names            = reinterpret_cast<const char*>(class_names.data);

        for (int64_t index = 0; index < class_names.format.shape[0]; ++index) {
            const char* name       = names + index * num_classes * max_name_bytes;
            const char* name_end   = name + max_name_bytes;
            const char* terminator = std::find(name, name_end, '\0');
            const std::string class_name(name, terminator);
            std::printf("Detected: %s\n", class_name.c_str());
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
    std::string manifest_path = "";
    std::string input_path    = "";
    std::string thread_spec   = "1";
    std::string json_file;
    bool enable_op_profiling            = false;
    static struct option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int32_t option_value = 0;

    while ((option_value = ::getopt_long(argc, argv, "m:i:t:j:h", long_options, nullptr)) != -1) {
        switch (option_value) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'i':
                input_path = optarg;
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

    // Check the mandatory arguments
    if (manifest_path.empty() || (json_file.empty() && input_path.empty()) ||
        (!json_file.empty() && !input_path.empty())) {
        std::fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(manifest_path, json_file, thread_spec, enable_op_profiling);
    } else {
        run_single(manifest_path, input_path, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::object_detection

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::object_detection::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
