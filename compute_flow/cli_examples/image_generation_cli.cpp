//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <algorithm>
#include <cmath>
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
#include "compute_flow/compute_flow.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace compute_flow::bricks::image_generation {

namespace {

constexpr int32_t k_num_channels = 3;

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-p <prompt> -o <output.png> | -j <runs.json>) "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -p <prompt>            Text describing the image to generate.\n"
                 "  -o <output.png>        Output PNG path.\n"
                 "  -j <runs.json>         JSON file with prompt and output path for each run.\n"
                 "  -t <thread_spec>       Default threads and optional model overrides, e.g. "
                 "4,text_encoder:2 (default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h, --help             Show this help message.\n",
                 executable);
}

void save_image(const TensorData& image, const std::string& path) {
    COMPUTE_FLOW_CHECK(image.format.dt == DataType::F32 && image.format.shape.size() == 3 &&
                       image.format.shape[0] == k_num_channels && image.format.shape[1] > 0 &&
                       image.format.shape[2] > 0 && image.data != nullptr);

    const int32_t height     = static_cast<int32_t>(image.format.shape[1]);
    const int32_t width      = static_cast<int32_t>(image.format.shape[2]);
    const int64_t num_pixels = static_cast<int64_t>(height) * width;
    const auto* planar       = static_cast<const float*>(image.data);
    std::vector<uint8_t> rgb_data(static_cast<size_t>(k_num_channels * num_pixels));

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const int64_t pixel_offset = (static_cast<int64_t>(y) * width) + x;

            for (int32_t channel = 0; channel < k_num_channels; ++channel) {
                const float value = planar[(channel * num_pixels) + pixel_offset];
                rgb_data[static_cast<size_t>((pixel_offset * k_num_channels) + channel)] =
                    static_cast<uint8_t>(std::round(std::clamp(value, 0.0F, 1.0F) * 255.0F));
            }
        }
    }

    COMPUTE_FLOW_CHECK(
        stbi_write_png(path.c_str(), width, height, k_num_channels, rgb_data.data(), width * k_num_channels) != 0);

    std::cout << "Output resolution: " << width << " x " << height << '\n'
              << "Wrote generated image to " << path << std::endl;
}

void run_single(std::string manifest_path, const std::string& prompt, const std::string& output_path,
                const std::string& thread_spec, bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "image_generation");

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
    brick->reset_states();

    TensorDict input;

    // Set the input
    input.declare_tensor("prompt").format = {DataType::String, {static_cast<int64_t>(prompt.size())}};
    input["prompt"].data                  = const_cast<char*>(prompt.data());

    TensorDict output;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }

    // Save the output image
    save_image(output["image"], output_path);

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = internal::get_manifest_path(manifest_path, "image_generation");

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

    const auto run_configs = internal::load_run_configs(runs_file, {"p", "o"}, {"p", "o"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        const std::string prompt      = internal::get_run_value<std::string>(config, "p", "", run_index);
        const std::string output_path = internal::get_run_value<std::string>(config, "o", "", run_index);

        COMPUTE_FLOW_CHECK(!prompt.empty());
        COMPUTE_FLOW_CHECK(!output_path.empty());

        std::cout << "Run " << run_id << std::endl;

        TensorDict input;

        // Set the input
        input.declare_tensor("prompt").format = {DataType::String, {static_cast<int64_t>(prompt.size())}};
        input["prompt"].data                  = const_cast<char*>(prompt.data());

        TensorDict output;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            output = brick->run(input);
        }

        // Save the output image
        save_image(output["image"], output_path);

        // Display profiler run
        brick->write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // Parse the command-line arguments
    std::string manifest_path;
    std::string prompt;
    std::string output_path;
    std::string runs_file;
    std::string thread_spec  = "1";
    bool enable_op_profiling = false;

    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option = 0;
    while ((option = ::getopt_long(argc, argv, "m:p:o:j:t:h", long_options, nullptr)) != -1) {
        switch (option) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'p':
                prompt = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 'j':
                runs_file = optarg;
                break;
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

    // Check the mandatory arguments
    if (manifest_path.empty() || (runs_file.empty() && (prompt.empty() || output_path.empty())) ||
        (!runs_file.empty() && (!prompt.empty() || !output_path.empty()))) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (runs_file.empty()) {
        run_single(manifest_path, prompt, output_path, thread_spec, enable_op_profiling);
    } else {
        run_multiple(manifest_path, runs_file, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::image_generation

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::image_generation::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
