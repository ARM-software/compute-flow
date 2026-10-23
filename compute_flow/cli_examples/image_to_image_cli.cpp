//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace compute_flow::bricks::image_to_image {

namespace {

constexpr int32_t k_num_channels       = 3;
constexpr int32_t k_dimension_multiple = 16;

int32_t round_up(int32_t value, int32_t multiple) { return ((value + multiple - 1) / multiple) * multiple; }

std::vector<float> pad_image_chw(const std::vector<float>& image, int32_t width, int32_t height, int32_t padded_width,
                                 int32_t padded_height) {
    const int64_t padded_num_pixels = static_cast<int64_t>(padded_width) * padded_height;
    std::vector<float> padded_image(static_cast<size_t>(k_num_channels * padded_num_pixels), 0.0F);
    const int64_t num_pixels = static_cast<int64_t>(width) * height;

    for (int32_t channel = 0; channel < k_num_channels; ++channel) {
        for (int32_t y = 0; y < height; ++y) {
            const int64_t source_offset =
                (static_cast<int64_t>(channel) * num_pixels) + (static_cast<int64_t>(y) * width);
            const int64_t destination_offset =
                (static_cast<int64_t>(channel) * padded_num_pixels) + (static_cast<int64_t>(y) * padded_width);

            std::copy_n(image.data() + source_offset, width, padded_image.data() + destination_offset);
        }
    }

    return padded_image;
}

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-i <input_image> -p <prompt> | -j <runs.json>) [-o <output.png>] "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -i <input_image>       Image path or @sample/... selector.\n"
                 "  -p <prompt>            Natural-language editing instruction.\n"
                 "  -j <runs.json>         Input JSON file path for multiple run settings.\n"
                 "  -o <output.png>        Output PNG path (default: <input>_edited.png).\n"
                 "  -t <thread_spec>       Default threads and optional model overrides, e.g. "
                 "4,text_encoder:2 (default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n",
                 executable);
}

std::string default_output_path(const std::string& input_path) {
    const size_t dot = input_path.find_last_of('.');
    if (dot == std::string::npos) {
        return input_path + "_edited.png";
    }

    return input_path.substr(0, dot) + "_edited.png";
}

void run_single(IBrickV1& brick, const std::string& manifest_path, const std::string& input_value,
                const std::string& prompt, const std::optional<std::string>& output_path) {
    const std::string input_path = internal::get_sample_path(input_value, manifest_path);
    int32_t width                = 0;
    int32_t height               = 0;

    // Load the image
    std::vector<float> image =
        internal::load_rgb_image_chw(input_path, width, height, internal::ImageDecodeMode::NormalizedU8);
    const int32_t source_width  = width;
    const int32_t source_height = height;
    const int32_t padded_width  = round_up(width, k_dimension_multiple);
    const int32_t padded_height = round_up(height, k_dimension_multiple);

    if (padded_width != width || padded_height != height) {
        image  = pad_image_chw(image, width, height, padded_width, padded_height);
        width  = padded_width;
        height = padded_height;
    }

    TensorDict input;

    // Set the input
    input.declare_tensor("image").format  = {DataType::F32, {k_num_channels, height, width}};
    input["image"].data                   = image.data();
    input.declare_tensor("prompt").format = {DataType::String, {static_cast<int64_t>(prompt.size())}};
    input["prompt"].data                  = const_cast<char*>(prompt.data());

    TensorDict output;

    // Run the model inference
    while (!brick.is_generation_complete()) {
        output = brick.run(input);
    }

    const TensorData& output_image = output["image"];

    COMPUTE_FLOW_CHECK(output_image.format.dt == DataType::F32 && output_image.format.shape.size() == 3 &&
                       output_image.format.shape[0] == k_num_channels);

    const int32_t output_height = static_cast<int32_t>(output_image.format.shape[1]);
    const int32_t output_width  = static_cast<int32_t>(output_image.format.shape[2]);
    std::cout << "Original input resolution: " << source_width << " x " << source_height << '\n'
              << "Output resolution: " << output_width << " x " << output_height << '\n';

    // Display profiler run
    brick.write_profile(std::cout);

    if (!output_path.has_value()) {
        return;
    }

    // Save the output image
    const int64_t num_output_pixels = static_cast<int64_t>(output_height) * output_width;
    const auto* output_planar       = static_cast<const float*>(output_image.data);
    std::vector<uint8_t> output_interleaved(static_cast<size_t>(k_num_channels * num_output_pixels));

    for (int32_t y = 0; y < output_height; ++y) {
        for (int32_t x = 0; x < output_width; ++x) {
            const int64_t pixel_offset = (static_cast<int64_t>(y) * output_width) + x;

            for (int32_t channel = 0; channel < k_num_channels; ++channel) {
                const float value = output_planar[(channel * num_output_pixels) + pixel_offset];
                output_interleaved[static_cast<size_t>((pixel_offset * k_num_channels) + channel)] =
                    static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
            }
        }
    }

    const std::string default_input =
        input_value == input_path ? input_path : std::filesystem::path(input_path).filename().string();
    const std::string path = output_path->empty() ? default_output_path(default_input) : *output_path;

    COMPUTE_FLOW_CHECK(stbi_write_png(path.c_str(), output_width, output_height, k_num_channels,
                                      output_interleaved.data(), output_width * k_num_channels) != 0);

    std::cout << "Wrote edited image to " << path << std::endl;
}

void run_multiple(IBrickV1& brick, const std::string& runs_file) {
    const auto run_configs = internal::load_run_configs(runs_file, {"i", "p", "o"}, {"i", "p"});
    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick.reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::optional<std::string> output_path =
            config.contains("o") ? std::make_optional(internal::get_run_value<std::string>(config, "o", "", run_index))
                                 : std::nullopt;
        const std::string input_path = internal::get_run_value<std::string>(config, "i", "", run_index);
        const std::string prompt     = internal::get_run_value<std::string>(config, "p", "", run_index);

        COMPUTE_FLOW_CHECK(!input_path.empty());
        COMPUTE_FLOW_CHECK(!prompt.empty());

        if (output_path.has_value()) {
            COMPUTE_FLOW_CHECK(!output_path->empty());
        }

        int32_t width  = 0;
        int32_t height = 0;

        // Load the image
        std::vector<float> image =
            internal::load_rgb_image_chw(input_path, width, height, internal::ImageDecodeMode::NormalizedU8);
        const int32_t source_width  = width;
        const int32_t source_height = height;
        const int32_t padded_width  = round_up(width, k_dimension_multiple);
        const int32_t padded_height = round_up(height, k_dimension_multiple);

        if (padded_width != width || padded_height != height) {
            image  = pad_image_chw(image, width, height, padded_width, padded_height);
            width  = padded_width;
            height = padded_height;
        }

        TensorDict input;

        // Set the input
        input.declare_tensor("image").format  = {DataType::F32, {k_num_channels, height, width}};
        input["image"].data                   = image.data();
        input.declare_tensor("prompt").format = {DataType::String, {static_cast<int64_t>(prompt.size())}};
        input["prompt"].data                  = const_cast<char*>(prompt.data());

        TensorDict output;

        // Run the model inference
        while (!brick.is_generation_complete()) {
            output = brick.run(input);
        }

        const TensorData& output_image = output["image"];

        COMPUTE_FLOW_CHECK(output_image.format.dt == DataType::F32 && output_image.format.shape.size() == 3 &&
                           output_image.format.shape[0] == k_num_channels);

        const int32_t output_height = static_cast<int32_t>(output_image.format.shape[1]);
        const int32_t output_width  = static_cast<int32_t>(output_image.format.shape[2]);
        std::cout << "Original input resolution: " << source_width << " x " << source_height << '\n'
                  << "Output resolution: " << output_width << " x " << output_height << '\n';

        // Display profiler run
        brick.write_profile(std::cout);

        if (!output_path.has_value()) {
            continue;
        }

        // Save the output image
        const int64_t num_output_pixels = static_cast<int64_t>(output_height) * output_width;
        const auto* output_planar       = static_cast<const float*>(output_image.data);
        std::vector<uint8_t> output_interleaved(static_cast<size_t>(k_num_channels * num_output_pixels));

        for (int32_t y = 0; y < output_height; ++y) {
            for (int32_t x = 0; x < output_width; ++x) {
                const int64_t pixel_offset = (static_cast<int64_t>(y) * output_width) + x;

                for (int32_t channel = 0; channel < k_num_channels; ++channel) {
                    const float value = output_planar[(channel * num_output_pixels) + pixel_offset];
                    output_interleaved[static_cast<size_t>((pixel_offset * k_num_channels) + channel)] =
                        static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
                }
            }
        }

        const std::string path = *output_path;

        COMPUTE_FLOW_CHECK(stbi_write_png(path.c_str(), output_width, output_height, k_num_channels,
                                          output_interleaved.data(), output_width * k_num_channels) != 0);

        std::cout << "Wrote edited image to " << path << std::endl;
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // Parse the command-line arguments
    std::string manifest_path;
    std::string input_path;
    std::string output_path;
    std::string prompt;
    std::string json_file;
    std::string thread_spec  = "1";
    bool enable_op_profiling = false;

    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option = 0;
    while ((option = ::getopt_long(argc, argv, "m:i:o:p:t:j:h", long_options, nullptr)) != -1) {
        switch (option) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'i':
                input_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 'p':
                prompt = optarg;
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
    if (manifest_path.empty() || (json_file.empty() && (input_path.empty() || prompt.empty())) ||
        (!json_file.empty() && (!input_path.empty() || !prompt.empty() || !output_path.empty()))) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "image_to_image");

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

    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(*brick, json_file);
    } else {
        brick->reset_states();
        run_single(*brick, manifest_path, input_path, prompt, output_path);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::image_to_image

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::image_to_image::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
