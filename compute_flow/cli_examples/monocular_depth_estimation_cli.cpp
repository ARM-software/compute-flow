//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace compute_flow::bricks::monocular_depth_estimation {
namespace {

constexpr int64_t k_num_channels = 3;

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> -i <input_image> [-o <output.png>] "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -i <input_image>       Image path or @sample/... selector.\n"
                 "  -o <output.png>        Relative-depth visualization path "
                 "(default: ./<input-stem>_relative_depth.png).\n"
                 "  -t <thread_spec>       Default threads and optional model overrides "
                 "(default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n",
                 executable);
}

std::string default_output_path(const std::string& input_path) {
    const std::filesystem::path path(input_path);
    return path.stem().string() + "_relative_depth.png";
}

std::array<uint8_t, 3> depth_color(float normalized) {
    constexpr std::array<std::array<uint8_t, 3>, 5> k_palette = {
        {{{48, 18, 59}}, {{50, 100, 180}}, {{45, 200, 170}}, {{245, 220, 70}}, {{180, 4, 38}}}};
    constexpr int64_t k_num_segments = static_cast<int64_t>(k_palette.size()) - 1;

    const float scaled  = std::clamp(normalized, 0.0F, 1.0F) * static_cast<float>(k_num_segments);
    const int64_t lower = std::min(static_cast<int64_t>(std::floor(scaled)), k_num_segments - 1);
    const float weight  = scaled - static_cast<float>(lower);
    std::array<uint8_t, 3> color{};
    for (int64_t channel = 0; channel < 3; ++channel) {
        const float value =
            static_cast<float>(k_palette[lower][channel]) +
            ((static_cast<float>(k_palette[lower + 1][channel]) - static_cast<float>(k_palette[lower][channel])) *
             weight);
        color[channel] = static_cast<uint8_t>(std::round(value));
    }
    return color;
}

std::vector<uint8_t> colorize_relative_depth(const float* relative_depth, int64_t num_pixels) {
    COMPUTE_FLOW_CHECK(relative_depth != nullptr);
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t index = 0; index < num_pixels; ++index) {
        const float value = relative_depth[index];
        if (std::isfinite(value) && value != 0.0F) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }

    std::vector<uint8_t> image(static_cast<std::size_t>(num_pixels * 3), 0);
    if (!std::isfinite(minimum)) {
        return image;
    }

    for (int64_t index = 0; index < num_pixels; ++index) {
        const float value = relative_depth[index];
        if (!std::isfinite(value) || value == 0.0F) {
            continue;
        }
        const float normalized             = maximum > minimum ? (value - minimum) / (maximum - minimum) : 0.5F;
        const std::array<uint8_t, 3> color = depth_color(normalized);
        const std::size_t offset           = static_cast<std::size_t>(index * 3);
        for (int64_t channel = 0; channel < 3; ++channel) {
            image[offset + static_cast<std::size_t>(channel)] = color[channel];
        }
    }
    return image;
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string input_path;
    std::string output_path;
    std::string thread_spec  = "1";
    bool enable_op_profiling = false;

    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option = 0;
    while ((option = ::getopt_long(argc, argv, "m:i:o:t:h", long_options, nullptr)) != -1) {
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

    if (manifest_path.empty() || input_path.empty()) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    manifest_path = internal::get_manifest_path(manifest_path, "monocular_depth_estimation");
    input_path    = internal::get_sample_path(input_path, manifest_path);
    if (output_path.empty()) {
        output_path = default_output_path(input_path);
    }

    int32_t width  = 0;
    int32_t height = 0;
    std::vector<float> image =
        internal::load_rgb_image_chw(input_path, width, height, internal::ImageDecodeMode::NormalizedU8);
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    auto manifest               = load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    auto brick = load_brick(manifest);
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;
    brick->setup(manifest, options);
    brick->reset_states();

    TensorDict input;
    input.declare_tensor("image").format = {DataType::F32, {k_num_channels, height, width}};
    input["image"].data                  = image.data();

    TensorDict output;
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }
    brick->write_profile(std::cout);

    const TensorData& relative_depth = output["relative_depth"];
    if (relative_depth.format.dt != DataType::F32 || relative_depth.format.shape.size() != 2 ||
        relative_depth.format.shape[0] <= 0 || relative_depth.format.shape[1] <= 0 ||
        relative_depth.format.shape[0] > std::numeric_limits<int32_t>::max() ||
        relative_depth.format.shape[1] > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Monocular depth-estimation brick returned an unsupported output format");
    }

    const int32_t output_height = static_cast<int32_t>(relative_depth.format.shape[0]);
    const int32_t output_width  = static_cast<int32_t>(relative_depth.format.shape[1]);
    const int64_t num_pixels    = static_cast<int64_t>(output_height) * output_width;
    const std::vector<uint8_t> visualization =
        colorize_relative_depth(static_cast<const float*>(relative_depth.data), num_pixels);
    if (stbi_write_png(output_path.c_str(), output_width, output_height, 3, visualization.data(), output_width * 3) ==
        0) {
        throw std::runtime_error("Failed to write relative-depth visualization: " + output_path);
    }

    std::cout << "Input image resolution: " << width << "x" << height << std::endl;
    std::cout << "Output image resolution: " << output_width << "x" << output_height << std::endl;
    std::cout << "Wrote relative-depth visualization to " << output_path << std::endl;
    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::monocular_depth_estimation

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::monocular_depth_estimation::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
