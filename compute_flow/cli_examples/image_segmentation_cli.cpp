//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace compute_flow::bricks::image_segmentation {
namespace {

constexpr int64_t k_num_channels = 3;

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-i <input_image> | -j <runs.json>) [-o <output.png>] "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -i <input_image>       Image path or @sample/... selector.\n"
                 "  -j <runs.json>         Input JSON for multiple runs; each run saves a PNG.\n"
                 "  -o <output.png>        Segmentation visualization path "
                 "(default: ./<input-stem>_segmentation.png for one run).\n"
                 "                        Multi-run defaults add the run number; JSON 'o' overrides it.\n"
                 "  -t <thread_spec>       Default threads and optional model overrides "
                 "(default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n",
                 executable);
}

std::string default_output_path(const std::string& input_path, int64_t run_number = 0) {
    const std::filesystem::path path(input_path);
    const std::string suffix = run_number == 0 ? "" : "_" + std::to_string(run_number);
    return path.stem().string() + "_segmentation" + suffix + ".png";
}

std::array<uint8_t, 3> class_color(int32_t class_id) {
    const uint32_t value = static_cast<uint32_t>(class_id) * 0x9E3779B1U;
    return {static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value)};
}

std::vector<uint8_t> colorize_mask(const int32_t* mask, int64_t num_pixels) {
    std::vector<uint8_t> image(static_cast<std::size_t>(num_pixels * 3));

    for (int64_t index = 0; index < num_pixels; ++index) {
        const std::array<uint8_t, 3> color = class_color(mask[index]);
        const std::size_t offset = static_cast<std::size_t>(index * 3);

        for (int64_t channel = 0; channel < 3; ++channel) {
            image[offset + static_cast<std::size_t>(channel)] = color[channel];
        }
    }

    return image;
}

void display_result(const TensorDict& output, int32_t width, int32_t height, const std::string& output_path) {
    const TensorData& mask = output["mask"];
    const TensorData& class_names = output["class_name"];

    COMPUTE_FLOW_CHECK(mask.data != nullptr && mask.format.dt == DataType::I32 && mask.format.shape.size() == 2 &&
                       mask.format.shape[0] == height && mask.format.shape[1] == width);
    COMPUTE_FLOW_CHECK(class_names.data != nullptr && class_names.format.dt == DataType::U8 &&
                       class_names.format.shape.size() == 2 && class_names.format.shape[0] > 0 &&
                       class_names.format.shape[1] > 0);

    const int64_t num_classes = class_names.format.shape[0];
    const int64_t name_length = class_names.format.shape[1];
    const int32_t* mask_values = static_cast<const int32_t*>(mask.data);
    const uint8_t* names = static_cast<const uint8_t*>(class_names.data);
    const int64_t num_pixels = static_cast<int64_t>(width) * height;

    for (int64_t index = 0; index < num_pixels; ++index) {
        COMPUTE_FLOW_CHECK(mask_values[index] >= 0 && mask_values[index] < num_classes);
    }

    std::cout << "Segmentation resolution: " << width << "x" << height << '\n';
    std::cout << "Classes:\n";

    for (int64_t index = 0; index < num_classes; ++index) {
        std::string name;

        for (int64_t byte_index = 0; byte_index < name_length; ++byte_index) {
            const uint8_t byte = names[index * name_length + byte_index];

            if (byte == 0) {
                break;
            }

            name.push_back(static_cast<char>(byte));
        }

        std::cout << index << " " << name << '\n';
    }

    const std::vector<uint8_t> visualization = colorize_mask(mask_values, num_pixels);

    COMPUTE_FLOW_CHECK(stbi_write_png(output_path.c_str(), width, height, 3, visualization.data(), width * 3) != 0);

    std::cout << "Wrote segmentation visualization to " << output_path << '\n';
}

void run_single(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                const std::string& input_value, const std::string& output_path) {
    manifest_path               = internal::get_manifest_path(manifest_path, "image_segmentation");
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

    const std::string input_path = internal::get_sample_path(input_value, manifest_path);
    int32_t width                = 0;
    int32_t height               = 0;

    // Load the image
    std::vector<float> image =
        internal::load_rgb_image_chw(input_path, width, height, internal::ImageDecodeMode::NormalizedU8);

    TensorDict input;

    // Set the input
    input.declare_tensor("image").format = {DataType::F32, {k_num_channels, height, width}};
    input["image"].data                  = image.data();

    TensorDict output;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        output = brick->run(input);
    }

    const std::string default_input =
        input_value == input_path ? input_path : std::filesystem::path(input_path).filename().string();
    const std::string path = output_path.empty() ? default_output_path(default_input) : output_path;

    display_result(output, width, height, path);

    // Display profiler run
    brick->write_profile(std::cout);
}

void run_multiple(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                  const std::string& runs_file) {
    manifest_path               = internal::get_manifest_path(manifest_path, "image_segmentation");
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    auto manifest               = load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    auto brick = load_brick(manifest);
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    brick->setup(manifest, options);

    const auto run_configs = internal::load_run_configs(runs_file, {"i", "o"}, {"i"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string input_path = internal::get_run_value<std::string>(config, "i", "", run_index);

        COMPUTE_FLOW_CHECK(!input_path.empty());

        const std::string output_path =
            config.contains("o") ? internal::get_run_value<std::string>(config, "o", "", run_index)
                                 : default_output_path(input_path, run_index + 1);

        COMPUTE_FLOW_CHECK(!output_path.empty());

        int32_t width  = 0;
        int32_t height = 0;

        // Load the image
        std::vector<float> image =
            internal::load_rgb_image_chw(input_path, width, height, internal::ImageDecodeMode::NormalizedU8);

        TensorDict input;

        // Set the input
        input.declare_tensor("image").format = {DataType::F32, {k_num_channels, height, width}};
        input["image"].data                  = image.data();

        TensorDict output;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            output = brick->run(input);
        }

        display_result(output, width, height, output_path);

        // Display profiler run
        brick->write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string input_path;
    std::string output_path;
    std::string json_file;
    std::string thread_spec  = "1";
    bool enable_op_profiling = false;

    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option = 0;
    while ((option = ::getopt_long(argc, argv, "m:i:o:t:j:h", long_options, nullptr)) != -1) {
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

    if (manifest_path.empty() || (json_file.empty() && input_path.empty()) ||
        (!json_file.empty() && (!input_path.empty() || !output_path.empty()))) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(manifest_path, thread_spec, enable_op_profiling, json_file);
    } else {
        run_single(manifest_path, thread_spec, enable_op_profiling, input_path, output_path);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::image_segmentation

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::image_segmentation::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
