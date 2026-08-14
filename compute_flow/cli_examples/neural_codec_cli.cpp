//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include <npy.hpp>

namespace compute_flow::bricks::neural_codec {

static void print_usage(const char* name) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -c <codes_bin> [-t <thread_spec>] [-o <output_file>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -c <codes_npy>            Codes path or @sample/... selector (.npy)\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -o <output_file>          Output wav file (default: output.wav)\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string codes_path;
    std::string output_file = "output.wav";
    std::string thread_spec = "1";
    bool enable_op_profiling = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:c:t:o:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path = optarg; break;
            case 'c': codes_path = optarg; break;
            case 't': thread_spec = optarg; break;
            case 'o': output_file = optarg; break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (manifest_path.empty() || codes_path.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "neural_codec");
    codes_path = compute_flow::bricks::internal::get_sample_path(codes_path, manifest_path);

    npy::npy_data codes_npy = npy::read_npy<int32_t>(codes_path);
    if (codes_npy.data.empty()) {
        throw std::runtime_error("Codes file is empty: " + codes_path);
    }
    std::vector<int32_t> codes = std::move(codes_npy.data);

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup = false;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    TensorDict input;
    input.declare_tensor("codes");

    // Set the input
    const int64_t total_codes = static_cast<int64_t>(codes.size());
    input["codes"].data = (void*)codes.data();
    input["codes"].format.dt = DataType::I32;
    input["codes"].format.shape = {1, 1, total_codes}; // #batch, mono, time

    // Run the model inference
    TensorDict output = brick->run(input);

    // Display profiler run
    brick->write_profile(std::cout);

    const auto& left = output["left_channel"];
    const auto& right = output["right_channel"];

    const size_t samples = static_cast<size_t>(left.format.shape.back());
    const float* left_ch = static_cast<const float*>(left.data);
    const float* right_ch = static_cast<const float*>(right.data);
    const int32_t output_sample_rate = manifest.get_constant<int32_t>("output_sample_rate");
    internal::write_stereo_wav(
        output_file,
        left_ch,
        right_ch,
        samples,
        output_sample_rate
    );

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::neural_codec

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::neural_codec::run_cli(argc, argv);
}
