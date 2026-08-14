//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/image.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

namespace compute_flow::bricks::object_detection {

namespace {

constexpr int32_t k_num_channels = 3;

} // namespace

static void print_usage(const char *name) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -i <image> [-t <thread_spec>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -i <image>                Input image path or @sample/... selector\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

int32_t run_cli(int32_t argc, char** argv) {
    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path = "";
    std::string input_path    = "";
    std::string thread_spec = "1";
    bool enable_op_profiling  = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:i:t:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path = optarg; break;
            case 'i': input_path    = optarg; break;
            case 't': thread_spec = optarg; break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || input_path.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "object_detection");
    input_path = compute_flow::bricks::internal::get_sample_path(input_path, manifest_path);

    int32_t img_w = 0;
    int32_t img_h = 0;

    // Load the image
    auto img = internal::load_rgb_image_chw(input_path, img_w, img_h, internal::ImageDecodeMode::StbFloat);

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the manifest
    BrickOptions options;
    options.runtime.enable_warmup = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    TensorDict input;

    // Set the input
    input.declare_tensor("img");

    input["img"].data         = (void *)img.data();
    input["img"].format.dt    = DataType::F32;
    input["img"].format.shape = {1, k_num_channels, img_h, img_w};

    // Run the model inference
    TensorDict output = brick->run(input);

    // Display profiler run
    brick->write_profile(std::cout);

    const auto class_names = output["name"];
    const int64_t num_classes = class_names.format.shape[1];
    const int64_t max_name_bytes = class_names.format.shape[2];
    const char* names = reinterpret_cast<const char*>(class_names.data);

    for(int64_t i = 0; i < class_names.format.shape[0]; ++i) {
        const char* name = names + i * num_classes * max_name_bytes;
        const char* name_end = name + max_name_bytes;
        const char* terminator = std::find(name, name_end, '\0');
        const std::string class_name(name, terminator);
        printf("Detected: %s\n", class_name.c_str());
    }

    return 0;
}

}  // namespace compute_flow::bricks::object_detection

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::object_detection::run_cli(argc, argv);
}
