//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"

#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <unistd.h>
#include <vector>

namespace compute_flow::bricks::stt {

static void print_usage(const char *name) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -a <audio_path> [-t <thread_spec>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -a <audio_path>           Input WAV path or @sample/... selector\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -o <output_file>          (Optional) Output audio file name (Default: <prompt>_<seed>.wav)\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

int32_t run_cli(int32_t argc, char** argv) {

    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path    = "";
    std::string audio_path       = "";
    std::string thread_spec = "1";
    bool enable_op_profiling     = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:a:t:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path    = optarg; break;
            case 'a': audio_path       = optarg; break;
            case 't': thread_spec = optarg; break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || audio_path.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "stt");
    audio_path = compute_flow::bricks::internal::get_sample_path(audio_path, manifest_path);

    // Load the manifest
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = compute_flow::bricks::load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    TensorDict input;

    std::vector<float> mono_raw_audio;

    // Read audio file
    const int64_t sample_rate = internal::read_mono_wav(audio_path, mono_raw_audio);

    // Set the input
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");

    // Text prompt - shape[number of chars, 1]
    input["raw_wav"].data         = (void *)mono_raw_audio.data();
    input["raw_wav"].format.dt    = DataType::F32;
    input["raw_wav"].format.shape = {1, (int64_t)mono_raw_audio.size()};
    input["sample_rate"].data         = (void*)&sample_rate;
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    // Run the model inference
    std::cout << std::endl;
    std::cout << "----------------" <<std::endl;
    std::cout << "[TRANSCRIPTION]" << std::endl;

    while (!brick->is_generation_complete()) {
        TensorDict output = brick->run(input);
        const auto text_output = output["text"];
        const int64_t nbytes = text_output.format.get_nbytes();

        if (nbytes > 0) {
            std::cout.write(static_cast<const char*>(text_output.data), nbytes);
            std::cout << std::flush;
        }
    }
    std::cout << std::endl;
    std::cout << "----------------" <<std::endl;
    std::cout << std::endl;

    // Display profiler run
    brick->write_profile(std::cout);

    return 0;
}

}  // namespace compute_flow::bricks::stt

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::stt::run_cli(argc, argv);
}
