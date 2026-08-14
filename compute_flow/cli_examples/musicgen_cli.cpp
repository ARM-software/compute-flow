//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

constexpr size_t k_seed_default = 99;
constexpr size_t k_audio_len_sec_default = 180;
constexpr size_t k_num_steps_default = 8;

namespace compute_flow::bricks::musicgen {

static void print_usage(const char *name) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -p <prompt> [-t <thread_spec>] [-s <seed> -l <audio_len>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -p <prompt>               Input prompt text (e.g., warm arpeggios on house beats 120BPM with drums effect)\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -s <seed>                 (Optional) Random seed for reproducibility. Different seeds generate different audio samples (Default: %zu)\n"
        "  -l <audio_len_sec>        (Optional) Length of generated audio (Default: %zu s)\n"
        "  -n <num_steps>            (Optional) Number of steps (Default: %zu)\n"
        "  -o <output_file>          (Optional) Output audio file name (Default: <prompt>_<seed>.wav)\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name,
        k_seed_default,
        k_audio_len_sec_default,
        k_num_steps_default);
}

static std::string get_filename(std::string prompt, size_t seed) {
    // Convert spaces to underscores
    std::replace(prompt.begin(), prompt.end(), ' ', '_');

    // Convert to lowercase
    std::transform(prompt.begin(), prompt.end(), prompt.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return prompt + "_" + std::to_string(seed) + ".wav";
}

int32_t run_cli(int32_t argc, char** argv) {

    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path    = "";
    std::string txt_prompt       = "";
    std::string thread_spec = "1";
    // Optional arguments
    int64_t seed                 = k_seed_default;
    int64_t num_steps            = k_num_steps_default;
    float audio_len_sec          = static_cast<float>(k_audio_len_sec_default);
    bool enable_op_profiling     = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:s:n:l:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path    = optarg; break;
            case 'p': txt_prompt       = optarg; break;
            case 't': thread_spec = optarg; break;
            case 's': seed             = static_cast<int64_t>(std::stoull(optarg)); break;
            case 'n': num_steps        = static_cast<int64_t>(std::stoull(optarg)); break;
            case 'l': audio_len_sec    = static_cast<float>(std::stoull(optarg)); break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || txt_prompt.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "musicgen");

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

    // Set the input
    input.declare_tensor("text_prompt");
    input.declare_tensor("seed");
    input.declare_tensor("audio_length_sec");

    // Text prompt - shape[number of chars, 1]
    input["text_prompt"].data         = (void *)txt_prompt.c_str();
    input["text_prompt"].format.dt    = DataType::String;
    input["text_prompt"].format.shape = {(int64_t)txt_prompt.size(), 1};

    // Seed - Scalar
    input["seed"].data         = (void *)&seed;
    input["seed"].format.dt    = DataType::U64;
    input["seed"].format.shape = {1, 1};

    // Audio length - Scalar
    input["audio_length_sec"].data         = (void *)&audio_len_sec;
    input["audio_length_sec"].format.dt    = DataType::F32;
    input["audio_length_sec"].format.shape = {1};

    // Run the model inference
    TensorDict output = brick->run(input);

    // Display profiler run
    brick->write_profile(std::cout);

    // filename = <prompt>_<seed>.wav
    std::string output_file = get_filename(txt_prompt, seed);

    internal::write_stereo_wav(
        output_file.c_str(),
        (float*)(output["left_channel"].data),
        (float*)(output["right_channel"].data),
        output["left_channel"].format.shape[0],
        44100);

    return 0;
}

}  // namespace compute_flow::bricks::musicgen

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::musicgen::run_cli(argc, argv);
}
