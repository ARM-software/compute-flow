//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::audiogen {
namespace {

void print_usage(const char* name) {
    fprintf(
        stderr,
        "Usage: %s -m <manifest_path> (-p <prompt> | -j <runs.json>) [-t <thread_spec>] [-o <output_file>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -p <prompt>               Input prompt text (e.g., warm arpeggios on house beats 120BPM with drums effect)\n"
        "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally "
        "override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -o <output_file>          (Optional) Output audio file name (Default: <prompt>_<seed>.wav)\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

std::string get_filename(std::string prompt, size_t seed) {
    // Convert spaces to underscores
    std::replace(prompt.begin(), prompt.end(), ' ', '_');

    // Convert to lowercase
    std::transform(prompt.begin(), prompt.end(), prompt.begin(), [](unsigned char c) { return std::tolower(c); });

    return prompt + "_" + std::to_string(seed) + ".wav";
}

void run_single(IBrickV1& brick, const std::string& prompt, int64_t seed,
                const std::optional<std::string>& output_path) {
    TensorDict input;

    // Set the input
    input.declare_tensor("text_prompt");

    // Text prompt - shape[number of chars, 1]
    input["text_prompt"].data         = const_cast<char*>(prompt.data());
    input["text_prompt"].format.dt    = DataType::String;
    input["text_prompt"].format.shape = {static_cast<int64_t>(prompt.size()), 1};

    // Run the model inference
    TensorDict output = brick.run(input);

    const int64_t sample_rate = reinterpret_cast<const int64_t*>(output["sample_rate"].data)[0];

    if (output_path.has_value()) {
        // filename = <prompt>_<seed>.wav
        const std::string path = output_path->empty() ? get_filename(prompt, static_cast<size_t>(seed)) : *output_path;
        internal::write_stereo_wav(path, static_cast<const float*>(output["left_channel"].data),
                                   static_cast<const float*>(output["right_channel"].data),
                                   output["left_channel"].format.shape[0], sample_rate);
    }

    // Display profiler run
    brick.write_profile(std::cout);
}

void run_multiple(IBrickV1& brick, const std::string& runs_file, int64_t seed) {
    const auto run_configs = internal::load_run_configs(runs_file, {"p", "o"}, {"p"});
    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];
        brick.reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::optional<std::string> output_path =
            config.contains("o") ? std::make_optional(internal::get_run_value<std::string>(config, "o", "", run_index))
                                 : std::nullopt;
        const std::string prompt = internal::get_run_value<std::string>(config, "p", "", run_index);

        COMPUTE_FLOW_CHECK(!prompt.empty());

        if (output_path.has_value()) {
            COMPUTE_FLOW_CHECK(!output_path->empty());
        }

        TensorDict input;

        // Set the input
        input.declare_tensor("text_prompt");

        // Text prompt - shape[number of chars, 1]
        input["text_prompt"].data         = const_cast<char*>(prompt.data());
        input["text_prompt"].format.dt    = DataType::String;
        input["text_prompt"].format.shape = {static_cast<int64_t>(prompt.size()), 1};

        // Run the model inference
        TensorDict output         = brick.run(input);
        const int64_t sample_rate = reinterpret_cast<const int64_t*>(output["sample_rate"].data)[0];

        if (output_path.has_value()) {
            // filename = <prompt>_<seed>.wav
            const std::string path =
                output_path->empty() ? get_filename(prompt, static_cast<size_t>(seed)) : *output_path;
            internal::write_stereo_wav(path, static_cast<const float*>(output["left_channel"].data),
                                       static_cast<const float*>(output["right_channel"].data),
                                       output["left_channel"].format.shape[0], sample_rate);
        }

        // Display profiler run
        brick.write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path = "";
    std::string txt_prompt    = "";
    std::string thread_spec   = "1";
    std::string output_file;
    std::string json_file;
    bool enable_op_profiling            = false;
    static struct option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:o:j:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'p':
                txt_prompt = optarg;
                break;
            case 't':
                thread_spec = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'j':
                json_file = optarg;
                break;
            case 1001:
                enable_op_profiling = true;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || (json_file.empty() && txt_prompt.empty()) ||
        (!json_file.empty() && (!txt_prompt.empty() || !output_file.empty()))) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "audiogen");

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

    const int64_t seed            = manifest.get_constant<int64_t>("seed");
    const bool multiple_runs_mode = !json_file.empty();

    if (multiple_runs_mode) {
        run_multiple(*brick, json_file, seed);
    } else {
        run_single(*brick, txt_prompt, seed, output_file);
    }

    return 0;
}

}  // namespace compute_flow::bricks::audiogen

int32_t main(int32_t argc, char** argv) { return compute_flow::bricks::audiogen::run_cli(argc, argv); }
