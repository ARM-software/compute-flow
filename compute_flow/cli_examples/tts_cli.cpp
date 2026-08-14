//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <cassert>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::tts {
namespace {

void print_usage(const char* name) {
    fprintf(stderr,
            "Usage: %s -m <manifest_path> (-p <prompt> | -j <runs.json>) [-t <thread_spec>] [-o <output_file>]\n\n"
            "Options:\n"
            "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
            "  -p <prompt>               Input prompt text (e.g., Hi! I am your personal assistant)\n"
            "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
            "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally "
            "override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
            "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
            "  -o <output_file>          (Optional) Output audio file name (Default: output.wav)\n"
            "  -h                        Show this help message\n",
            name);
}

void run_single(std::string manifest_path, const std::string& prompt, const std::optional<std::string>& output_path,
                const std::string& thread_spec, bool enable_op_profiling) {
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = compute_flow::bricks::internal::get_manifest_path(manifest_path, "tts");

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

    TensorDict input;

    // Set the input
    input.declare_tensor("text_prompt");

    // Text prompt - shape[number of chars, 1]
    input["text_prompt"].data         = const_cast<char*>(prompt.data());
    input["text_prompt"].format.dt    = DataType::String;
    input["text_prompt"].format.shape = {static_cast<int64_t>(prompt.size())};

    size_t total_samples    = 0;
    double run_time_s       = 0.0;
    double first_run_time_s = -1.0;
    int64_t sample_rate     = 0;
    std::vector<float> left_full;
    std::vector<float> right_full;

    // Run the model inference
    while (!brick->is_generation_complete()) {
        // Start profiling
        const auto run_start        = std::chrono::steady_clock::now();
        TensorDict output           = brick->run(input);

        // End profiling
        const auto run_end          = std::chrono::steady_clock::now();
        const double run_duration_s = std::chrono::duration<double>(run_end - run_start).count();
        run_time_s += run_duration_s;
        if (first_run_time_s < 0.0) {
            first_run_time_s = run_duration_s;
        }

        const auto left_tensor    = output["left_channel"];
        const auto right_tensor   = output["right_channel"];

        // Assuming left and right channels with the same number of samples
        const int64_t num_samples = left_tensor.format.shape[0];
        const auto* left_ptr      = static_cast<const float*>(left_tensor.data);
        const auto* right_ptr     = static_cast<const float*>(right_tensor.data);

        total_samples += static_cast<size_t>(num_samples);
        left_full.insert(left_full.end(), left_ptr, left_ptr + num_samples);
        right_full.insert(right_full.end(), right_ptr, right_ptr + num_samples);

        // Get the sample rate
        sample_rate = reinterpret_cast<const int64_t*>(output["sample_rate"].data)[0];
    }

    // Display profiler run
    brick->write_profile(std::cout);

    const double audio_len_s =
        sample_rate > 0 ? static_cast<double>(total_samples) / static_cast<double>(sample_rate) : 0.0;
    const double rtf = audio_len_s > 0.0 ? run_time_s / audio_len_s : 0.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Run time (sum of brick.run): " << run_time_s << "s\n";
    std::cout << "Audio length: " << audio_len_s << "s\n";
    if (audio_len_s > 0.0) {
        std::cout << "Real-time factor (RTF): " << rtf << " ( < 1 means realtime )\n";
    } else {
        std::cout << "Real-time factor (RTF): unavailable\n";
    }
    if (first_run_time_s < 0.0) {
        std::cout << "Time to first chunk played: unavailable\n";
    } else {
        std::cout << "Time to first chunk played: " << first_run_time_s << "s\n";
    }

    if (output_path.has_value()) {
        const std::string path = output_path->empty() ? "output.wav" : *output_path;
        internal::write_stereo_wav(path, left_full.data(), right_full.data(), left_full.size(), sample_rate);
    }
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path            = compute_flow::bricks::internal::get_manifest_path(manifest_path, "tts");

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

    const auto run_configs = internal::load_run_configs(runs_file, {"p", "o"}, {"p"});
    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::optional<std::string> output_path =
            config.contains("o")
                ? std::make_optional(internal::get_run_value<std::string>(config, "o", "", run_index))
                : std::nullopt;
        const std::string prompt = internal::get_run_value<std::string>(config, "p", "", run_index);
        COMPUTE_FLOW_CHECK(!prompt.empty());

        TensorDict input;

        // Set the input
        input.declare_tensor("text_prompt");

        // Text prompt - shape[number of chars, 1]
        input["text_prompt"].data         = const_cast<char*>(prompt.data());
        input["text_prompt"].format.dt    = DataType::String;
        input["text_prompt"].format.shape = {static_cast<int64_t>(prompt.size())};

        size_t total_samples    = 0;
        double run_time_s       = 0.0;
        double first_run_time_s = -1.0;
        int64_t sample_rate     = 0;
        std::vector<float> left_full;
        std::vector<float> right_full;

        // Run the model inference
        while (!brick->is_generation_complete()) {
            // Start profiling
            const auto run_start        = std::chrono::steady_clock::now();
            TensorDict output           = brick->run(input);

            // End profiling
            const auto run_end          = std::chrono::steady_clock::now();
            const double run_duration_s = std::chrono::duration<double>(run_end - run_start).count();
            run_time_s += run_duration_s;
            if (first_run_time_s < 0.0) {
                first_run_time_s = run_duration_s;
            }

            const auto left_tensor    = output["left_channel"];
            const auto right_tensor   = output["right_channel"];

            // Assuming left and right channels with the same number of samples
            const int64_t num_samples = left_tensor.format.shape[0];
            const auto* left_ptr      = static_cast<const float*>(left_tensor.data);
            const auto* right_ptr     = static_cast<const float*>(right_tensor.data);

            total_samples += static_cast<size_t>(num_samples);
            left_full.insert(left_full.end(), left_ptr, left_ptr + num_samples);
            right_full.insert(right_full.end(), right_ptr, right_ptr + num_samples);

            // Get the sample rate
            sample_rate = reinterpret_cast<const int64_t*>(output["sample_rate"].data)[0];
        }

        // Display profiler run
        brick->write_profile(std::cout);

        const double audio_len_s =
            sample_rate > 0 ? static_cast<double>(total_samples) / static_cast<double>(sample_rate) : 0.0;
        const double rtf = audio_len_s > 0.0 ? run_time_s / audio_len_s : 0.0;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Run time (sum of brick.run): " << run_time_s << "s\n";
        std::cout << "Audio length: " << audio_len_s << "s\n";
        if (audio_len_s > 0.0) {
            std::cout << "Real-time factor (RTF): " << rtf << " ( < 1 means realtime )\n";
        } else {
            std::cout << "Real-time factor (RTF): unavailable\n";
        }
        if (first_run_time_s < 0.0) {
            std::cout << "Time to first chunk played: unavailable\n";
        } else {
            std::cout << "Time to first chunk played: " << first_run_time_s << "s\n";
        }

        if (output_path.has_value()) {
            const std::string path = output_path->empty() ? "output.wav" : *output_path;
            internal::write_stereo_wav(path, left_full.data(), right_full.data(), left_full.size(), sample_rate);
        }
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // ----- Parse the cmd line arguments
    // ----------------------------------
    // Required arguments
    std::string manifest_path    = "";
    std::string txt_prompt       = "";
    std::string thread_spec      = "1";
    std::string output_file_name = "";
    std::string json_file;
    bool enable_op_profiling = false;

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
                output_file_name = optarg;
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
                return (opt == 'h') ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || (json_file.empty() && txt_prompt.empty()) ||
        (!json_file.empty() && (!txt_prompt.empty() || !output_file_name.empty()))) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool multiple_runs_mode = !json_file.empty();
    if (multiple_runs_mode) {
        run_multiple(manifest_path, json_file, thread_spec, enable_op_profiling);
    } else {
        run_single(manifest_path, txt_prompt, output_file_name, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::tts

int32_t main(int32_t argc, char** argv) { return compute_flow::bricks::tts::run_cli(argc, argv); }
