//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::lm_text {
namespace {

struct GenerationStats {
    double prefill_time_s     = 0.0;
    double decode_time_s      = 0.0;
    int64_t num_decode_tokens = 0;
};

void print_usage(const char* name) {
    std::fprintf(
        stderr,
        "Usage: %s -m <manifest_path> (-p <prompt> | -j <runs.json>) [-t <thread_spec>] [-n "
        "<max_generated_tokens>] [-o <output_file>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -p <prompt>               Input prompt text\n"
        "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally "
        "override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -n <max_generated_tokens> Maximum number of generated tokens (default=max int64)\n"
        "  -o <output_file>          Write generated text to this file instead of stdout\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

GenerationStats run_single(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                           const std::string& text_prompt, int64_t max_generated_tokens,
                           const std::string& output_file) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "lm_text");

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

    std::ofstream output_file_stream;
    std::ostream* generated_text_stream = &std::cout;

    if (!output_file.empty()) {
        output_file_stream.open(output_file, std::ios::binary);

        COMPUTE_FLOW_CHECK(output_file_stream.is_open());

        generated_text_stream = &output_file_stream;
    }

    TensorDict input;

    // Set the input
    input.declare_tensor("text_prompt");
    input["text_prompt"].data         = const_cast<char*>(text_prompt.data());
    input["text_prompt"].format.dt    = DataType::String;
    input["text_prompt"].format.shape = {static_cast<int64_t>(text_prompt.size())};

    std::string pending_text;
    int64_t num_generated_tokens = 0;
    GenerationStats generation_stats;

    std::cout << "Input prompt: " << text_prompt << std::endl;

    if (output_file.empty()) {
        std::cout << "Output text: ";
    } else {
        std::cout << "Output file: " << output_file << std::endl;
    }

    // Run the model inference
    while (!brick->is_generation_complete() && num_generated_tokens < max_generated_tokens) {
        const auto run_start    = std::chrono::steady_clock::now();
        TensorDict output       = brick->run(input);
        const auto run_end      = std::chrono::steady_clock::now();
        const auto out_text     = output["text"];
        const int64_t num_bytes = out_text.format.get_nbytes();

        const double run_time_s = std::chrono::duration<double>(run_end - run_start).count();

        // The first call includes prompt prefill; subsequent calls decode one token each.
        if (num_generated_tokens == 0) {
            generation_stats.prefill_time_s = run_time_s;
        } else {
            generation_stats.decode_time_s += run_time_s;
            ++generation_stats.num_decode_tokens;
        }

        if (num_bytes > 0) {
            const auto* chunk_ptr = static_cast<const char*>(out_text.data);

            pending_text.append(chunk_ptr, static_cast<size_t>(num_bytes));

            const std::string_view whitespace = " \t\r\n";
            const size_t last_whitespace      = pending_text.find_last_of(whitespace);

            if (last_whitespace != std::string::npos) {
                const size_t count = last_whitespace + 1;

                *generated_text_stream << pending_text.substr(0, count) << std::flush;
                pending_text.erase(0, count);
            }
        }

        ++num_generated_tokens;
    }

    // Display the left-over text
    if (!pending_text.empty()) {
        *generated_text_stream << pending_text;
    }

    // Display profiler run
    brick->write_profile(std::cout);

    if (!output_file.empty()) {
        output_file_stream.flush();

        COMPUTE_FLOW_CHECK(output_file_stream.good());
    } else {
        std::cout << std::endl;
    }

    std::cout << std::endl;

    return generation_stats;
}

GenerationStats run_multiple(std::string manifest_path, const std::string& thread_spec, bool enable_op_profiling,
                             const std::string& runs_file) {
    // Parse the number of CPU threads to use
    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "lm_text");

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

    const auto run_configs = internal::load_run_configs(runs_file, {"p", "n", "o"}, {"p"});
    GenerationStats total_generation_stats;

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];

        brick->reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string text_prompt = internal::get_run_value<std::string>(config, "p", "", run_index);
        const int64_t max_generated_tokens =
            internal::get_run_value<int64_t>(config, "n", std::numeric_limits<int64_t>::max(), run_index);
        const std::string output_file = internal::get_run_value<std::string>(config, "o", "", run_index);

        COMPUTE_FLOW_CHECK(!text_prompt.empty());
        COMPUTE_FLOW_CHECK(max_generated_tokens > 0);

        std::ofstream output_file_stream;
        std::ostream* generated_text_stream = &std::cout;

        if (!output_file.empty()) {
            output_file_stream.open(output_file, std::ios::binary);

            COMPUTE_FLOW_CHECK(output_file_stream.is_open());

            generated_text_stream = &output_file_stream;
        }

        TensorDict input;

        // Set the input
        input.declare_tensor("text_prompt");
        input["text_prompt"].data         = const_cast<char*>(text_prompt.data());
        input["text_prompt"].format.dt    = DataType::String;
        input["text_prompt"].format.shape = {static_cast<int64_t>(text_prompt.size())};

        std::string pending_text;
        int64_t num_generated_tokens = 0;
        GenerationStats generation_stats;

        std::cout << "Input prompt: " << text_prompt << std::endl;

        if (output_file.empty()) {
            std::cout << "Output text: ";
        } else {
            std::cout << "Output file: " << output_file << std::endl;
        }

        // Run the model inference
        while (!brick->is_generation_complete() && num_generated_tokens < max_generated_tokens) {
            const auto run_start    = std::chrono::steady_clock::now();
            TensorDict output       = brick->run(input);
            const auto run_end      = std::chrono::steady_clock::now();
            const auto out_text     = output["text"];
            const int64_t num_bytes = out_text.format.get_nbytes();

            const double run_time_s = std::chrono::duration<double>(run_end - run_start).count();

            // The first call includes prompt prefill; subsequent calls decode one token each.
            if (num_generated_tokens == 0) {
                generation_stats.prefill_time_s = run_time_s;
            } else {
                generation_stats.decode_time_s += run_time_s;
                ++generation_stats.num_decode_tokens;
            }

            if (num_bytes > 0) {
                const auto* chunk_ptr = static_cast<const char*>(out_text.data);

                pending_text.append(chunk_ptr, static_cast<size_t>(num_bytes));

                const std::string_view whitespace = " \t\r\n";
                const size_t last_whitespace      = pending_text.find_last_of(whitespace);

                if (last_whitespace != std::string::npos) {
                    const size_t count = last_whitespace + 1;

                    *generated_text_stream << pending_text.substr(0, count) << std::flush;
                    pending_text.erase(0, count);
                }
            }

            ++num_generated_tokens;
        }

        // Display the left-over text
        if (!pending_text.empty()) {
            *generated_text_stream << pending_text;
        }

        // Display profiler run
        brick->write_profile(std::cout);

        if (!output_file.empty()) {
            output_file_stream.flush();

            COMPUTE_FLOW_CHECK(output_file_stream.good());
        } else {
            std::cout << std::endl;
        }

        std::cout << std::endl;

        total_generation_stats.prefill_time_s += generation_stats.prefill_time_s;
        total_generation_stats.num_decode_tokens += generation_stats.num_decode_tokens;
        total_generation_stats.decode_time_s += generation_stats.decode_time_s;
    }

    return total_generation_stats;
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    // Parse the command-line arguments
    std::string manifest_path;
    std::string text_prompt;
    std::string thread_spec = "1";
    std::string output_file;
    std::string json_file;
    int64_t max_generated_tokens = std::numeric_limits<int64_t>::max();
    bool has_max_tokens_argument = false;
    bool enable_op_profiling     = false;
    static option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                    {"help", no_argument, nullptr, 'h'},
                                    {nullptr, 0, nullptr, 0}};

    int32_t option = 0;
    while ((option = ::getopt_long(argc, argv, "m:p:t:n:o:j:h", long_options, nullptr)) != -1) {
        switch (option) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'p':
                text_prompt = optarg;
                break;
            case 't':
                thread_spec = optarg;
                break;
            case 'n':
                max_generated_tokens    = static_cast<int64_t>(std::stoull(optarg));
                has_max_tokens_argument = true;
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
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Check the mandatory arguments
    if (manifest_path.empty() || (json_file.empty() && text_prompt.empty()) ||
        (!json_file.empty() && (!text_prompt.empty() || !output_file.empty() || has_max_tokens_argument))) {
        std::fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool multiple_runs_mode = !json_file.empty();

    const GenerationStats generation_stats =
        multiple_runs_mode
            ? run_multiple(manifest_path, thread_spec, enable_op_profiling, json_file)
            : run_single(manifest_path, thread_spec, enable_op_profiling, text_prompt, max_generated_tokens,
                         output_file);

    std::cout << std::fixed << std::setprecision(3);

    if (generation_stats.prefill_time_s > 0.0) {
        std::cout << "Prompt prefill time: " << generation_stats.prefill_time_s << "s" << std::endl;
    } else {
        std::cout << "Prompt prefill time: unavailable" << std::endl;
    }

    if (generation_stats.decode_time_s > 0.0) {
        const double tokens_per_second =
            static_cast<double>(generation_stats.num_decode_tokens) / generation_stats.decode_time_s;

        std::cout << "Decode tokens per second: " << tokens_per_second << std::endl;
    } else {
        std::cout << "Decode tokens per second: unavailable" << std::endl;
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::lm_text

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::lm_text::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
