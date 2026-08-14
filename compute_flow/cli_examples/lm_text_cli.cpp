//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/compute_flow.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/thread_config.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace compute_flow::bricks::lm_text {

static void print_usage(const char *name) {
    fprintf(stderr,
        "Usage: %s -m <manifest_path> -p <prompt> [-t <thread_spec>] [-n <max_generated_tokens>] [-o <output_file>]\n\n"
        "Options:\n"
        "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
        "  -p <prompt>               Input prompt text\n"
        "  -t <thread_spec>          Number of threads to use. Specify a default for all models and optionally override individual models, e.g. <default_num_threads>,<model_name0>:2,<model_name1>:4 (default: 1)\n"
        "  -n <max_generated_tokens> Maximum number of generated tokens (default=max int64)\n"
        "  -o <output_file>          Write generated text to this file instead of stdout\n"
        "  --enable-op-profiling     Enable per-op profiling (default: off)\n"
        "  -h                        Show this help message\n",
        name);
}

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string text_prompt;
    std::string thread_spec = "1";
    std::string output_file;
    int64_t max_generated_tokens = std::numeric_limits<int64_t>::max();
    bool enable_op_profiling = false;
    static struct option long_options[] = {
        {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int32_t opt;
    while ((opt = getopt_long(argc, argv, "m:p:t:n:o:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': manifest_path = optarg; break;
            case 'p': text_prompt = optarg; break;
            case 't': thread_spec = optarg; break;
            case 'n': max_generated_tokens = static_cast<int64_t>(std::stoull(optarg)); break;
            case 'o': output_file = optarg; break;
            case 1001: enable_op_profiling = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (manifest_path.empty() || text_prompt.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::ofstream output_file_stream;
    std::ostream* generated_text_stream = &std::cout;
    if (!output_file.empty()) {
        output_file_stream.open(output_file, std::ios::binary);
        if (!output_file_stream.is_open()) {
            std::cerr << "ERROR: Failed to open output file: " << output_file << std::endl;
            return EXIT_FAILURE;
        }
        generated_text_stream = &output_file_stream;
    }

    // Parse the number of CPU threads to use
    size_t default_threads = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "lm_text");

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
    input.declare_tensor("text_prompt");

    // Set the input
    input["text_prompt"].data         = (void *)text_prompt.c_str();
    input["text_prompt"].format.dt    = DataType::String;
    input["text_prompt"].format.shape = {(int64_t)text_prompt.size()};

    std::string pending_text;

    // Run the model inference
    int64_t num_generated_tokens = 0;

    std::cout << std::endl;
    std::cout << "Input prompt: " << text_prompt << std::endl;
    if (output_file.empty()) {
        std::cout << "Output text: ";
    } else {
        std::cout << "Output file: " << output_file << std::endl;
    }

    while (!brick->is_generation_complete() && num_generated_tokens < max_generated_tokens) {
        TensorDict output = brick->run(input);

        const auto out_text = output["text"];

        const int64_t nbytes = out_text.format.get_nbytes();

        if (nbytes > 0) {
            const auto* chunk_ptr = static_cast<const char*>(out_text.data);

            pending_text.append(chunk_ptr, static_cast<size_t>(nbytes));

            const std::string_view whitespace = " \t\r\n";
            const size_t last_ws = pending_text.find_last_of(whitespace);
            if (last_ws != std::string::npos) {
                const size_t count = last_ws + 1;
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

    if (!output_file.empty()) {
        output_file_stream.flush();
        if (!output_file_stream) {
            std::cerr << "ERROR: Failed to write output file: " << output_file << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Display profiler run
    brick->write_profile(std::cout);

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::lm_text

int32_t main(int32_t argc, char** argv) {
    return compute_flow::bricks::lm_text::run_cli(argc, argv);
}
