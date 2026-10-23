//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::audio_tool_calling {
namespace {

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-a <audio.wav> | -j <runs.json>) [-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -a <audio.wav>         Mono WAV path or @sample/... selector.\n"
                 "  -j <runs.json>         Input JSON for multiple runs; each entry needs id and a.\n"
                 "  -t <thread_spec>       Default threads and optional model overrides (default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n"
                 "  Long aliases: --manifest (-m), --input (-a), --threads (-t), --help (-h).\n",
                 executable);
}

std::string get_calls_chunk(const TensorDict& output) {
    const TensorData& calls = output["calls"];

    COMPUTE_FLOW_CHECK(calls.format.dt == DataType::String && calls.format.shape.size() == 1);

    const int64_t num_bytes = calls.format.get_nbytes();

    COMPUTE_FLOW_CHECK(num_bytes >= 0);
    if (num_bytes == 0) {
        return {};
    }

    COMPUTE_FLOW_CHECK(calls.data != nullptr);

    return {static_cast<const char*>(calls.data), static_cast<std::size_t>(num_bytes)};
}

void validate_calls_v1(const nlohmann::json& calls) {
    if (!calls.is_array()) {
        throw std::runtime_error("Tool calls output must be a JSON array");
    }

    for (std::size_t index = 0; index < calls.size(); ++index) {
        const auto& call = calls[index];
        if (!call.is_object() || call.size() != 2 || !call.contains("name") || !call["name"].is_string() ||
            !call.contains("arguments") || !call["arguments"].is_object()) {
            throw std::runtime_error("Tool call at index " + std::to_string(index) +
                                     " must contain only a string 'name' and an object 'arguments'");
        }
    }
}

void generate_calls(IBrickV1& brick, const std::string& audio_path) {
    brick.reset_states();

    std::vector<float> audio;

    // Read audio file
    int64_t sample_rate = internal::read_mono_wav(audio_path, audio);

    TensorDict input;

    // Set the input
    input.declare_tensor("raw_wav").format     = {DataType::F32, {static_cast<int64_t>(audio.size())}};
    input["raw_wav"].data                      = audio.data();
    input.declare_tensor("sample_rate").format = {DataType::I64, {1}};
    input["sample_rate"].data                  = &sample_rate;

    // Run the model inference; JSON may be incomplete until generation finishes.
    std::string calls_json;

    std::cout << "Tool calls:" << std::endl;

    while (!brick.is_generation_complete()) {
        const TensorDict output = brick.run(input);
        const std::string chunk = get_calls_chunk(output);

        calls_json += chunk;
        std::cout << chunk << std::flush;
    }

    std::cout << std::endl;

    const auto calls = nlohmann::json::parse(calls_json);

    validate_calls_v1(calls);

    // Display profiler run
    brick.write_profile(std::cout);
}

void run_single(std::string manifest_path, const std::string& audio_value, const std::string& thread_spec,
                bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path               = internal::get_manifest_path(manifest_path, "audio_tool_calling");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    const std::string audio_path = internal::get_sample_path(audio_value, manifest_path);

    generate_calls(*brick, audio_path);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path               = internal::get_manifest_path(manifest_path, "audio_tool_calling");

    // Load the manifest
    auto manifest = load_manifest(manifest_path);

    // Configure default and per-model CPU thread counts
    internal::set_cpu_threads(manifest, default_threads, model_threads);

    // Load the brick
    auto brick = load_brick(manifest);

    // Initialize the brick
    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    // Setup the brick
    brick->setup(manifest, options);

    const auto run_configs = internal::load_run_configs(runs_file, {"a"}, {"a"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<std::size_t>(run_index)];

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string audio_path = internal::get_run_value<std::string>(config, "a", "", run_index);

        COMPUTE_FLOW_CHECK(!audio_path.empty());

        generate_calls(*brick, audio_path);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string audio_path;
    std::string json_file;
    std::string thread_spec = "1";
    bool has_audio_argument  = false;
    bool multiple_runs_mode  = false;
    bool enable_op_profiling = false;

    static option long_options[] = {
        {"manifest", required_argument, nullptr, 'm'}, {"input", required_argument, nullptr, 'a'},
        {"threads", required_argument, nullptr, 't'},  {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},           {nullptr, 0, nullptr, 0}};

    int32_t option_value = 0;
    while ((option_value = ::getopt_long(argc, argv, "m:a:j:t:h", long_options, nullptr)) != -1) {
        switch (option_value) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'a':
                audio_path         = optarg;
                has_audio_argument = true;
                break;
            case 'j':
                json_file          = optarg;
                multiple_runs_mode = true;
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

    if (optind != argc) {
        std::fprintf(stderr, "Unexpected positional argument: %s\n", argv[optind]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Check the mandatory arguments
    if (multiple_runs_mode && has_audio_argument) {
        std::fprintf(stderr, "Use either -a or -j.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (manifest_path.empty() || (multiple_runs_mode ? json_file.empty() : audio_path.empty())) {
        std::fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (multiple_runs_mode) {
        run_multiple(manifest_path, json_file, thread_spec, enable_op_profiling);
    } else {
        run_single(manifest_path, audio_path, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::audio_tool_calling

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::audio_tool_calling::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
