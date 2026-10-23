//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::audio_classification {
namespace {

constexpr int64_t k_default_top_k = 5;

void print_usage(const char* executable) {
    std::fprintf(stderr,
                 "Usage: %s -m <manifest.json> (-a <audio.wav> [-k <top_k>] | -j <runs.json>) "
                 "[-t <thread_spec>]\n\n"
                 "Options:\n"
                 "  -m <manifest.json>     Manifest path or @manifest/... selector.\n"
                 "  -a <audio.wav>         Mono WAV path or @sample/... selector.\n"
                 "  -j <runs.json>         Input JSON for multiple runs; each entry needs id and a, and may set k.\n"
                 "  -k <top_k>             Maximum classes to display, capped at available results (default: 5).\n"
                 "  -t <thread_spec>       Default threads and optional model overrides (default: 1).\n"
                 "  --enable-op-profiling  Enable per-operation profiling.\n"
                 "  -h                     Show this help message.\n"
                 "  Long aliases: --manifest (-m), --input (-a), --threads (-t), --help (-h).\n",
                 executable);
}

void display_result(const TensorDict& output, int64_t top_k) {
    const TensorData& ids    = output["id"];
    const TensorData& scores = output["score"];
    const TensorData& names  = output["name"];

    COMPUTE_FLOW_CHECK(ids.data != nullptr && ids.format.dt == DataType::I64 && ids.format.shape.size() == 1 &&
                       ids.format.shape[0] > 0);
    COMPUTE_FLOW_CHECK(scores.data != nullptr && scores.format.dt == DataType::F32 &&
                       scores.format.shape == ids.format.shape);
    COMPUTE_FLOW_CHECK(names.data != nullptr && names.format.dt == DataType::U8 && names.format.shape.size() == 2 &&
                       names.format.shape[0] == ids.format.shape[0] && names.format.shape[1] > 0);

    const int64_t num_classes            = ids.format.shape[0];
    const int64_t max_name_length        = names.format.shape[1];
    const int64_t num_classes_to_display = std::min(top_k, num_classes);
    const int64_t* class_ids             = static_cast<const int64_t*>(ids.data);
    const float* class_scores            = static_cast<const float*>(scores.data);
    const char* class_names              = static_cast<const char*>(names.data);

    for (int64_t index = 0; index < num_classes_to_display; ++index) {
        const char* name_start = class_names + index * max_name_length;
        const char* name_end   = std::find(name_start, name_start + max_name_length, '\0');
        const std::string name(name_start, name_end);

        std::printf("%lld. %s (id %lld, score %.4f)\n", static_cast<long long>(index + 1), name.c_str(),
                    static_cast<long long>(class_ids[index]), class_scores[index]);
    }
}

void classify(IBrickV1& brick, const std::string& audio_path, int64_t top_k) {
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

    TensorDict output;

    // Run the model inference
    while (!brick.is_generation_complete()) {
        output = brick.run(input);
    }

    display_result(output, top_k);

    // Display profiler run
    brick.write_profile(std::cout);
}

void run_single(std::string manifest_path, const std::string& audio_value, int64_t top_k,
                const std::string& thread_spec, bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path               = internal::get_manifest_path(manifest_path, "audio_classification");

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

    classify(*brick, audio_path, top_k);
}

void run_multiple(std::string manifest_path, const std::string& runs_file, const std::string& thread_spec,
                  bool enable_op_profiling) {
    // Parse the number of CPU threads to use
    std::size_t default_threads = 1;
    const auto model_threads    = internal::parse_hint_cpu_threads(thread_spec, default_threads);
    manifest_path               = internal::get_manifest_path(manifest_path, "audio_classification");

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

    const auto run_configs = internal::load_run_configs(runs_file, {"a", "k"}, {"a"});

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<std::size_t>(run_index)];

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::string audio_path = internal::get_run_value<std::string>(config, "a", "", run_index);

        COMPUTE_FLOW_CHECK(!audio_path.empty());

        // Validate before conversion: JSON numeric casts can truncate fractions or overflow.
        if (config.contains("k")) {
            const auto& value = config.at("k");

            COMPUTE_FLOW_CHECK(value.is_number_integer());
            COMPUTE_FLOW_CHECK(value > 0 && value <= std::numeric_limits<int64_t>::max());
        }

        const int64_t top_k = internal::get_run_value<int64_t>(config, "k", k_default_top_k, run_index);

        classify(*brick, audio_path, top_k);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string audio_path;
    std::string json_file;
    std::string thread_spec  = "1";
    int64_t top_k            = k_default_top_k;
    bool has_top_k_argument  = false;
    bool has_audio_argument  = false;
    bool multiple_runs_mode  = false;
    bool enable_op_profiling = false;

    static option long_options[] = {
        {"manifest", required_argument, nullptr, 'm'}, {"input", required_argument, nullptr, 'a'},
        {"threads", required_argument, nullptr, 't'},  {"enable-op-profiling", no_argument, nullptr, 1001},
        {"help", no_argument, nullptr, 'h'},           {nullptr, 0, nullptr, 0}};

    int32_t option_value = 0;
    while ((option_value = ::getopt_long(argc, argv, "m:a:j:k:t:h", long_options, nullptr)) != -1) {
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
            case 'k': {
                const char* end   = optarg + std::char_traits<char>::length(optarg);
                const auto result = std::from_chars(optarg, end, top_k);

                if (result.ec != std::errc{} || result.ptr != end || top_k <= 0) {
                    std::fprintf(stderr, "-k must be a positive integer.\n");
                    return EXIT_FAILURE;
                }

                has_top_k_argument = true;
                break;
            }
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
    if (multiple_runs_mode && (has_audio_argument || has_top_k_argument)) {
        std::fprintf(stderr, "Use either -a [-k] or -j.\n\n");
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
        run_single(manifest_path, audio_path, top_k, thread_spec, enable_op_profiling);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::audio_classification

int32_t main(int32_t argc, char** argv) {
    try {
        return compute_flow::bricks::audio_classification::run_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
