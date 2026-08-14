//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/cli_examples/internal/run_config.h"
#include "compute_flow/cli_examples/internal/thread_config.h"
#include "compute_flow/cli_examples/internal/wav.h"
#include "compute_flow/compute_flow.h"

namespace compute_flow::bricks::audio_enhancement {
namespace {

void print_usage(const char* exec) {
    fprintf(stderr,
            "Usage: %s -m <manifest_path> (-a <audio.wav> | -j <runs.json>) [-t <thread_spec>] "
            "[-o <output.wav>]\n\n"
            "Options:\n"
            "  -m <manifest_path>        Manifest path or @manifest/... selector\n"
            "  -a <audio.wav>            Input WAV path or @sample/... selector.\n"
            "  -j <runs.json>            Input JSON file path for multiple run settings.\n"
            "  -t <thread_spec>          Default threads and optional per-model overrides, e.g. 2,enhancer:4 (default: "
            "1).\n"
            "  -o <output.wav>           Output enhanced mono float32 WAV file.\n"
            "  --enable-op-profiling     Enable per-op profiling output.\n"
            "  -h                        Show this help message and exit.\n",
            exec);
}

std::string default_output_path(const std::string& input_path) {
    const size_t dot = input_path.find_last_of('.');
    if (dot == std::string::npos) {
        return input_path + "_enhanced.wav";
    }

    return input_path.substr(0, dot) + "_enhanced.wav";
}

void run_single(IBrickV1& brick, const std::string& manifest_path, const std::string& audio_value,
                const std::optional<std::string>& output_path) {
    const std::string audio_file = internal::get_sample_path(audio_value, manifest_path);
    std::vector<float> mono_raw_audio;

    // Read audio file
    const int64_t sample_rate = internal::read_mono_wav(audio_file, mono_raw_audio);

    TensorDict input;

    // Set the input
    input.declare_tensor("raw_wav");
    input.declare_tensor("sample_rate");
    input["raw_wav"].data             = mono_raw_audio.data();
    input["raw_wav"].format.dt        = DataType::F32;
    input["raw_wav"].format.shape     = {static_cast<int64_t>(mono_raw_audio.size())};
    input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
    input["sample_rate"].format.dt    = DataType::I64;
    input["sample_rate"].format.shape = {1};

    TensorDict output;

    // Run the model inference
    while (!brick.is_generation_complete()) {
        output = brick.run(input);
    }

    if (output_path.has_value()) {
        const std::string default_input =
            audio_value == audio_file ? audio_file : std::filesystem::path(audio_file).filename().string();
        const std::string path             = output_path->empty() ? default_output_path(default_input) : *output_path;
        const TensorData& enhanced_wav     = output["enhanced_wav"];
        const TensorData& output_sr        = output["sample_rate"];
        const int64_t enhanced_sample_rate = reinterpret_cast<const int64_t*>(output_sr.data)[0];
        internal::write_mono_wav(path, reinterpret_cast<const float*>(enhanced_wav.data), enhanced_wav.format.shape[0],
                                 enhanced_sample_rate);
        std::cout << "Wrote enhanced audio to " << path << std::endl;
    }

    // Display profiler run
    brick.write_profile(std::cout);
}

void run_multiple(IBrickV1& brick, const std::string& runs_file) {
    const auto run_configs = internal::load_run_configs(runs_file, {"a", "o"}, {"a"});
    for (int64_t run_index = 0; run_index < static_cast<int64_t>(run_configs.size()); ++run_index) {
        const auto& config = run_configs[static_cast<size_t>(run_index)];
        brick.reset_states();

        const std::string run_id =
            internal::get_run_value<std::string>(config, "id", std::to_string(run_index + 1), run_index);
        std::cout << "Run " << run_id << std::endl;
        const std::optional<std::string> output_path =
            config.contains("o") ? std::make_optional(internal::get_run_value<std::string>(config, "o", "", run_index))
                                 : std::nullopt;
        const std::string audio_file = internal::get_run_value<std::string>(config, "a", "", run_index);

        COMPUTE_FLOW_CHECK(!audio_file.empty());

        if (output_path.has_value()) {
            COMPUTE_FLOW_CHECK(!output_path->empty());
        }

        std::vector<float> mono_raw_audio;

        // Read audio file
        const int64_t sample_rate = internal::read_mono_wav(audio_file, mono_raw_audio);

        TensorDict input;

        // Set the input
        input.declare_tensor("raw_wav");
        input.declare_tensor("sample_rate");
        input["raw_wav"].data             = mono_raw_audio.data();
        input["raw_wav"].format.dt        = DataType::F32;
        input["raw_wav"].format.shape     = {static_cast<int64_t>(mono_raw_audio.size())};
        input["sample_rate"].data         = const_cast<int64_t*>(&sample_rate);
        input["sample_rate"].format.dt    = DataType::I64;
        input["sample_rate"].format.shape = {1};

        TensorDict output;

        // Run the model inference
        while (!brick.is_generation_complete()) {
            output = brick.run(input);
        }

        if (output_path.has_value()) {
            const std::string path             = *output_path;
            const TensorData& enhanced_wav     = output["enhanced_wav"];
            const TensorData& output_sr        = output["sample_rate"];
            const int64_t enhanced_sample_rate = reinterpret_cast<const int64_t*>(output_sr.data)[0];
            internal::write_mono_wav(path, reinterpret_cast<const float*>(enhanced_wav.data),
                                     enhanced_wav.format.shape[0], enhanced_sample_rate);
            std::cout << "Wrote enhanced audio to " << path << std::endl;
        }

        // Display profiler run
        brick.write_profile(std::cout);
    }
}

}  // namespace

int32_t run_cli(int32_t argc, char** argv) {
    std::string manifest_path;
    std::string audio_path;
    std::string output_file;
    std::string json_file;
    std::string thread_spec  = "1";
    bool enable_op_profiling = false;

    static struct option long_options[] = {{"enable-op-profiling", no_argument, nullptr, 1001},
                                           {"help", no_argument, nullptr, 'h'},
                                           {nullptr, 0, nullptr, 0}};

    int opt = 0;
    while ((opt = ::getopt_long(argc, argv, "m:a:t:o:j:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                manifest_path = optarg;
                break;
            case 'a':
                audio_path = optarg;
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

    if (manifest_path.empty() || (json_file.empty() && audio_path.empty()) ||
        (!json_file.empty() && (!audio_path.empty() || !output_file.empty()))) {
        fprintf(stderr, "Missing required arguments.\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    size_t default_threads   = 1;
    const auto model_threads = internal::parse_hint_cpu_threads(thread_spec, default_threads);

    manifest_path = compute_flow::bricks::internal::get_manifest_path(manifest_path, "audio_enhancement");
    auto manifest = compute_flow::bricks::load_manifest(manifest_path);

    internal::set_cpu_threads(manifest, default_threads, model_threads);

    auto brick = compute_flow::bricks::load_brick(manifest);

    BrickOptions options;
    options.runtime.enable_warmup         = true;
    options.profiling.enable_op_profiling = enable_op_profiling;

    brick->setup(manifest, options);

    const bool multiple_runs_mode = !json_file.empty();
    if (multiple_runs_mode) {
        run_multiple(*brick, json_file);
    } else {
        brick->reset_states();
        run_single(*brick, manifest_path, audio_path, output_file);
    }

    return EXIT_SUCCESS;
}

}  // namespace compute_flow::bricks::audio_enhancement

int main(int argc, char** argv) { return compute_flow::bricks::audio_enhancement::run_cli(argc, argv); }
