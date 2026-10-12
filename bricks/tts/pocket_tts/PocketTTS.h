//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <memory>
#include <random>

#include "compute_flow/manifest.h"
#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/tensor_dict.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/lm_runners/lm_runner.h"

namespace compute_flow::bricks::tts {

class PocketTTS : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "tts.v1"; }
    ValidationResult validate(const ComputeFlowManifest& model) const override;
    bool is_generation_complete() const override;

protected:
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void warmup_impl(ProfilerView profiler) override;
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    void reset_states_impl(ProfilerView profiler) override;

private:
    void custom_op_fill_random_box_muller_transform_f32(const TensorData& dst);
    int64_t get_num_elems(const std::vector<int64_t>& values);
    TensorFormat get_flowlm_conditioning_format(int64_t sequence_length);
    void set_flowlm_conditioning(TensorData tensor);
    TensorData get_flowlm_output();
    TensorDict run_inference(const TensorDict& input);
    TensorDict run_inference_streaming(const TensorDict& input);

    using RunnerPlanner   = std::map<std::string, std::unique_ptr<IRunner>>;
    using LMRunnerPlanner = std::map<std::string, std::unique_ptr<ILMRunner>>;

    MemoryPlanner   _internal_memory;
    RunnerPlanner   _runner;
    LMRunnerPlanner _lm_runner;
    std::mt19937    _rng;

    // Vector to hold audio frames
    std::vector<float> _generated_audio_frames;

    int64_t _seed;
    int64_t _num_max_frames;
    int64_t _sample_rate;
    float   _rng_std;
    float   _rng_mean;
    float   _eos_threshold;
    bool    _enable_streaming_mode;

    bool    _eos_detected;
    bool    _is_output_ready;
    bool    _is_first_run_streaming_mode;
    bool    _pending_trailing_silence;
    int64_t _eos_frames_countdown;
    int64_t _fade_out_samples;
    float*  _last_audio_chunk_data;
    int64_t _last_audio_chunk_num_samples;
    bool    _last_audio_chunk_faded;

    };
}  // namespace compute_flow::bricks::tts
