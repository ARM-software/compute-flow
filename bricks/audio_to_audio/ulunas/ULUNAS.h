//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/manifest.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/tensor_dict.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace compute_flow::bricks::audio_to_audio {

class ULUNAS : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "audio_to_audio.v1"; }
    ValidationResult validate(const ComputeFlowManifest& manifest) const override;
    bool is_generation_complete() const override;

protected:
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void warmup_impl(ProfilerView profiler) override;
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    void reset_states_impl(ProfilerView profiler) override;

private:
    int64_t num_stft_frames(int64_t num_samples) const;
    void resize_audio_buffers(int64_t num_samples, int64_t num_frames);
    void prepare_model_frame(float* frame,
                             const float* audio,
                             int64_t num_samples,
                             int64_t frame_idx);
    void append_enhanced_frame(const float* enhanced_frame, int64_t frame_idx);
    void finalize_output_audio(int64_t num_samples);

    std::unique_ptr<IRunner> _runner;
    MemoryPlanner _internal_memory;

    bool _is_output_ready = false;
    int64_t _sample_rate = 0;
    int64_t _n_fft = 0;
    int64_t _hop_length = 0;
    int64_t _win_length = 0;
    int64_t _output_sample_rate = 0;
    int64_t _model_chunk_size = 1;

    std::vector<float> _window;
    std::vector<float> _output_audio;
    std::vector<float> _overlap_audio;
    std::vector<float> _window_square_sum;
};

}  // namespace compute_flow::bricks::audio_to_audio
