//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/runners/runner.h"

namespace compute_flow::bricks::audio_detection {

/** @brief Assign recording-local speaker IDs to overlapping speech intervals. */
class Nemotron3Diarization : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "audio_detection.v1"; }
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
    struct Segment {
        int64_t start;
        int64_t end;
        int64_t speaker;
        double score;
    };

    void run_window(const float* audio, int64_t num_samples, bool update_cache);
    void process_window(const float* audio, int64_t num_samples, int64_t num_output_frames, bool is_last_window);
    TensorDict output_segments();

    std::unique_ptr<IRunner> _frontend;
    std::unique_ptr<IRunner> _encoder;
    std::unique_ptr<IRunner> _speaker_cache;
    MemoryPlanner _memory;
    std::vector<int64_t> _open_segments;
    std::vector<Segment> _segments;
    std::vector<double> _score_sums;
    std::vector<std::string> _speaker_names;
    int64_t _max_name_bytes   = 0;
    int64_t _sample_rate      = 0;
    int64_t _hop              = 0;
    int64_t _fft_length       = 0;
    int64_t _factor           = 0;
    int64_t _chunk_length     = 0;
    int64_t _window_length    = 0;
    int64_t _cache_length     = 0;
    int64_t _context_length   = 0;
    int64_t _encoder_length   = 0;
    int64_t _max_chunk_length = 0;
    int64_t _hidden_size      = 0;
    int64_t _num_speakers     = 0;
    int64_t _window_samples   = 0;
    int64_t _num_recording_samples = -1;
    int64_t _frame_offset     = 0;
    int64_t _num_context      = 0;
    float _threshold          = 0.5F;
    bool _en_streaming_mode   = false;
    bool _complete            = false;
};

}  // namespace compute_flow::bricks::audio_detection
