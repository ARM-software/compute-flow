//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/manifest.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow::bricks::stt {

class ParakeetTDT : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "stt.v1"; }
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
    using RunnerPlanner = std::map<std::string, std::unique_ptr<IRunner>>;

    void load_token_vocab(const std::string& path);
    void compute_global_feature_normalization(const TensorData& audio);
    int64_t prepare_featurizer_inputs(const TensorData& audio,
                                      int64_t start_sample,
                                      int64_t num_chunk_samples);
    void run_featurizer(const TensorData& audio, int64_t start_sample, int64_t num_chunk_samples);
    void compute_features(const TensorData& audio, int64_t start_sample, int64_t num_chunk_samples);
    void run_encoder();
    void start_encoder_decode();
    bool decode_next_text_piece();
    TensorData make_scalar_input_id(int64_t token_id);
    TensorData run_decoder_step(int64_t token_id);
    bool append_token_text(int64_t token_id);
    int64_t argmax(const float* values, int64_t count) const;

    MemoryPlanner _internal_memory;
    RunnerPlanner _runner;

    bool _is_output_ready = false;

    int32_t _sample_rate = 0;
    int32_t _feature_size = 0;
    int32_t _num_valid_feature_frames = 0;
    int32_t _hop_length = 0;
    int32_t _encoder_hidden_size = 0;
    int32_t _num_encoder_frames = 0;
    int32_t _max_symbols_per_step = 0;
    int32_t _streaming_chunk_samples = 0;
    int32_t _streaming_pre_encode_cache_frames = 0;
    int64_t _blank_token_id = 0;
    int64_t _vocab_size = 0;

    bool _stream_started = false;
    bool _has_emitted_text = false;
    bool _has_active_encoder_output = false;
    int64_t _stream_num_audio_samples = 0;
    int64_t _stream_sample_offset = 0;
    int64_t _current_num_valid_feature_frames = 0;
    int64_t _decode_frame_idx = 0;
    int64_t _decode_final_encoder_frame = 0;
    int32_t _decode_symbols_on_frame = 0;
    bool _global_feature_normalization_ready = false;
    TensorData _current_encoder_output;
    TensorData _current_input_features;

    std::vector<float> _feature_mean;
    std::vector<float> _feature_inv_std;
    std::vector<float> _pre_encode_feature_cache;
    std::vector<std::string> _tokens_vocab;
    std::string _output_chunk_text;
    std::string _pending_text;
};

}  // namespace compute_flow::bricks::stt
