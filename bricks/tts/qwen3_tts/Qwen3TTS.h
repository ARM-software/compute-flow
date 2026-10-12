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

#include "compute_flow/manifest.h"
#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/tensor_dict.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/lm_runners/lm_runner.h"

namespace compute_flow::bricks::tts {

class Qwen3TTS: public BrickBaseV1 {

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
    enum class Qwen3TTSVariant {
        Base_0_6B,
        Base_ICL_0_6B,
        CustomVoice_0_6B,
        Unknown
    };

    ValidationResult validate_base(const ComputeFlowManifest& model) const;
    ValidationResult validate_custom_voice(const ComputeFlowManifest& model) const;
    void prepare_impl_base(const ComputeFlowManifest& manifest, ProfilerView profiler);
    void prepare_impl_custom_voice(const ComputeFlowManifest& manifest, ProfilerView profiler);
    TensorDict run_impl_base(const TensorDict& input, ProfilerView profiler);
    TensorDict run_impl_base_icl(const TensorDict& input, ProfilerView profiler);
    TensorDict run_impl_custom_voice(const TensorDict& input, ProfilerView profiler);
    void custom_op_argmax(const TensorData& t_logits, TensorData& t_token_id);
    void custom_op_in_place_add(const TensorData& t_lhs, TensorData& t_rhs_out);
    void run_text_embedding(const int32_t* ids, int64_t n, float* out);
    void run_codec_embedding(const int32_t* ids, int64_t n, float* out);
    void load_speaker_embedding_from_ref_audio(const std::string& path, TensorData& t_speaker_embedding);
    void icl_load_ref_codes_from_ref_audio_base(const std::string& path);
    void icl_precompute_embeddings_base(const std::string& ref_text);
    void icl_precompute_talker_prefix_state_base();
    void icl_precompute_speech_decoder_state_base(TensorData& t_codes,
                                                  TensorData t_audio_out,
                                                  int64_t max_output_audio_samples);
    void icl_trim_initial_audio_base(TensorData& t_audio_out);
    void run_code_predictor(const TensorData& t_last_talker_hidden, TensorData& t_codes, TensorData& t_step_embd);
    void run_speech_decoder(const TensorData& t_codes, int64_t num_frames_processed, TensorData& t_audio_out, int64_t max_output_audio_samples, bool en_streaming_mode);
    void add_special_embedding(const float* special_emb, int64_t num_tokens, int64_t hidden, float* out);

    using RunnerPlanner   = std::map<std::string, std::unique_ptr<IRunner>>;
    using LMRunnerPlanner = std::map<std::string, std::unique_ptr<ILMRunner>>;

    MemoryPlanner   _internal_memory;
    RunnerPlanner   _runner;
    LMRunnerPlanner _lm_runner;

    Qwen3TTSVariant _qwen3_tts_variant;

    // Token IDs (loaded from manifest metadata).
    int64_t _k_n_embd_sz;
    int32_t _tts_pad_token_id;
    int32_t _tts_bos_token_id;
    int32_t _tts_eos_token_id;
    int32_t _codec_eos_token_id;
    int32_t _codec_bos_token_id;
    int32_t _codec_pad_token_id;
    int32_t _codec_think_token_id;
    int32_t _codec_no_think_token_id;
    int32_t _codec_think_bos_token_id;
    int32_t _codec_think_eos_token_id;
    int32_t _im_start_token_id;
    int32_t _im_end_token_id;
    int32_t _assistant_token_id;
    int32_t _newline_token_id;
    int32_t _lang_token_id;
    int32_t _spk_token_id;
    int32_t _num_max_frames;
    int32_t _num_max_frames_speech_dec;
    int32_t _num_audio_samples_per_frame;
    int32_t _num_code_pred_groups;
    int32_t _num_active_code_pred_groups;
    int64_t _sample_rate;
    int32_t _min_ref_audio_duration_sec;
    int32_t _max_ref_audio_duration_sec;
    int64_t _ref_text_token_len;
    int64_t _num_ref_code_frames;
    int64_t _icl_prefill_prefix_len;
    int64_t _icl_prefill_suffix_len;
    int64_t _num_generated_frames;
    bool    _enable_streaming_mode;
    bool    _is_output_ready;
    bool    _is_first_run_streaming_mode;

    // Speech decoder states
    TensorData _t_state_pre_conv;
    TensorData _t_state_latent;
};

} // namespace compute_flow::bricks::tts
