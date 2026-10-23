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
#include "internal/log_mel.h"

namespace compute_flow::bricks::stt {

/**
 * @brief Transcribe a complete waveform incrementally with cache-aware RNN-T models.
 *
 * See IBrickV1 and BrickBaseV1 for the shared interface and lifecycle documentation,
 * and brick_apis/stt.v1.json for the input/output contract. The comments here
 * describe Nemotron-specific behavior and requirements not covered there.
 *
 * Nemotron 3.5 requires a language map and an encoder prompt input. The manifest
 * selects the model input mappings, frontend constants, and tokenizer. Loaded
 * tensors define the tensor dimensions.
 * A run advances one encoder chunk and returns append-only UTF-8 text.
 *
 * Reset after inference errors; model and tokenizer replacement requires setup().
 * One instance owns one stream and must not be called concurrently.
 */
class NemotronASR : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "stt.v1"; }
    /**
     * @brief Check manifest wiring and required frontend constants without loading models.
     * Loaded tensor shapes and dtypes are checked in prepare_impl(), after runner setup.
     */
    ValidationResult validate(const ComputeFlowManifest& manifest) const override;
    /** @brief Report whether the last run consumed the final chunk, including empty input. */
    bool is_generation_complete() const override;

protected:
    /** @brief Load three inference runners and the matching SentencePiece decoder. */
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    /** @brief Check loaded tensor contracts, allocate reusable scratch, and apply configs. */
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    /** @brief Run one silence chunk through the frontend and decoder. */
    void warmup_impl(ProfilerView profiler) override;
    /** @brief Resolve the language prompt and apply the decoding cap. */
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    /**
     * @brief Advance one encoder chunk and return only newly stable text.
     * @param[in] input stt.v1 tensors with finite raw_wav samples ([N] or [1,N])
     * and sample_rate matching the selected package.
     * @return Runner-independent, brick-owned UTF-8 text, possibly empty. Copy it
     * before another run(), reset_states(), setup(), or destruction.
     * @throws std::runtime_error On invalid input, inference failure, or symbol-cap
     * exhaustion. Reset before reuse; partial inference state is not rolled back.
     */
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    /** @brief Clear runner caches, frontend overlap, transcript, and completion state. */
    void reset_states_impl(ProfilerView profiler) override;

private:
    /** @brief Advance predictor feedback once, initially with blank and then with each token. */
    void run_predictor(int64_t token);
    /** @brief Greedily consume valid encoder frames; a blank leaves predictor state unchanged. */
    void decode_frames();
    /** @brief Decode the full history and withhold any suffix that later tokens could revise. */
    void update_text();

    std::unique_ptr<IRunner> _encoder;
    std::unique_ptr<IRunner> _predictor;
    std::unique_ptr<IRunner> _joiner;
    std::unique_ptr<IRunner> _detokenizer;
    std::unique_ptr<nemotron_asr::LogMel> _frontend;
    MemoryPlanner _memory;
    int64_t _sample_rate           = 0;
    int64_t _hop_samples           = 0;
    int64_t _mel_bins              = 0;
    int64_t _fresh_frames          = 0;
    int64_t _overlap_frames        = 0;
    int64_t _input_frames          = 0;
    int64_t _encoder_dimension     = 0;
    int64_t _blank_token_id        = 0;
    int64_t _max_symbols_per_frame = 10;
    int64_t _prompt_id             = 0;
    int64_t _frame_offset          = 0;
    int64_t _num_samples           = 0;
    bool _started                  = false;
    bool _complete                 = false;
    std::vector<int64_t> _tokens;
    std::string _emitted_text;
    std::string _output_text;
};

}  // namespace compute_flow::bricks::stt
