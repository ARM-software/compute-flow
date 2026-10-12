//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compute_flow/manifest.h"
#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/tensor_dict.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"

namespace compute_flow::bricks::tts {

class Kokoro : public BrickBaseV1 {
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
    /**
     * @brief Supported Kokoro language identifiers.
     */
    enum class KokoroLanguages {
        Chinese,
        English,
        French,
        Italian,
        Spanish,
        Unknown
    };

    int64_t custom_op_get_num_token_ids(const TensorData& phonemes);
    void custom_op_phonemes_to_token_ids(const TensorData& phonemes, TensorData& token_ids);
    void convert_prompt_to_phonemes_generic(const std::string& prompt, std::string& phonemes);
    void convert_prompt_to_phonemes_zh(const std::string& prompt, std::string& phonemes);
    KokoroLanguages kokoro_language_from_string(std::string language_str);

    using RunnerPlanner = std::map<std::string, std::unique_ptr<IRunner>>;

    MemoryPlanner                            _internal_memory;
    RunnerPlanner                            _runner;
    bool                                     _is_output_ready = false;
    int64_t                                  _max_tokens;
    KokoroLanguages                          _language;
    std::unordered_map<std::string, int64_t> _vocab_phonemes_ids;
};

}  // namespace compute_flow::bricks::tts
