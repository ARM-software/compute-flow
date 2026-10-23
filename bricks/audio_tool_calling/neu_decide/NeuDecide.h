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
#include <unordered_map>
#include <vector>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/runners/runner.h"
#include "internal/tool_call_grammar.h"

namespace compute_flow::bricks::audio_tool_calling {

/** @brief Generates structured tool calls from spoken requests. */
class NeuDecide : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "audio_tool_calling.v1"; }
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
    void load_tools(const ComputeFlowManifest& manifest);
    void encode_request(const TensorDict& input);
    TensorDict output();

    std::unordered_map<std::string, std::unique_ptr<IRunner>> _runners;
    MemoryPlanner _memory;
    std::vector<std::string> _token_bytes;
    std::vector<neu_decide::Tool> _tools;
    std::vector<int64_t> _candidates;
    std::string _text;
    int64_t _sample_rate       = 0;
    int64_t _max_audio_samples = 0;
    int64_t _max_answer_tokens = 0;
    int64_t _max_output_tokens = 0;
    int64_t _vocab_size        = 0;
    int64_t _eos_token         = 0;
    int64_t _tool_call_token   = 0;
    bool _complete             = false;
};

}  // namespace compute_flow::bricks::audio_tool_calling
