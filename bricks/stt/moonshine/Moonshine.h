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

namespace compute_flow::bricks::stt {

class Moonshine : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "stt.v1"; }
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
    void custom_op_sampler_max_row(const TensorData& logits, TensorData& ids);
    void custom_op_ids_to_text(const TensorData& ids, TensorData& text);

    using RunnerPlanner = std::map<std::string, std::unique_ptr<IRunner>>;

    MemoryPlanner            _internal_memory;
    RunnerPlanner            _runner;
    bool                     _is_output_ready = false;
    int32_t                  _max_tokens;
    int32_t                  _token_rate;
    int32_t                  _sample_rate;
    bool                     _force_use_kv_cache;
    int64_t                  _bos_token_id;
    int64_t                  _eos_token_id;
    int64_t                  _vocab_size;
    std::vector<std::string> _tokens_map;
    std::string              _output_text;
    std::vector<std::string> _tokens_vocab;
};

}  // namespace compute_flow::bricks::stt
