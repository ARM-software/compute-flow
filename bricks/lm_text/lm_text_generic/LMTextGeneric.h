//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <cstdio>
#include <cstdlib>

#include "compute_flow/manifest.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/lm_runners/lm_runner.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow::bricks::lm_text {

class LMTextGeneric : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "lm_text.v1"; }
    ValidationResult validate(const ComputeFlowManifest& model) const override;
    bool is_generation_complete() const override;

protected:
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void warmup_impl(ProfilerView profiler) override;
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    void reset_states_impl(ProfilerView profiler) override;

    void allocate_memory_preprocessing(int64_t max_context_length);
    void allocate_memory_postprocessing(int64_t max_token_ids);

private:
    using LMRunnerPlanner = std::map<std::string, std::unique_ptr<ILMRunner>>;

    MemoryPlanner   _internal_memory;
    LMRunnerPlanner _runner;
    bool            _is_eog;
    bool            _is_first_run;
    std::string     _prompt_template;
};

}  // namespace compute_flow::bricks::lm_text
