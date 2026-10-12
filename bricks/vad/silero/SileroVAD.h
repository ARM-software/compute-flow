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
#include <memory>
#include "compute_flow/manifest.h"
#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/tensor_dict.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"

namespace compute_flow::bricks::vad {

class SileroVAD : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "audio_binary_classification.v1"; }
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

    int64_t get_num_output_frames(int64_t total_input_samples) const;

    MemoryPlanner  _internal_memory;
    RunnerPlanner  _runner;
    bool           _is_output_ready = false;
    int64_t        _frame_sz;
    int64_t        _context_sz;
};

} // namespace compute_flow::bricks::vad
