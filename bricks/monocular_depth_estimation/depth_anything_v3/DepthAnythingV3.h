//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/manifest.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow::bricks::monocular_depth_estimation {

/**
 * @brief Depth Anything 3 monocular relative-depth estimation brick.
 */
class DepthAnythingV3 : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "monocular_depth_estimation.v1"; }
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
    std::unique_ptr<IRunner> _resize_input_runner;
    std::unique_ptr<IRunner> _depth_runner;
    std::unique_ptr<IRunner> _resize_output_runner;
    int64_t _maximum_dynamic_dimension = 0;
    int64_t _maximum_input_pixels      = 0;
    bool _is_output_ready              = false;
};

}  // namespace compute_flow::bricks::monocular_depth_estimation
