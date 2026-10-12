//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compute_flow/accelerator.h"
#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/manifest.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/profiler/profiler.h"
#include "compute_flow/runners/runner.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow::bricks::audio_detection {

class Perch : public BrickBaseV1 {
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
    static int64_t num_windows(int64_t num_samples, int64_t window_samples);

    void sort_scores(const TensorData& scores);

    std::unique_ptr<IRunner> _runner;
    MemoryPlanner _internal_memory;

    bool _is_output_ready = false;
    int64_t _idx_window = 0;
    int64_t _num_windows = 0;
    int64_t _num_samples_per_frame = 0;

    int64_t _sample_rate = 0;
    int64_t _num_classes = 0;
    int64_t _max_class_name_bytes = 0;

    std::vector<std::string> _class_names;
};

}  // namespace compute_flow::bricks::audio_detection
