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

namespace compute_flow::bricks::object_detection {

class Yolo : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "object_detection.v1"; }
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
    using RunnerPlanner = std::map<std::string, std::unique_ptr<IRunner>>;

    MemoryPlanner _internal_memory;
    RunnerPlanner _runner;

    bool                     _is_output_ready;
    float                    _confidence_thr;
    float                    _nms_thr;
    int64_t                  _max_detections_per_image;
    int64_t                  _max_str_in_bytes;
    int64_t                  _version;
    bool                     _is_nms_exported;
    bool                     _are_coordinates_normalized;
    std::vector<std::string> _class_names;

    std::vector<float>   _out_boxes;
    std::vector<float>   _out_scores;
    std::vector<int64_t> _out_class_ids;
    std::vector<uint8_t> _out_class_names;
    std::vector<int64_t> _src_model_shape;

    /**
     * @brief Resize a CHW (or NCHW with N=1) float32 image tensor using bilinear filtering.
     *
     * This matches the behavior of `PIL.Image.resize(..., Image.BILINEAR)`:
     * the source is directly resized to the destination width/height with independent
     * X/Y scaling (aspect ratio is not preserved), and no cropping or padding is applied.
     */
    void custom_op_scale_crop_bilinear_chw(const TensorData& src, TensorData& dst);
    void custom_op_postprocessing_v5su(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids);
    void custom_op_postprocessing_v5s(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids);
    void custom_op_postprocessing_v26(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids);
};

}  // namespace compute_flow::bricks::object_detection
