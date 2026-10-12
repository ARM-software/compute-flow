//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "DepthAnythingV3.h"

#include <cstdint>
#include <vector>

namespace compute_flow::bricks::monocular_depth_estimation {
namespace {

constexpr int32_t k_brick_major_version = 1;
constexpr int64_t k_num_channels        = 3;
constexpr int64_t k_num_depth_channels  = 1;
constexpr int64_t k_warmup_height       = 1;
constexpr int64_t k_warmup_width        = 1;

}  // namespace

ValidationResult DepthAnythingV3::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "monocular_depth_estimation");
    COMPUTE_FLOW_VALIDATE(info.name == "DepthAnythingV3");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("maximum_dynamic_dimension"));
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("maximum_dynamic_dimension") > 0);
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("maximum_input_pixels"));
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("maximum_input_pixels") > 0);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 3);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("resize_input"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("depth_anything_v3"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("resize_output"));
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].framework_type == RunnerType::ExecuTorch);
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].has_input("image"));
    COMPUTE_FLOW_VALIDATE(manifest["resize_input"].has_output("image"));
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].has_input("image"));
    COMPUTE_FLOW_VALIDATE(manifest["depth_anything_v3"].has_output("relative_depth"));
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].has_input("relative_depth"));
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].has_input("image"));
    COMPUTE_FLOW_VALIDATE(manifest["resize_output"].has_output("relative_depth"));
    return {};
}

void DepthAnythingV3::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _resize_input_runner = make_runner(manifest["resize_input"].framework_type);
    _resize_input_runner->setup(manifest["resize_input"], profiler);
    _depth_runner = make_runner(manifest["depth_anything_v3"].framework_type);
    _depth_runner->setup(manifest["depth_anything_v3"], profiler);
    _resize_output_runner = make_runner(manifest["resize_output"].framework_type);
    _resize_output_runner->setup(manifest["resize_output"], profiler);
}

void DepthAnythingV3::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _maximum_dynamic_dimension = manifest.get_constant<int64_t>("maximum_dynamic_dimension");
    _maximum_input_pixels      = manifest.get_constant<int64_t>("maximum_input_pixels");

    const TensorFormat resized_input_format = _resize_input_runner->get_output_format("image");
    COMPUTE_FLOW_ASSERT(resized_input_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(resized_input_format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(resized_input_format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(resized_input_format.shape[1] == k_num_channels);
    COMPUTE_FLOW_ASSERT(resized_input_format.shape[2] > 0);
    COMPUTE_FLOW_ASSERT(resized_input_format.shape[3] > 0);

    const TensorFormat image_format = _depth_runner->get_input_format("image");
    COMPUTE_FLOW_ASSERT(image_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(image_format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(image_format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(image_format.shape == resized_input_format.shape);

    const TensorFormat relative_depth_format = _depth_runner->get_output_format("relative_depth");
    COMPUTE_FLOW_ASSERT(relative_depth_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(relative_depth_format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(relative_depth_format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(relative_depth_format.shape[1] == k_num_depth_channels);
    COMPUTE_FLOW_ASSERT(relative_depth_format.shape[2] > 0);
    COMPUTE_FLOW_ASSERT(relative_depth_format.shape[3] > 0);

    const TensorFormat resize_output_depth_format = _resize_output_runner->get_input_format("relative_depth");
    COMPUTE_FLOW_ASSERT(resize_output_depth_format.dt == relative_depth_format.dt);
    COMPUTE_FLOW_ASSERT(resize_output_depth_format.shape == relative_depth_format.shape);

    const TensorFormat resize_output_image_format = _resize_output_runner->get_input_format("image");
    COMPUTE_FLOW_ASSERT(resize_output_image_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(resize_output_image_format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(resize_output_image_format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(resize_output_image_format.shape[1] == k_num_channels);

    const TensorFormat resized_output_format = _resize_output_runner->get_output_format("relative_depth");
    COMPUTE_FLOW_ASSERT(resized_output_format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(resized_output_format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(resized_output_format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(resized_output_format.shape[1] == k_num_depth_channels);
}

void DepthAnythingV3::warmup_impl(ProfilerView profiler) {
    std::vector<float> image(k_num_channels * k_warmup_height * k_warmup_width);

    TensorDict input;
    input.declare_tensor("image").format =
        TensorFormat{DataType::F32, {k_num_channels, k_warmup_height, k_warmup_width}};
    input["image"].data = image.data();

    (void)run_impl(input, profiler);
    reset_states_impl(profiler);
}

void DepthAnythingV3::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
}

void DepthAnythingV3::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    if (_resize_input_runner) {
        _resize_input_runner->reset_states();
    }
    if (_depth_runner) {
        _depth_runner->reset_states();
    }
    if (_resize_output_runner) {
        _resize_output_runner->reset_states();
    }
    _is_output_ready = false;
}

bool DepthAnythingV3::is_generation_complete() const { return _is_output_ready; }

TensorDict DepthAnythingV3::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;
    const TensorData& source_image = input["image"];
    COMPUTE_FLOW_ASSERT(source_image.data != nullptr);
    COMPUTE_FLOW_ASSERT(source_image.format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(source_image.format.shape.size() == 3);
    COMPUTE_FLOW_ASSERT(source_image.format.shape[0] == k_num_channels);

    const int64_t source_height = source_image.format.shape[1];
    const int64_t source_width  = source_image.format.shape[2];
    COMPUTE_FLOW_CHECK(source_height > 0 && source_height <= _maximum_dynamic_dimension);
    COMPUTE_FLOW_CHECK(source_width > 0 && source_width <= _maximum_dynamic_dimension);
    COMPUTE_FLOW_CHECK(source_height <= _maximum_input_pixels / source_width);

    TensorData source_model_image;
    source_model_image.format      = {DataType::F32, {1, k_num_channels, source_height, source_width}};
    source_model_image.data        = source_image.data;
    source_model_image.user_hosted = true;

    const TensorFormat depth_image_format = _depth_runner->get_input_format("image");
    const bool requires_resize =
        source_height != depth_image_format.shape[2] || source_width != depth_image_format.shape[3];

    TensorData depth_image = source_model_image;
    if (requires_resize) {
        _resize_input_runner->set_input("image", source_model_image);
        _resize_input_runner->run();
        depth_image = _resize_input_runner->get_output("image");
        COMPUTE_FLOW_ASSERT(depth_image.data != nullptr);
        COMPUTE_FLOW_ASSERT(depth_image.format == depth_image_format);
    }

    _depth_runner->set_input("image", depth_image);
    _depth_runner->run();

    const TensorData model_relative_depth = _depth_runner->get_output("relative_depth");
    COMPUTE_FLOW_ASSERT(model_relative_depth.data != nullptr);
    COMPUTE_FLOW_ASSERT(model_relative_depth.format == _resize_output_runner->get_input_format("relative_depth"));

    TensorData relative_depth = model_relative_depth;
    if (requires_resize) {
        _resize_output_runner->set_input("relative_depth", model_relative_depth);
        _resize_output_runner->set_input("image", source_model_image);
        _resize_output_runner->run();
        relative_depth = _resize_output_runner->get_output("relative_depth");
    }
    COMPUTE_FLOW_ASSERT(relative_depth.data != nullptr);
    COMPUTE_FLOW_ASSERT(relative_depth.format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(relative_depth.format.shape.size() == 4);
    COMPUTE_FLOW_ASSERT(relative_depth.format.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(relative_depth.format.shape[1] == k_num_depth_channels);
    COMPUTE_FLOW_ASSERT(relative_depth.format.shape[2] == source_height);
    COMPUTE_FLOW_ASSERT(relative_depth.format.shape[3] == source_width);

    TensorDict output;
    output.declare_tensor("relative_depth") = relative_depth;
    output["relative_depth"].format.shape   = {source_height, source_width};
    _is_output_ready                        = true;
    return output;
}

}  // namespace compute_flow::bricks::monocular_depth_estimation
