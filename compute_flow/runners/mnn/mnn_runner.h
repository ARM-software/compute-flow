//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef COMPUTE_FLOW_HAS_MNN

#include "compute_flow/model_data.h"
#include "compute_flow/profiler/profiler_view.h"
#include "compute_flow/tensor_dict.h"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace compute_flow {

class MNNRunner {
public:
    MNNRunner() = default;

    void setup(const ModelData& model, ProfilerView profiler = {});
    const std::string& model_name() const;

    void map_input(std::string_view key, size_t idx);
    void map_input(std::string_view key, std::string_view tensor_name);
    void map_output(std::string_view key, size_t idx);
    void map_output(std::string_view key, std::string_view tensor_name);

    void set_input(const std::string& key, const TensorData& tensor);
    TensorData get_output(const std::string& key);
    TensorFormat get_input_format(const std::string& key);
    TensorFormat get_output_format(const std::string& key);
    void reset_states();
    void run();

private:
    static DataType to_compute_flow_type(halide_type_t type);
    static std::vector<int64_t> to_compute_flow_shape(const MNN::Tensor& tensor);
    static std::vector<int32_t> to_mnn_shape(const TensorFormat& format);
    static bool shapes_equal(const MNN::Tensor& tensor, const TensorFormat& format);

    size_t input_index(std::string_view key) const;
    size_t output_index(std::string_view key) const;
    size_t find_input_index(std::string_view tensor_name) const;
    size_t find_output_index(std::string_view tensor_name) const;
    void refresh_device_tensors();
    void copy_input_to_device(size_t index);
    void copy_initialized_inputs_to_device();
    void copy_outputs_to_host();
    void resize_input(size_t index, const TensorFormat& format);
    void ensure_owned_host_input(size_t index);
    void wrap_host_input(size_t index, const TensorData& tensor);
    void ensure_host_output(size_t index);

    std::string _model_name;
    ProfilerView _profiler;
    std::unique_ptr<MNN::Interpreter> _interpreter;
    MNN::Session* _session {nullptr};
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;
    std::vector<MNN::Tensor*> _device_inputs;
    std::vector<MNN::Tensor*> _device_outputs;
    std::vector<std::unique_ptr<MNN::Tensor>> _host_inputs;
    std::vector<std::unique_ptr<MNN::Tensor>> _host_outputs;
    std::vector<bool> _input_initialized;
    std::vector<bool> _state_inputs;
    std::vector<TensorData> _outputs;
    std::unordered_map<std::string, size_t> _input_indices;
    std::unordered_map<std::string, size_t> _output_indices;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> _states;
    bool _has_run {false};
};

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_MNN
