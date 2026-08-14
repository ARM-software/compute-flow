//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

 // ADD MACRO TO PREVENT BUILDING THIS FILE IF SENTENCEPIECE MACRO IS NOT PASSED

#ifdef COMPUTE_FLOW_HAS_SENTENCEPIECE

#include "runners/sentencepiece_encode/sentencepiece_encode_runner.h"

namespace compute_flow {

SentencePieceEncodeRunner::SentencePieceEncodeRunner() {}

void SentencePieceEncodeRunner::setup(const ModelData& model,
                                      ProfilerView profiler) {

    _model_name = model.model_name;
    _profiler   = profiler;

    const std::string model_path  = model.model_path;
    AcceleratorConfig accelerator = model.acc_config;

    // TODO: It can only be CPU!
    SENTENCEPIECE_ENCODE_CHECK(_sp.Load(model_path).ok());

    _inputs.resize(1);
    _outputs.resize(1);

    // Map input IDX to name
    for (const auto& [name_input, idx_input] : model.inputs) {
        map_input(name_input, idx_input);
    }

    // Map output IDX to name
    for (const auto& [name_output, idx_output] : model.outputs) {
        map_output(name_output, idx_output);
    }
}

void SentencePieceEncodeRunner::run() {
    // Since the output size depends on the input, we should call get_output after run.
    // Since it could lead to misuse or errors, we can remove get_output and return after run
    // by making the output as part of the arguments of this function
    // The input is not required here as the user may use the input before calling run. However,
    // the output is only used after calling run.
    // The output shape CAN only be static because it is known when the input is defined
    // The input can be DYNAMIC or STATIC. Because of that, the output MUST be part of the interface
    auto model_scope = _profiler.measure_model("spmenc", _model_name);

    _sp.Encode(_in, &_out);

    return;
}

const std::string& SentencePieceEncodeRunner::model_name() const {
    return _model_name;
}

void SentencePieceEncodeRunner::map_input(std::string_view key, size_t idx) {
    SENTENCEPIECE_ENCODE_CHECK(idx == 0);
    std::string k(key);
    if (_in_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    _in_tensor_idx.insert({k, idx});

    auto& tensor        = _inputs[idx];
    tensor.data         = nullptr;
    tensor.format.dt    = DataType::String;
    tensor.format.shape = std::vector<int64_t>(1, -1);
    tensor.user_hosted = true;

    return;
}

void SentencePieceEncodeRunner::map_input(std::string_view key, std::string_view tensor_name) {
    // Whatever tensor name is provided by the user, it will be always be mapped to the same tensor
    map_input(key, 0);
}

void SentencePieceEncodeRunner::map_output(std::string_view key, size_t idx) {
    SENTENCEPIECE_ENCODE_CHECK(idx == 0);
    std::string k(key);
    if (_out_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    _out_tensor_idx.insert({k, idx});

    auto& tensor        = _outputs[idx];
    tensor.data         = _out.data();
    tensor.format.dt    = DataType::I32;
    tensor.format.shape = std::vector<int64_t>(1, -1);
    tensor.user_hosted  = false;

    return;
}

void SentencePieceEncodeRunner::map_output(std::string_view key, std::string_view tensor_name) {
    // Whatever tensor name is provided by the user, it will be always be mapped to the same tensor
    map_output(key, 0);
}

TensorData SentencePieceEncodeRunner::get_output(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    const size_t idx = it->second;

    // Update the shape. We know the output shape only after running the model
    // At the moment, the output CANNOT be in the user space if the output is dynamic
    _outputs[idx].data            = (void*)_out.data();
    _outputs[idx].format.shape[0] = _out.size();
    return _outputs[idx];
}

void SentencePieceEncodeRunner::set_input(const std::string& key, const TensorData& tensor) {
    // Check shape
    // Check type
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }

    if (tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("Input tensor shape must be static for key: " + key);
    }

    _in.clear();
    _in.assign((const char*)tensor.data, tensor.format.shape[0]);

    return;
}

void SentencePieceEncodeRunner::reset_states() {
}

TensorFormat SentencePieceEncodeRunner::get_input_format(const std::string& key) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    const size_t idx = it->second;

    TensorFormat format;
    format.dt    = _inputs[idx].format.dt;
    format.shape = _inputs[idx].format.shape;
    return format;
}

TensorFormat SentencePieceEncodeRunner::get_output_format(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    const size_t idx = it->second;

    TensorFormat format;
    format.dt    = _outputs[idx].format.dt;
    format.shape = _outputs[idx].format.shape;
    return format;
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_SENTENCEPIECE
