//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_EXECUTORCH

#include "runners/executorch/executorch_runner.h"

#if defined(ET_USE_THREADPOOL)
#include "executorch/extension/threadpool/cpuinfo_utils.h"
#include "executorch/extension/threadpool/threadpool.h"
#endif
#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/platform/log.h"

#include <algorithm>

namespace compute_flow {

ExecuTorchRunner::ExecuTorchRunner() {}

void ExecuTorchRunner::setup(const ModelData& model,
                             ProfilerView profiler) {

    _model_name = model.model_name;
    _profiler   = profiler;

    const std::string model_path  = model.model_path;
    AcceleratorConfig accelerator = model.acc_config;

    // TODO ONLY CPU IS SUPPERTED FOR NOW
    const auto opts = std::get<CpuOptions>(accelerator.priorities[0].params);
    initialize_cpu_delegate(opts);

    _model = std::make_unique<executorch::extension::Module>(model_path, executorch::extension::Module::LoadMode::File);

    EXECUTORCH_CHECK(_model != nullptr);

    // Get the number of inputs and output
    // The user owns the inputs and outputs
    auto num_inputs  = _model->method_meta("forward")->num_inputs();
    auto num_outputs = _model->method_meta("forward")->num_outputs();

    _et_inputs.resize(num_inputs);
    _raw_inputs.resize(num_inputs);
    _aten_inputs.resize(num_inputs);
    _et_outputs.resize(num_outputs);
    _raw_outputs.resize(num_outputs);
    _inputs.resize(num_inputs);
    _outputs.resize(num_outputs);

    // In theory, we do not need to allocate memory. The user can allocate the memory
    for(int64_t i = 0; i < num_inputs; ++i) {
        auto type   = _model->method_meta("forward")->input_tensor_meta(i)->scalar_type();
        auto nbytes = _model->method_meta("forward")->input_tensor_meta(i)->nbytes();
        auto sizes  = _model->method_meta("forward")->input_tensor_meta(i)->sizes();

        std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());

        _raw_inputs[i].assign(static_cast<size_t>(nbytes), uint8_t{0});
        auto et_tensor = executorch::extension::from_blob(_raw_inputs[i].data(), shape, type);

        // The following is required to keep the lifetime of what from_blob returns
        _aten_inputs[i] = et_tensor;

        // Now we can store the tensor
        _et_inputs[i] = executorch::runtime::EValue(_aten_inputs[i]);
    }

    // Map input IDX to name
    for (const auto& [name_input, idx_input] : model.inputs) {
        map_input(name_input, idx_input);
    }

    // Map output IDX to name
    for (const auto& [name_output, idx_output] : model.outputs) {
        map_output(name_output, idx_output);
    }

    _states = model.states;
    for (const auto& [name, io] : _states) {
        const auto& [idx_input, idx_output] = io;
        map_input(name, idx_input);
        map_output(name, idx_output);
    }
}

void ExecuTorchRunner::run() {
    auto model_scope = _profiler.measure_model("executorch", _model_name);

    // It returns executorch::runtime::Result<std::vector<executorch::runtime::EValue>>
    auto result = _model->forward(_et_inputs);

    EXECUTORCH_CHECK(result.error() == executorch::runtime::Error::Ok);

    const int64_t num_outputs = result.get().size();

    for(int64_t i = 0; i < num_outputs; ++i) {
        _et_outputs[i] = result.get().at(i).toTensor();
    }

    for (const auto& [name, io] : _states) {
        (void)io;
        // ExecuTorch may reuse memory-planned output buffers during the next invocation.
        // Copy recurrent state into the runner-owned input buffer to avoid aliasing it.
        set_input(name, get_output(name));
    }

    return;
}

const std::string& ExecuTorchRunner::model_name() const {
    return _model_name;
}

void ExecuTorchRunner::map_input(std::string_view key, size_t idx) {
    // ExecuTorch must be initialized

    std::string k(key);
    if (_in_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    // We should add the tensor in the idx position
    _in_tensor_idx.insert({k, idx});

    auto type  = _et_inputs[idx].toTensor().scalar_type();
    auto sizes = _et_inputs[idx].toTensor().sizes();

    std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());

    auto& tensor        = _inputs[idx];
    tensor.data         = _et_inputs[idx].toTensor().data_ptr();
    tensor.format.dt    = convert_et_dt_to_internal_dt(type);
    tensor.format.shape = convert_et_shape_to_internal_shape(shape);
    tensor.user_hosted  = false;

    return;
}

void ExecuTorchRunner::map_input(std::string_view key, std::string_view tensor_name) {
    size_t idx = find_input_idx_by_name(tensor_name);
    map_input(key, idx);
}

void ExecuTorchRunner::map_output(std::string_view key, size_t idx) {
    std::string k(key);
    if (_out_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    _out_tensor_idx.insert({k, idx});

    auto type   = _model->method_meta("forward")->output_tensor_meta(idx)->scalar_type();
    auto sizes  = _model->method_meta("forward")->output_tensor_meta(idx)->sizes();

    std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());

    auto& tensor        = _outputs[idx];
    tensor.data         = nullptr;
    tensor.format.dt    = convert_et_dt_to_internal_dt(type);
    tensor.format.shape = convert_et_shape_to_internal_shape(shape);
    tensor.user_hosted  = false;

    return;
}

void ExecuTorchRunner::map_output(std::string_view key, std::string_view tensor_name) {
    size_t idx = find_output_idx_by_name(tensor_name);
    map_output(key, idx);
}

TensorData ExecuTorchRunner::get_output(const std::string& key) {

    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }

    const size_t idx = it->second;

    auto type  = _et_outputs[idx].toTensor().scalar_type();
    auto sizes = _et_outputs[idx].toTensor().sizes();

    std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());

    auto& tensor        = _outputs[idx];
    tensor.data         = _et_outputs[idx].toTensor().data_ptr();
    tensor.format.dt    = convert_et_dt_to_internal_dt(type);
    tensor.format.shape = convert_et_shape_to_internal_shape(shape);
    tensor.user_hosted  = false;
    return tensor;

}

void ExecuTorchRunner::set_input(const std::string& key, const TensorData& tensor) {

    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    const size_t idx = it->second;

    if (tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("Input tensor shape must be static for key: " + key);
    }

    if (tensor.data == nullptr) {
        throw std::invalid_argument("Input tensor data is null for key: " + key);
    }

    const auto expected_type = convert_et_dt_to_internal_dt(
        _model->method_meta("forward")->input_tensor_meta(idx)->scalar_type());
    if (tensor.format.dt != expected_type) {
        throw std::invalid_argument("Input tensor dtype mismatch for key: " + key);
    }

    const int64_t input_nbytes = tensor.format.get_nbytes();
    if (input_nbytes < 0) {
        throw std::invalid_argument("Input tensor must have a static shape for key: " + key);
    }

    const auto current_sizes = _et_inputs[idx].toTensor().sizes();
    std::vector<int64_t> current_shape(current_sizes.begin(), current_sizes.end());

    if (!_inputs[idx].user_hosted &&
        tensor.format.shape == current_shape &&
        input_nbytes == static_cast<int64_t>(_et_inputs[idx].toTensor().nbytes())) {
        memcpy(_et_inputs[idx].toTensor().data_ptr(), tensor.data, static_cast<size_t>(input_nbytes));
        _inputs[idx].data = _et_inputs[idx].toTensor().data_ptr();
        _inputs[idx].format = tensor.format;
        _inputs[idx].user_hosted = false;
        return;
    }

    std::vector<executorch::aten::SizesType> shape(
        tensor.format.shape.begin(), tensor.format.shape.end());
    const auto et_type = convert_internal_dt_to_et_dt(tensor.format.dt);

    auto et_tensor = executorch::extension::from_blob(tensor.data, shape, et_type);
    _aten_inputs[idx] = et_tensor;
    _et_inputs[idx] = executorch::runtime::EValue(_aten_inputs[idx]);

    _inputs[idx].data = tensor.data;
    _inputs[idx].format = tensor.format;
    _inputs[idx].user_hosted = true;
}

void ExecuTorchRunner::reset_states() {
    for (const auto& [name, io] : _states) {
        (void)name;
        const size_t idx_input = static_cast<size_t>(io.first);
        const auto input_meta  = _model->method_meta("forward")->input_tensor_meta(idx_input);
        const auto type        = input_meta->scalar_type();
        const auto sizes       = input_meta->sizes();

        auto& raw_input = _raw_inputs.at(idx_input);
        std::fill(raw_input.begin(), raw_input.end(), uint8_t{0});

        std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());
        auto et_tensor          = executorch::extension::from_blob(raw_input.data(), shape, type);
        _aten_inputs[idx_input] = et_tensor;
        _et_inputs[idx_input]   = executorch::runtime::EValue(_aten_inputs[idx_input]);

        auto& input        = _inputs[idx_input];
        input.data         = raw_input.data();
        input.format.dt    = convert_et_dt_to_internal_dt(type);
        input.format.shape = convert_et_shape_to_internal_shape(shape);
        input.user_hosted  = false;
    }
}

TensorFormat ExecuTorchRunner::get_input_format(const std::string& key) {
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

TensorFormat ExecuTorchRunner::get_output_format(const std::string& key) {
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

void ExecuTorchRunner::initialize_cpu_delegate(const CpuOptions& cpu_options) {
#if defined(ET_USE_THREADPOOL)
    ::executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(cpu_options.num_threads);
#endif
}

inline DataType ExecuTorchRunner::convert_et_dt_to_internal_dt(executorch::aten::ScalarType t) {
    switch (t) {
        case executorch::aten::ScalarType::Float: return DataType::F32;
        case executorch::aten::ScalarType::Bool: return DataType::Bool;
        case executorch::aten::ScalarType::Half: return DataType::F16;
        case executorch::aten::ScalarType::QInt8: return DataType::I8;
        case executorch::aten::ScalarType::QUInt8: return DataType::U8;
        case executorch::aten::ScalarType::Int: return DataType::I32;
        case executorch::aten::ScalarType::UInt32: return DataType::U32;
        case executorch::aten::ScalarType::UInt64: return DataType::U64;
        case executorch::aten::ScalarType::Long:   return DataType::I64;
        default:
            throw std::invalid_argument("Unsupported ExecuTorch type: " + std::to_string(static_cast<int>(t)));
    }
}

inline executorch::aten::ScalarType ExecuTorchRunner::convert_internal_dt_to_et_dt(DataType t) {
    switch (t) {
        case DataType::F32: return executorch::aten::ScalarType::Float;
        case DataType::Bool: return executorch::aten::ScalarType::Bool;
        case DataType::F16: return executorch::aten::ScalarType::Half;
        case DataType::I8:  return executorch::aten::ScalarType::QInt8;
        case DataType::U8:  return executorch::aten::ScalarType::QUInt8;
        case DataType::I32: return executorch::aten::ScalarType::Int;
        case DataType::U32: return executorch::aten::ScalarType::UInt32;
        case DataType::U64: return executorch::aten::ScalarType::UInt64;
        case DataType::I64: return executorch::aten::ScalarType::Long;
        default:
            throw std::invalid_argument(
                "Unsupported internal type for ExecuTorch input: " + std::to_string(static_cast<int>(t)));
    }
}

inline std::vector<int64_t>
ExecuTorchRunner::convert_et_shape_to_internal_shape(const std::vector<executorch::aten::SizesType>& dims) {
    std::vector<int64_t> out;
    out.reserve(dims.size());

    for (executorch::aten::SizesType d : dims) {
        // Optional (defensive) overflow guard in case SizesType > int64_t
        if (static_cast<long double>(d) >
            static_cast<long double>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("dimension exceeds int64_t range");
        }
        out.push_back(static_cast<int64_t>(d));
    }
    return out;
}

size_t ExecuTorchRunner::find_input_idx_by_name(std::string_view name) const {

    auto num_inputs = _model->method_meta("forward")->num_inputs();

    // In theory, we do not need to allocate memory. The user can allocate the memory
    for(int64_t i = 0; i < num_inputs; ++i) {
        auto tensor_name = _model->method_meta("forward")->input_tensor_meta(i)->name();

        if (tensor_name == name) {
            return i;
        }
    }

    throw std::invalid_argument("Input not found: " + std::string(name));
}

size_t ExecuTorchRunner::find_output_idx_by_name(std::string_view name) const {

    auto num_outputs = _model->method_meta("forward")->num_outputs();

    // In theory, we do not need to allocate memory. The user can allocate the memory
    for(int64_t i = 0; i < num_outputs; ++i) {
        auto tensor_name = _model->method_meta("forward")->output_tensor_meta(i)->name();

        if (tensor_name == name) {
            return i;
        }
    }

    throw std::invalid_argument("Output not found: " + std::string(name));
}
}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_EXECUTORCH
