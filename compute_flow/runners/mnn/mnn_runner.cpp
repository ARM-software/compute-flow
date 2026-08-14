//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_MNN

#include "runners/mnn/mnn_runner.h"

#include "compute_flow/check.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace compute_flow {

namespace {

bool tensor_layout_matches(const MNN::Tensor& lhs, const MNN::Tensor& rhs) {
    return lhs.getType() == rhs.getType() && lhs.getDimensionType() == rhs.getDimensionType() &&
           lhs.shape() == rhs.shape();
}

}  // namespace

void MNNRunner::setup(const ModelData& model, ProfilerView profiler) {
    if (model.acc_config.priorities.empty()) {
        throw std::invalid_argument("MNN runner requires an accelerator preference");
    }

    MNN::ScheduleConfig config;
    MNN::BackendConfig backend_config;
    backend_config.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backend_config;
    const AcceleratorPreference& accelerator = model.acc_config.priorities.front();

    switch (accelerator.type) {
        case AcceleratorType::Cpu: {
            const CpuOptions& cpu_options = std::get<CpuOptions>(accelerator.params);

            if (cpu_options.num_threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument("MNN runner thread count exceeds int32_t range");
            }

            config.type = MNN_FORWARD_CPU;

            if (cpu_options.num_threads > 0) {
                config.numThread = static_cast<int32_t>(cpu_options.num_threads);
            }

            break;
        }
        case AcceleratorType::Gpu: {
            const GpuOptions& gpu_options = std::get<GpuOptions>(accelerator.params);

            if (gpu_options.device_index != 0) {
                throw std::invalid_argument("MNN runner supports only GPU device index 0");
            }

#ifdef COMPUTE_FLOW_HAS_MNN_GPU
#if defined(__ANDROID__)
            config.type = MNN_FORWARD_OPENCL;
#else
            throw std::invalid_argument("MNN GPU acceleration is supported on Android™ only");
#endif
            config.mode = MNN_GPU_TUNING_FAST;
#else
            throw std::invalid_argument(
                "MNN GPU acceleration was disabled at build time; rebuild with "
                "-DCOMPUTE_FLOW_ENABLE_MNN_GPU=ON");
#endif
            break;
        }
        default:
            throw std::invalid_argument("MNN runner received an unsupported accelerator");
    }

    auto interpreter = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.model_path.c_str()));
    COMPUTE_FLOW_CHECK(interpreter != nullptr);
    interpreter->setSessionHint(MNN::Interpreter::HintMode::CPU_ENABLE_KLEIDIAI, true);

    MNN::Session* const session = interpreter->createSession(config);
    COMPUTE_FLOW_CHECK(session != nullptr);

    _model_name = model.model_name;
    _profiler = profiler;
    _interpreter = std::move(interpreter);
    _session = session;
    _input_indices.clear();
    _output_indices.clear();
    _states = model.states;

    const auto& inputs = _interpreter->getSessionInputAll(_session);
    const auto& outputs = _interpreter->getSessionOutputAll(_session);
    _input_names.clear();
    _output_names.clear();
    _input_names.reserve(inputs.size());
    _output_names.reserve(outputs.size());
    for (const auto& [name, tensor] : inputs) {
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        _input_names.push_back(name);
    }
    for (const auto& [name, tensor] : outputs) {
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        _output_names.push_back(name);
    }

    refresh_device_tensors();
    _host_inputs.clear();
    _host_inputs.resize(_device_inputs.size());
    _host_outputs.clear();
    _host_outputs.resize(_device_outputs.size());
    _input_initialized.assign(_device_inputs.size(), false);
    _state_inputs.assign(_device_inputs.size(), false);
    _outputs.assign(_device_outputs.size(), {});
    _has_run = false;

    for (const auto& [name, index] : model.inputs) {
        map_input(name, static_cast<size_t>(index));
    }
    for (const auto& [name, index] : model.outputs) {
        map_output(name, static_cast<size_t>(index));
    }
    for (const auto& [name, indices] : _states) {
        const size_t input_index = static_cast<size_t>(indices.first);

        map_input(name, input_index);
        map_output(name, static_cast<size_t>(indices.second));
        _state_inputs[input_index] = true;
    }

    reset_states();
}

const std::string& MNNRunner::model_name() const {
    return _model_name;
}

void MNNRunner::map_input(std::string_view key, size_t idx) {
    if (idx >= _device_inputs.size()) {
        throw std::out_of_range("MNN input index out of range: " + std::to_string(idx));
    }
    const std::string owned_key(key);
    if (!_input_indices.emplace(owned_key, idx).second) {
        throw std::invalid_argument("MNN input key already exists: " + owned_key);
    }
}

void MNNRunner::map_input(std::string_view key, std::string_view tensor_name) {
    map_input(key, find_input_index(tensor_name));
}

void MNNRunner::map_output(std::string_view key, size_t idx) {
    if (idx >= _device_outputs.size()) {
        throw std::out_of_range("MNN output index out of range: " + std::to_string(idx));
    }
    const std::string owned_key(key);
    if (!_output_indices.emplace(owned_key, idx).second) {
        throw std::invalid_argument("MNN output key already exists: " + owned_key);
    }
}

void MNNRunner::map_output(std::string_view key, std::string_view tensor_name) {
    map_output(key, find_output_index(tensor_name));
}

void MNNRunner::set_input(const std::string& key, const TensorData& tensor) {
    const size_t index = input_index(key);
    if (tensor.data == nullptr) {
        throw std::invalid_argument("MNN input tensor data is null for key: " + key);
    }
    if (tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("MNN input tensor shape must be static for key: " + key);
    }
    if (tensor.format.dt != to_compute_flow_type(_device_inputs[index]->getType())) {
        throw std::invalid_argument("MNN input tensor dtype mismatch for key: " + key);
    }

    const bool requires_resize = !shapes_equal(*_device_inputs[index], tensor.format);
    if (requires_resize) {
        resize_input(index, tensor.format);
    }

    const int64_t num_bytes = tensor.format.get_nbytes();
    if (num_bytes < 0 || static_cast<uint64_t>(num_bytes) != _device_inputs[index]->usize()) {
        throw std::invalid_argument("MNN input tensor byte size mismatch for key: " + key);
    }

    if (_state_inputs[index]) {
        ensure_owned_host_input(index);
        std::memcpy(_host_inputs[index]->host<void>(), tensor.data, static_cast<size_t>(num_bytes));
    } else {
        wrap_host_input(index, tensor);
    }

    _input_initialized[index] = true;

    if (requires_resize) {
        copy_initialized_inputs_to_device();
    } else {
        copy_input_to_device(index);
    }
}

TensorData MNNRunner::get_output(const std::string& key) {
    const size_t index = output_index(key);
    if (!_has_run) throw std::logic_error("MNN output requested before running the model");

    TensorData& output = _outputs[index];
    output.data = _host_outputs[index]->host<void>();
    output.format.dt = to_compute_flow_type(_host_outputs[index]->getType());
    output.format.shape = to_compute_flow_shape(*_host_outputs[index]);
    output.user_hosted = false;
    return output;
}

TensorFormat MNNRunner::get_input_format(const std::string& key) {
    const MNN::Tensor& tensor = *_device_inputs[input_index(key)];
    return {to_compute_flow_type(tensor.getType()), to_compute_flow_shape(tensor)};
}

TensorFormat MNNRunner::get_output_format(const std::string& key) {
    const MNN::Tensor& tensor = *_device_outputs[output_index(key)];
    return {to_compute_flow_type(tensor.getType()), to_compute_flow_shape(tensor)};
}

void MNNRunner::reset_states() {
    for (const auto& [name, indices] : _states) {
        (void)indices;
        const size_t index = input_index(name);
        const TensorFormat format = get_input_format(name);
        if (format.has_dynamic_shape()) {
            throw std::invalid_argument("MNN state tensor shape must be static for key: " + name);
        }
        ensure_owned_host_input(index);
        std::memset(_host_inputs[index]->host<void>(), 0, _host_inputs[index]->usize());
        _input_initialized[index] = true;
        copy_input_to_device(index);
    }
}

void MNNRunner::run() {
    if (std::find(_input_initialized.begin(), _input_initialized.end(), false) != _input_initialized.end()) {
        throw std::logic_error("All MNN inputs must be set before running the model");
    }
    auto model_scope = _profiler.measure_model("mnn", _model_name);
    const MNN::ErrorCode result = _interpreter->runSession(_session);
    COMPUTE_FLOW_CHECK(result == MNN::NO_ERROR);
    copy_outputs_to_host();
    _has_run = true;

    for (const auto& [name, indices] : _states) {
        (void)indices;
        set_input(name, get_output(name));
    }
}

DataType MNNRunner::to_compute_flow_type(halide_type_t type) {
    if (type.lanes != 1) {
        throw std::invalid_argument("MNN vector tensor types are unsupported");
    }
    if (type.code == halide_type_float) {
        if (type.bits == 16) return DataType::F16;
        if (type.bits == 32) return DataType::F32;
        if (type.bits == 64) return DataType::F64;
    } else if (type.code == halide_type_bfloat && type.bits == 16) {
        return DataType::BF16;
    } else if (type.code == halide_type_int) {
        if (type.bits == 8) return DataType::I8;
        if (type.bits == 32) return DataType::I32;
        if (type.bits == 64) return DataType::I64;
    } else if (type.code == halide_type_uint) {
        if (type.bits == 1) return DataType::Bool;
        if (type.bits == 8) return DataType::U8;
        if (type.bits == 32) return DataType::U32;
        if (type.bits == 64) return DataType::U64;
    }
    throw std::invalid_argument(
        "Unsupported MNN tensor type: code=" + std::to_string(type.code) +
        ", bits=" + std::to_string(type.bits));
}

std::vector<int64_t> MNNRunner::to_compute_flow_shape(const MNN::Tensor& tensor) {
    const std::vector<int32_t> mnn_shape = tensor.shape();
    return {mnn_shape.begin(), mnn_shape.end()};
}

std::vector<int32_t> MNNRunner::to_mnn_shape(const TensorFormat& format) {
    std::vector<int32_t> shape;
    shape.reserve(format.shape.size());
    for (const int64_t dimension : format.shape) {
        if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("MNN input dimension is outside the supported int32_t range");
        }
        shape.push_back(static_cast<int32_t>(dimension));
    }
    return shape;
}

bool MNNRunner::shapes_equal(const MNN::Tensor& tensor, const TensorFormat& format) {
    return to_compute_flow_shape(tensor) == format.shape;
}

size_t MNNRunner::input_index(std::string_view key) const {
    const auto it = _input_indices.find(std::string(key));
    if (it == _input_indices.end()) {
        throw std::out_of_range("MNN input key not found: " + std::string(key));
    }
    return it->second;
}

size_t MNNRunner::output_index(std::string_view key) const {
    const auto it = _output_indices.find(std::string(key));
    if (it == _output_indices.end()) {
        throw std::out_of_range("MNN output key not found: " + std::string(key));
    }
    return it->second;
}

size_t MNNRunner::find_input_index(std::string_view tensor_name) const {
    const auto it = std::find(_input_names.begin(), _input_names.end(), tensor_name);
    if (it == _input_names.end()) {
        throw std::invalid_argument("MNN input tensor not found: " + std::string(tensor_name));
    }
    return static_cast<size_t>(std::distance(_input_names.begin(), it));
}

size_t MNNRunner::find_output_index(std::string_view tensor_name) const {
    const auto it = std::find(_output_names.begin(), _output_names.end(), tensor_name);
    if (it == _output_names.end()) {
        throw std::invalid_argument("MNN output tensor not found: " + std::string(tensor_name));
    }
    return static_cast<size_t>(std::distance(_output_names.begin(), it));
}

void MNNRunner::refresh_device_tensors() {
    _device_inputs.clear();
    _device_outputs.clear();
    _device_inputs.reserve(_input_names.size());
    _device_outputs.reserve(_output_names.size());
    for (const std::string& name : _input_names) {
        MNN::Tensor* const tensor = _interpreter->getSessionInput(_session, name.c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        _device_inputs.push_back(tensor);
    }
    for (const std::string& name : _output_names) {
        MNN::Tensor* const tensor = _interpreter->getSessionOutput(_session, name.c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        _device_outputs.push_back(tensor);
    }
}

void MNNRunner::copy_input_to_device(size_t index) {
    COMPUTE_FLOW_CHECK(_host_inputs[index] != nullptr);
    if (!tensor_layout_matches(*_host_inputs[index], *_device_inputs[index])) {
        throw std::runtime_error("MNN input layout changed while resizing another input");
    }
    COMPUTE_FLOW_CHECK(_device_inputs[index]->copyFromHostTensor(_host_inputs[index].get()));
}

void MNNRunner::copy_initialized_inputs_to_device() {
    for (size_t index = 0; index < _input_initialized.size(); ++index) {
        if (_input_initialized[index]) copy_input_to_device(index);
    }
}

void MNNRunner::copy_outputs_to_host() {
    for (size_t index = 0; index < _device_outputs.size(); ++index) {
        ensure_host_output(index);
        COMPUTE_FLOW_CHECK(_device_outputs[index]->copyToHostTensor(_host_outputs[index].get()));
    }
}

void MNNRunner::resize_input(size_t index, const TensorFormat& format) {
    _interpreter->resizeTensor(_device_inputs[index], to_mnn_shape(format));
    _interpreter->resizeSession(_session);
    refresh_device_tensors();
    for (size_t input_index = 0; input_index < _host_inputs.size(); ++input_index) {
        if (_host_inputs[input_index] == nullptr ||
            tensor_layout_matches(*_host_inputs[input_index], *_device_inputs[input_index])) {
            continue;
        }
        _host_inputs[input_index].reset();
        _input_initialized[input_index] = false;
    }
    _host_inputs[index].reset();
    _input_initialized[index] = false;
    for (auto& output : _host_outputs) output.reset();
    _has_run = false;
}

void MNNRunner::ensure_owned_host_input(size_t index) {
    if (_host_inputs[index] == nullptr || !tensor_layout_matches(*_host_inputs[index], *_device_inputs[index])) {
        _host_inputs[index] = std::make_unique<MNN::Tensor>(
            _device_inputs[index], _device_inputs[index]->getDimensionType());
    }
}

void MNNRunner::wrap_host_input(size_t index, const TensorData& tensor) {
    if (!shapes_equal(*_device_inputs[index], tensor.format)) {
        throw std::invalid_argument("MNN host input tensor shape does not match the device input");
    }

    if (tensor.format.dt != to_compute_flow_type(_device_inputs[index]->getType())) {
        throw std::invalid_argument("MNN host input tensor dtype does not match the device input");
    }

    _host_inputs[index].reset(MNN::Tensor::create(
        _device_inputs[index]->shape(),
        _device_inputs[index]->getType(),
        tensor.data,
        _device_inputs[index]->getDimensionType()));
    COMPUTE_FLOW_CHECK(_host_inputs[index] != nullptr);
}

void MNNRunner::ensure_host_output(size_t index) {
    if (_host_outputs[index] == nullptr || !tensor_layout_matches(*_host_outputs[index], *_device_outputs[index])) {
        _host_outputs[index] = std::make_unique<MNN::Tensor>(
            _device_outputs[index], _device_outputs[index]->getDimensionType());
    }
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_MNN
