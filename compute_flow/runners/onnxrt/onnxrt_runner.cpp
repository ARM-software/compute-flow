//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// ADD MACRO TO PREVENT BUILDING THIS FILE IF LITERT MACRO IS NOT PASSED

#ifdef COMPUTE_FLOW_HAS_ONNXRT

#include "runners/onnxrt/onnxrt_runner.h"

namespace compute_flow {

std::string get_key_from_idx(const std::unordered_map<std::string, size_t>& map, size_t index) {
    for (const auto& [key, value] : map) {
        if (value == index) {
            return key;
        }
    }
    throw std::runtime_error("Index not found in map");
}

static inline Ort::Value make_tensor_from(
    const TensorData& td,
    Ort::MemoryInfo& meminfo,
    const std::string& name)
{
    auto num_bytes = td.format.get_nbytes();

    switch (td.format.dt) {
        case DataType::F32:
            return Ort::Value::CreateTensor<float>(meminfo,
                static_cast<float*>(td.data), num_bytes / sizeof(float),
                td.format.shape.data(), td.format.shape.size());
        case DataType::I64:
            return Ort::Value::CreateTensor<int64_t>(meminfo,
                static_cast<int64_t*>(td.data), num_bytes / sizeof(int64_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::I32:
            return Ort::Value::CreateTensor<int32_t>(meminfo,
                static_cast<int32_t*>(td.data), num_bytes / sizeof(int32_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::U8:
            return Ort::Value::CreateTensor<uint8_t>(meminfo,
                static_cast<uint8_t*>(td.data), num_bytes / sizeof(uint8_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::F64:
            return Ort::Value::CreateTensor<double>(meminfo,
                static_cast<double*>(td.data), num_bytes / sizeof(double),
                td.format.shape.data(), td.format.shape.size());
        case DataType::F16:
            return Ort::Value::CreateTensor<Ort::Float16_t>(meminfo,
                static_cast<Ort::Float16_t*>(td.data), num_bytes / sizeof(Ort::Float16_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::I8:
            return Ort::Value::CreateTensor<int8_t>(meminfo,
                static_cast<int8_t*>(td.data), num_bytes / sizeof(int8_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::U64:
            return Ort::Value::CreateTensor<uint64_t>(meminfo,
                static_cast<uint64_t*>(td.data), num_bytes / sizeof(uint64_t),
                td.format.shape.data(), td.format.shape.size());
        case DataType::Bool:
            ONNXRT_CHECK(sizeof(bool) == 1);
            return Ort::Value::CreateTensor<bool>(meminfo,
                static_cast<bool*>(td.data), num_bytes / sizeof(bool),
                td.format.shape.data(), td.format.shape.size());
        default:
            throw std::runtime_error("Unsupported or special dtype for zero-copy CreateTensor: ");
    }
}

inline const char* OnnxTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:  return "UNDEFINED";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      return "FLOAT";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:     return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:      return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:     return "STRING";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       return "BOOL";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     return "DOUBLE";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     return "UINT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:  return "COMPLEX64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "COMPLEX128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   return "BFLOAT16";
        default: return "UNKNOWN";
    }
}

ONNXRTRunner::ONNXRTRunner() : _env(ORT_LOGGING_LEVEL_WARNING, "ONNXRTRunner"), _model(nullptr), _memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

void ONNXRTRunner::setup(const ModelData& model,
                         ProfilerView profiler) {

    _model_name = model.model_name;
    _profiler   = profiler;

    const std::string model_path  = model.model_path;
    AcceleratorConfig accelerator = model.acc_config;

    // Initialize the ONNX environment
    //_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, _model_name);

    // Initialize the session options
    // TODO ONLY CPU IS SUPPERTED FOR NOW
    const int64_t num_threads = std::get<CpuOptions>(accelerator.priorities[0].params).num_threads;

    _session_opts.SetIntraOpNumThreads(num_threads);

    _session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Initialize the ONNX model
    _model.reset(); // drop any old session first
    _model = std::make_unique<Ort::Session>(_env, model_path.c_str(), _session_opts);

    // Initialize the ONNX memory
    _memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Check whether we have the same number of inputs and outputs in the model and in the manifest
    // TODO: it should be part of a validation step
    const int64_t onnx_model_num_input      = _model->GetInputCount();
    const int64_t onnx_model_num_output     = _model->GetOutputCount();
    const int64_t manifest_model_num_input  = model.inputs.size();
    const int64_t manifest_model_num_output = model.outputs.size();
    const int64_t manifest_model_num_states = model.states.size();

    ONNXRT_CHECK(onnx_model_num_input == manifest_model_num_input + manifest_model_num_states);
    ONNXRT_CHECK(onnx_model_num_output == manifest_model_num_output + manifest_model_num_states);

    // Check whether we have the same tensor name for inputs and outputs in the model and in the manifest
    // TODO: it should be part of a validation step

    // Pre-allocate
    _inputs.resize(_model->GetInputCount());
    _outputs.resize(_model->GetOutputCount());

    // Map input IDX to name
    // To be refactored because the mapping should be done with the tensor name and not on the tensor index
    for (const auto& [name_input, idx_input] : model.inputs) {
        map_input(name_input, idx_input);
    }

    // Map output IDX to name
    for (const auto& [name_output, idx_output] : model.outputs) {
        map_output(name_output, idx_output);
    }

    _states = model.states;
    for (const auto& [name, io] : model.states) {
        const auto& [idx_input, idx_output] = io;
        map_input(name, idx_input);
        map_output(name, idx_output);
    }

    // This will be needed when the user wants to reset the state
    set_initial_state_info();
}

void ONNXRTRunner::run() {

    auto model_scope = _profiler.measure_model("onnxrt", _model_name);

    const int64_t num_inputs  = _model->GetInputCount();
    const int64_t num_outputs = _model->GetOutputCount();

    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    std::vector<Ort::Value> inputs;
    input_names.reserve(num_inputs);
    inputs.reserve(num_inputs);
    output_names.resize(num_outputs);

    std::vector<std::string> input_name_buf;
    std::vector<std::string> output_name_buf;
    input_name_buf.reserve(num_inputs);
    output_name_buf.resize(num_outputs);

    Ort::AllocatorWithDefaultOptions allocator;
    // The input and output already contains the states because we have initialized the state in the setup method

    for (const auto& [internal_tensor_name, idx_input] : _in_tensor_idx) {
        const std::string real_tensor_name = std::string(_model->GetInputNameAllocated(idx_input, allocator).get());
        input_name_buf.push_back(real_tensor_name);
        input_names.push_back(input_name_buf.back().c_str());

        ONNXRT_CHECK(idx_input < _inputs.size());

        TensorData& td = _inputs[idx_input];

        bool is_state_flag = is_state(internal_tensor_name);
        if(td.data == nullptr && is_state_flag) {
            for(size_t i = 0; i < td.format.shape.size(); ++i) {
                if(td.format.shape[i] < 0) {
                    td.format.shape[i] = 1;
                }
            }

            // The following memory will be deallocated automatically when exiting the run method
            td.data = (void*)allocator.Alloc(td.format.get_nbytes());
            memset(td.data, 0, td.format.get_nbytes());
        } else {
            ONNXRT_CHECK(td.data != nullptr);                 // must be set via set_input(
        }

        // Create ORT tensor that borrows your buffer (zero-copy)
        inputs.emplace_back(std::move(make_tensor_from(td, _memory_info, real_tensor_name)));
    }

    for (const auto& [internal_tensor_name, idx_output] : _out_tensor_idx) {
        const std::string real_tensor_name = std::string(_model->GetOutputNameAllocated(idx_output, allocator).get());

        ONNXRT_CHECK(idx_output < output_names.size());
        output_name_buf[idx_output] = real_tensor_name;
        output_names[idx_output] = output_name_buf[idx_output].c_str();
    }

    _last_outputs = _model->Run(Ort::RunOptions{},
                                input_names.data(),
                                inputs.data(),
                                inputs.size(),
                                output_names.data(),
                                output_names.size());

    // Set the state by linking the output to the input
    // We set the input based on the past output
    for (const auto& [name, io] : _states) {
        const auto& [idx_input, idx_output] = io;

        auto tensor_info = _last_outputs[idx_output].GetTensorTypeAndShapeInfo();

        std::vector<int64_t> shape = tensor_info.GetShape();
        const std::string real_tensor_name = std::string(_model->GetOutputNameAllocated(idx_output, allocator).get());

        // It means zero-element tensor. Therefore, this tensor is not required anymore
        // Make sure it is not a scalar value. Otherwise, we have segfault
        if(shape.size() > 0) {
            if(shape[0] == 0) {
                continue;
            }
        }

        // Keep state outputs alive across runs to avoid dangling pointers when
        // a later run returns zero-length cache outputs (shape[0] == 0). We
        // move the Ort::Value into StateInfo so the backing buffer stays valid
        // until a new non-empty state arrives.
        TensorData out_td = get_output(name);
        StateInfo& st = _states_info[name];
        st.value = std::move(_last_outputs[idx_output]);
        out_td.data = st.value->GetTensorMutableData<void>();

        TensorData in_td = out_td;
        set_input(name, in_td);

        st.shape = out_td.format.shape;
        st.data = out_td.data;
        st.dt = out_td.format.dt;
    }

    return;
}

const std::string& ONNXRTRunner::model_name() const {
    return _model_name;
}

void ONNXRTRunner::map_input(std::string_view key, size_t idx) {
    ONNXRT_CHECK(idx < _model->GetInputCount());

    std::string k(key);
    if (_in_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    _in_tensor_idx.insert({k, idx});

    // TODO: Here we should check whether we are changing the dimensions
    // for a static tensor! We can only change the dimensions for dynamic tensors!
    // Therefore, we need a check in-place (currently missing!)

    Ort::TypeInfo type_info = _model->GetInputTypeInfo(idx);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto& tensor        = _inputs[idx];
    tensor.data         = nullptr;
    tensor.format.dt    = convert_onnxrt_dt_to_internal_dt(tensor_info.GetElementType());
    tensor.format.shape = convert_onnxrt_shape_to_internal_shape(tensor_info.GetShape());
    tensor.user_hosted  = true;

    return;
}

void ONNXRTRunner::map_input(std::string_view key, std::string_view tensor_name) {
    size_t idx = find_input_idx_by_name(tensor_name);
    map_input(key, idx);
}

void ONNXRTRunner::map_output(std::string_view key, size_t idx) {
    ONNXRT_CHECK(idx < _model->GetOutputCount());
    std::string k(key);
    if (_out_tensor_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }

    _out_tensor_idx.insert({k, idx});

    Ort::TypeInfo type_info = _model->GetOutputTypeInfo(idx);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto& tensor        = _outputs[idx];
    tensor.data         = nullptr;
    tensor.format.dt    = convert_onnxrt_dt_to_internal_dt(tensor_info.GetElementType());
    tensor.format.shape = convert_onnxrt_shape_to_internal_shape(tensor_info.GetShape());
    tensor.user_hosted  = false;

    return;
}

void ONNXRTRunner::map_output(std::string_view key, std::string_view tensor_name) {
    size_t idx = find_output_idx_by_name(tensor_name);
    map_output(key, idx);
}

TensorData ONNXRTRunner::get_output(const std::string& key) {

    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    const size_t idx = it->second;

    auto tensor_info = _last_outputs[idx].GetTensorTypeAndShapeInfo();

    TensorData t;
    t.data = _last_outputs[idx].GetTensorMutableData<void>();
    t.format.dt = convert_onnxrt_dt_to_internal_dt(tensor_info.GetElementType());
    t.format.shape = convert_onnxrt_shape_to_internal_shape(tensor_info.GetShape());
    t.user_hosted  = false;

    return t;
}

void ONNXRTRunner::set_input(const std::string& key, const TensorData& tensor) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    const size_t idx = it->second;

    if (tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("Input tensor shape must be static for key: " + key);
    }

    TensorData& td = _inputs[idx];
    td.data = tensor.data;

    if (tensor.format.has_dynamic_shape()) {
            throw std::invalid_argument("Input tensor rank mismatch for key: " + key);
    }
    // TODO: If it is dynamic shape, we should record it somewhere
    td.format.shape = tensor.format.shape;


    //Ort::TypeInfo type_info = _model->GetInputTypeInfo(idx);
    //auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    // WE MUST CHECK IF WE HAVE THE SAME NUMBER OF DIMENSIONS!
    //ONNXRT_CHECK(tensor.format.shape.size() == tensor_info.GetShape().size());

    Ort::AllocatorWithDefaultOptions allocator;
    //printf("Mapping %d %s\n", idx, _model->GetInputNameAllocated(idx, allocator).get());
    //printf("Num dims %d %d\n", tensor.format.shape.size(), tensor_info.GetShape().size());
    //printf("Dims internal %d %d %d\n", tensor.format.shape[0], tensor.format.shape[1], tensor.format.shape[2]);
    //printf("Dims onnx %d %d %d\n", tensor_info.GetShape()[0], tensor_info.GetShape()[1], tensor_info.GetShape()[2]);

    // When we set the input we should check:
    // - can we change the shape? Only for dynamic shape
    // - is the number of dimensions the same as what the model expects?
    // - if the shape is static, are the dimensions the same?
}

void ONNXRTRunner::reset_states() {
    // The aim of this function is to restore the state to the initial shape (if working with dynamic shape)
    // and set the data pointer to its original value (e.g., nullptr).
    // If the pointer is NOT a nullptr and if the shape is not dynamic, we initialize with zero
    for (const auto& [internal_tensor_name, io] : _states) {
        const auto& [idx_input, idx_output] = io;
        ONNXRT_CHECK(idx_input < _inputs.size());

        TensorData& td = _inputs[idx_input];
        StateInfo& st = _states_info[internal_tensor_name];
        td.format.shape = st.initial_shape;
        td.data         = nullptr;

        st.shape = st.initial_shape;
        st.data = nullptr;
        st.value.reset();

        // In ONNX, it will always be a nullptr the input
        // However, we implement something generic that can work with other runners as well
        if(td.data != nullptr) {
            // Check whether it is dynamic shape
            bool is_dynamic_shape = false;
            for(size_t i = 0; i < td.format.shape.size(); ++i) {
                if(td.format.shape[i] < 0) {
                    is_dynamic_shape = true;
                }
            }

            // If not dynamic shape, we initialize with zero
            if(!is_dynamic_shape) {
                memset(td.data, 0, td.format.get_nbytes());
            }
        }
    }
}

TensorFormat ONNXRTRunner::get_input_format(const std::string& key) {

    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    const size_t idx = it->second;
    ONNXRT_CHECK(idx < _inputs.size());
    TensorFormat format;
    format.dt    = _inputs[idx].format.dt;
    format.shape = _inputs[idx].format.shape;
    return format;
}

TensorFormat ONNXRTRunner::get_output_format(const std::string& key) {

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

inline DataType ONNXRTRunner::convert_onnxrt_dt_to_internal_dt(ONNXTensorElementDataType type) {

    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      return DataType::F32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      return DataType::U8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       return DataType::I8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      return DataType::I32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      return DataType::I64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:     return DataType::String;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    return DataType::F16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     return DataType::F64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     return DataType::U64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       return DataType::Bool;
        default: throw std::invalid_argument("Unsupported TfLiteType: " + std::to_string(static_cast<int>(type)));
    }
}

inline std::vector<int64_t> ONNXRTRunner::convert_onnxrt_shape_to_internal_shape(const std::vector<int64_t>& dims) {
    return dims;
}

inline bool ONNXRTRunner::is_state(const std::string& key) {
    auto it = _states.find(key);
    return it != _states.end();
}

void ONNXRTRunner::set_initial_state_info() {
    // We should only call this function once
    // In theory, this function is not needed because we can query the input tensor shape
    // from ONNX
    for (const auto& [internal_tensor_name, io] : _states) {
        const auto& [idx_input, idx_output] = io;
        ONNXRT_CHECK(idx_input < _inputs.size());

        const TensorData& td = _inputs[idx_input];

        StateInfo st;
        st.initial_shape = td.format.shape;
        st.shape = td.format.shape;
        st.data  = td.data;
        st.dt    = td.format.dt;

        // Store via emplace: StateInfo holds an optional Ort::Value (move-only),
        // so copy-assignment into the map is not allowed.
        _states_info.emplace(internal_tensor_name, std::move(st));
    }
}

size_t ONNXRTRunner::find_input_idx_by_name(std::string_view name) const {

    Ort::AllocatorWithDefaultOptions allocator;

    for (const auto& [internal_tensor_name, idx_input] : _in_tensor_idx) {
        const std::string tensor_name = std::string(_model->GetInputNameAllocated(idx_input, allocator).get());

        if (tensor_name == name) {
            return idx_input;
        }
    }

    throw std::invalid_argument("Input not found: " + std::string(name));
}

size_t ONNXRTRunner::find_output_idx_by_name(std::string_view name) const {

    Ort::AllocatorWithDefaultOptions allocator;

    for (const auto& [internal_tensor_name, idx_output] : _out_tensor_idx) {
        const std::string tensor_name = std::string(_model->GetOutputNameAllocated(idx_output, allocator).get());

        if (tensor_name == name) {
            return idx_output;
        }
    }

    throw std::invalid_argument("Output not found: " + std::string(name));
}
}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_ONNXRT
