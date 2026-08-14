//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_LITERT

#include "runners/litert/litert_runner.h"

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_runtime_options.h"

#if defined(__ANDROID__) && defined(COMPUTE_FLOW_HAS_LITERT_GPU)
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/runtime/accelerators/gpu_static_registry.h"
#include "tflite/delegates/gpu/delegate.h"
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace compute_flow {
namespace {

constexpr LiteRtParamIndex k_signature = 0;
constexpr int32_t k_litert_warning_severity = 2;
constexpr uint32_t k_xnnpack_flags = 0x1 | 0x2 | 0x8 | 0x10 | 0x40;
constexpr uint32_t k_xnnpack_force_fp16 = 0x4;
constexpr std::string_view k_fp16_model_marker = "dummy_fp16_constant";
constexpr size_t k_model_scan_buffer_size = 64 * 1024;

#if defined(__ANDROID__) && defined(COMPUTE_FLOW_HAS_LITERT_GPU)
constexpr char k_gpu_accelerator_name[] = "TensorFlowLiteGpuAccelerator";
constexpr LiteRtApiVersion k_gpu_accelerator_version {1, 0, 0};

LiteRtStatus get_gpu_accelerator_name(LiteRtAccelerator accelerator, const char** name) {
    if (!accelerator || !name) return kLiteRtStatusErrorInvalidArgument;
    *name = k_gpu_accelerator_name;
    return kLiteRtStatusOk;
}

LiteRtStatus get_gpu_accelerator_version(LiteRtAccelerator accelerator, LiteRtApiVersion* version) {
    if (!accelerator || !version) return kLiteRtStatusErrorInvalidArgument;
    *version = k_gpu_accelerator_version;
    return kLiteRtStatusOk;
}

LiteRtStatus get_gpu_hardware_support(LiteRtAccelerator accelerator, LiteRtHwAcceleratorSet* hardware) {
    if (!accelerator || !hardware) return kLiteRtStatusErrorInvalidArgument;
    *hardware = kLiteRtHwAcceleratorGpu;
    return kLiteRtStatusOk;
}

LiteRtStatus is_gpu_jit_compilation_delegate(LiteRtAccelerator accelerator, bool* is_jit_compilation) {
    if (!accelerator || !is_jit_compilation) return kLiteRtStatusErrorInvalidArgument;
    *is_jit_compilation = true;
    return kLiteRtStatusOk;
}

void destroy_gpu_delegate(TfLiteOpaqueDelegate* delegate) {
    TfLiteGpuDelegateV2Delete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

LiteRtStatus create_gpu_delegate(LiteRtRuntimeContext*,
                                 LiteRtEnvironment,
                                 LiteRtAccelerator accelerator,
                                 LiteRtOptions,
                                 LiteRtDelegateWrapper* delegate_wrapper) {
    if (!accelerator || !delegate_wrapper) return kLiteRtStatusErrorInvalidArgument;

    const auto options = TfLiteGpuDelegateOptionsV2Default();
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (!delegate) return kLiteRtStatusErrorRuntimeFailure;

    const auto status = LiteRtWrapDelegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate), destroy_gpu_delegate, delegate_wrapper);
    if (status != kLiteRtStatusOk) TfLiteGpuDelegateV2Delete(delegate);
    return status;
}

LiteRtAcceleratorDef k_gpu_accelerator_definition {
    LITERT_ACCELERATOR_DEF_CURRENT_VERSION,
    get_gpu_accelerator_name,
    get_gpu_accelerator_version,
    get_gpu_hardware_support,
    is_gpu_jit_compilation_delegate,
    create_gpu_delegate,
    nullptr,
    nullptr,
    {},
};

void register_gpu_accelerator() {
    LiteRtStaticLinkedAcceleratorGpuDef = &k_gpu_accelerator_definition;
}
#endif

void check(LiteRtStatus status, const char* action) {
    if (status != kLiteRtStatusOk) {
        throw std::runtime_error(
            std::string(action) + ": " + LiteRtGetStatusString(status) + " (" +
            std::to_string(static_cast<int>(status)) + ")");
    }
}

bool model_has_fp16_weights(LiteRtModel model) {
    LiteRtParamIndex subgraph_count = 0;
    check(LiteRtGetNumModelSubgraphs(model, &subgraph_count), "Get LiteRT subgraphs");
    for (LiteRtParamIndex subgraph_index = 0; subgraph_index < subgraph_count; ++subgraph_index) {
        LiteRtSubgraph subgraph = nullptr;
        check(LiteRtGetModelSubgraph(model, subgraph_index, &subgraph), "Get LiteRT subgraph");
        LiteRtParamIndex op_count = 0;
        check(LiteRtGetNumSubgraphOps(subgraph, &op_count), "Get LiteRT subgraph ops");
        for (LiteRtParamIndex op_index = 0; op_index < op_count; ++op_index) {
            LiteRtOp op = nullptr;
            check(LiteRtGetSubgraphOp(subgraph, op_index, &op), "Get LiteRT op");
            LiteRtParamIndex input_count = 0;
            check(LiteRtGetNumOpInputs(op, &input_count), "Get LiteRT op inputs");
            for (LiteRtParamIndex input_index = 0; input_index < input_count; ++input_index) {
                LiteRtTensor tensor = nullptr;
                check(LiteRtGetOpInput(op, input_index, &tensor), "Get LiteRT op input");
                LiteRtRankedTensorType type{};
                check(LiteRtGetRankedTensorType(tensor, &type), "Get LiteRT op input type");
                if (type.element_type != kLiteRtElementTypeFloat16) continue;
                LiteRtWeights weights = nullptr;
                check(LiteRtGetTensorWeights(tensor, &weights), "Get LiteRT tensor weights");
                const void* data = nullptr;
                size_t size = 0;
                check(LiteRtGetWeightsBytes(weights, &data, &size), "Get LiteRT weight bytes");
                if (data && size) return true;
            }
        }
    }
    return false;
}

bool model_has_fp16_marker(const std::string& model_path) {
    // Existing FP16 bundles use a disconnected tensor as an execution-mode marker. LiteRT's graph
    // API omits disconnected tensors, so inspect the serialized model to preserve that contract.
    std::ifstream model(model_path, std::ios::binary);
    if (!model) return false;

    std::array<char, k_model_scan_buffer_size> chunk{};
    std::string window;
    window.reserve(chunk.size() + k_fp16_model_marker.size());
    while (model) {
        model.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        window.append(chunk.data(), static_cast<size_t>(model.gcount()));
        if (window.find(k_fp16_model_marker) != std::string::npos) return true;
        if (window.size() >= k_fp16_model_marker.size()) {
            window.erase(0, window.size() - k_fp16_model_marker.size() + 1);
        }
    }
    return false;
}

bool is_operator_profile_event(LiteRtProfilerEventType event_type) {
    switch (event_type) {
        case OPERATOR_INVOKE_EVENT:
        case DELEGATE_OPERATOR_INVOKE_EVENT:
        case DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT:
            return true;
        default:
            return false;
    }
}

DataType convert_type(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeFloat32: return DataType::F32;
        case kLiteRtElementTypeFloat16: return DataType::F16;
        case kLiteRtElementTypeBFloat16: return DataType::BF16;
        case kLiteRtElementTypeFloat64: return DataType::F64;
        case kLiteRtElementTypeInt64: return DataType::I64;
        case kLiteRtElementTypeUInt64: return DataType::U64;
        case kLiteRtElementTypeInt32: return DataType::I32;
        case kLiteRtElementTypeUInt32: return DataType::U32;
        case kLiteRtElementTypeInt8: return DataType::I8;
        case kLiteRtElementTypeUInt8: return DataType::U8;
        case kLiteRtElementTypeTfString: return DataType::String;
        case kLiteRtElementTypeBool: return DataType::Bool;
        default: throw std::invalid_argument("Unsupported LiteRT element type");
    }
}

LiteRtRankedTensorType ranked_type(LiteRtTensor tensor) {
    // LiteRT uses this type to pair a tensor's element type with its fixed-rank shape and strides.
    LiteRtRankedTensorType type{};
    check(LiteRtGetRankedTensorType(tensor, &type), "Get LiteRT tensor type");
    return type;
}

TensorFormat tensor_format(LiteRtElementType element_type, const LiteRtLayout& layout) {
    TensorFormat format;
    format.dt = convert_type(element_type);
    for (unsigned int i = 0; i < layout.rank; ++i) {
        format.shape.push_back(layout.dimensions[i] < 0 ? -1 : layout.dimensions[i]);
    }
    return format;
}

TensorFormat tensor_format(LiteRtTensor tensor) {
    const auto type = ranked_type(tensor);
    return tensor_format(type.element_type, type.layout);
}

LiteRtTensorBuffer create_buffer(LiteRtEnvironment environment,
                                 LiteRtCompiledModel compiled,
                                 LiteRtElementType element_type,
                                 const LiteRtLayout& layout,
                                 bool use_host_memory,
                                 bool input,
                                 size_t index) {
    LiteRtTensorBufferRequirements requirements = nullptr;
    const auto status = input
        ? LiteRtGetCompiledModelInputBufferRequirements(
              compiled, k_signature, static_cast<LiteRtParamIndex>(index), &requirements)
        : LiteRtGetCompiledModelOutputBufferRequirements(
              compiled, k_signature, static_cast<LiteRtParamIndex>(index), &requirements);
    check(status, "Get LiteRT buffer requirements");
    const LiteRtRankedTensorType type {element_type, layout};
    LiteRtTensorBuffer buffer = nullptr;
    LiteRtStatus create_status = kLiteRtStatusOk;
    if (use_host_memory) {
        size_t buffer_size = 0;
        check(
            LiteRtGetTensorBufferRequirementsBufferSize(requirements, &buffer_size),
            "Get LiteRT buffer size");
        create_status = LiteRtCreateManagedTensorBuffer(
            environment, kLiteRtTensorBufferTypeHostMemory, &type, buffer_size, &buffer);
    } else {
        create_status =
            LiteRtCreateManagedTensorBufferFromRequirements(environment, &type, requirements, &buffer);
    }
    check(create_status, "Create LiteRT buffer");
    return buffer;
}

std::vector<uint8_t> read_buffer(LiteRtTensorBuffer buffer) {
    size_t size = 0;
    check(LiteRtGetTensorBufferPackedSize(buffer, &size), "Get LiteRT buffer size");
    std::vector<uint8_t> contents(size);
    void* data = nullptr;
    check(LiteRtLockTensorBuffer(buffer, &data, kLiteRtTensorBufferLockModeRead), "Lock LiteRT buffer");
    std::memcpy(contents.data(), data, size);
    check(LiteRtUnlockTensorBuffer(buffer), "Unlock LiteRT buffer");
    return contents;
}

void write_buffer(LiteRtTensorBuffer buffer, const void* source, size_t source_size) {
    size_t size = 0;
    check(LiteRtGetTensorBufferPackedSize(buffer, &size), "Get LiteRT buffer size");
    if (source_size != size) throw std::invalid_argument("LiteRT buffer size mismatch");
    void* data = nullptr;
    check(LiteRtLockTensorBuffer(buffer, &data, kLiteRtTensorBufferLockModeWrite), "Lock LiteRT buffer");
    std::memcpy(data, source, size);
    check(LiteRtUnlockTensorBuffer(buffer), "Unlock LiteRT buffer");
}

void configure_cpu(LiteRtOptions options, const CpuOptions& cpu_options, bool enable_fp16) {
    if (cpu_options.num_threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("LiteRT CPU thread count is too large");
    }
    LrtCpuOptions* cpu = nullptr;
    check(LrtCreateCpuOptions(&cpu), "Create LiteRT CPU options");
    check(LrtSetCpuOptionsKernelMode(cpu, kLiteRtCpuKernelModeXnnpack), "Set LiteRT CPU mode");
    if (cpu_options.num_threads != 0) {
        check(
            LrtSetCpuOptionsNumThread(cpu, static_cast<int>(cpu_options.num_threads)),
            "Set LiteRT thread count");
    }
    check(
        LrtSetCpuOptionsXNNPackFlags(
            cpu,
            k_xnnpack_flags | (enable_fp16 ? k_xnnpack_force_fp16 : 0)),
        "Set LiteRT XNNPACK flags");
    const char* identifier = nullptr;
    void* payload = nullptr;
    void (*deleter)(void*) = nullptr;
    check(LrtGetOpaqueCpuOptionsData(cpu, &identifier, &payload, &deleter), "Serialize CPU options");
    LiteRtOpaqueOptions opaque = nullptr;
    check(LiteRtCreateOpaqueOptions(identifier, payload, deleter, &opaque), "Create opaque CPU options");
    const auto add_status = LiteRtAddOpaqueOptions(options, opaque);
    if (add_status != kLiteRtStatusOk) LiteRtDestroyOpaqueOptions(opaque);
    LrtDestroyCpuOptions(cpu);
    check(add_status, "Attach LiteRT CPU options");
}

void configure_profiling(LiteRtOptions options) {
    LrtRuntimeOptions* runtime_options = nullptr;
    check(LrtCreateRuntimeOptions(&runtime_options), "Create LiteRT runtime options");
    check(LrtSetRuntimeOptionsEnableProfiling(runtime_options, true), "Enable LiteRT profiling");
    const char* identifier = nullptr;
    void* payload = nullptr;
    void (*deleter)(void*) = nullptr;
    check(
        LrtGetOpaqueRuntimeOptionsData(runtime_options, &identifier, &payload, &deleter),
        "Serialize LiteRT runtime options");
    LiteRtOpaqueOptions opaque = nullptr;
    check(LiteRtCreateOpaqueOptions(identifier, payload, deleter, &opaque), "Create opaque runtime options");
    const auto add_status = LiteRtAddOpaqueOptions(options, opaque);
    if (add_status != kLiteRtStatusOk) LiteRtDestroyOpaqueOptions(opaque);
    LrtDestroyRuntimeOptions(runtime_options);
    check(add_status, "Attach LiteRT runtime options");
}

}  // namespace

struct LiteRTRunner::RuntimeState {
    LiteRtEnvironment environment {nullptr};
    LiteRtModel model {nullptr};
    LiteRtSignature signature {nullptr};
    LiteRtCompiledModel compiled {nullptr};
    LiteRtProfiler profiler {nullptr};
    std::vector<LiteRtTensorBuffer> inputs;
    std::vector<LiteRtTensorBuffer> outputs;
    std::vector<LiteRtElementType> input_types;
    std::vector<LiteRtElementType> output_types;
    std::vector<bool> input_initialized;
    std::vector<bool> output_locked;
    std::vector<ProfiledEventData> profile_events;
    bool use_host_memory {false};
    bool resized {false};
    bool has_run {false};

    ~RuntimeState() {
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (output_locked[i]) LiteRtUnlockTensorBuffer(outputs[i]);
        }
        if (compiled) LiteRtDestroyCompiledModel(compiled);
        for (auto buffer : inputs) LiteRtDestroyTensorBuffer(buffer);
        for (auto buffer : outputs) LiteRtDestroyTensorBuffer(buffer);
        if (model) LiteRtDestroyModel(model);
        if (environment) LiteRtDestroyEnvironment(environment);
    }
};

LiteRTRunner::LiteRTRunner() = default;
LiteRTRunner::~LiteRTRunner() = default;
LiteRTRunner::LiteRTRunner(LiteRTRunner&&) noexcept = default;
LiteRTRunner& LiteRTRunner::operator=(LiteRTRunner&&) noexcept = default;

void LiteRTRunner::setup(const ModelData& model, ProfilerView profiler) {
    if (model.acc_config.priorities.empty()) throw std::invalid_argument("LiteRT requires an accelerator");
    const auto& accelerator = model.acc_config.priorities.front();
#ifndef __ANDROID__
    if (accelerator.type == AcceleratorType::Gpu) {
        throw std::invalid_argument("LiteRT GPU acceleration is currently supported on Android only");
    }
#elif !defined(COMPUTE_FLOW_HAS_LITERT_GPU)
    if (accelerator.type == AcceleratorType::Gpu) {
        throw std::invalid_argument(
            "LiteRT GPU acceleration is disabled; build Compute Flow from source with "
            "COMPUTE_FLOW_ENABLE_LITERT_GPU=ON");
    }
#endif
    const bool use_gpu = accelerator.type == AcceleratorType::Gpu;
    const bool enable_cpu_fallback = use_gpu && model.acc_config.has_accelerator(AcceleratorType::Cpu);
    LiteRtHwAcceleratorSet hardware = kLiteRtHwAcceleratorNone;
    switch (accelerator.type) {
        case AcceleratorType::Cpu:
            hardware = kLiteRtHwAcceleratorCpu;
            break;
        case AcceleratorType::Gpu:
            hardware = kLiteRtHwAcceleratorGpu;
            if (enable_cpu_fallback) hardware |= kLiteRtHwAcceleratorCpu;
            break;
        default:
            throw std::invalid_argument("LiteRT received an unsupported accelerator");
    }
    if (accelerator.type == AcceleratorType::Gpu && std::get<GpuOptions>(accelerator.params).device_index != 0) {
        throw std::invalid_argument("LiteRT supports only the default GPU device");
    }
#if defined(__ANDROID__) && defined(COMPUTE_FLOW_HAS_LITERT_GPU)
    if (use_gpu) register_gpu_accelerator();
#endif
    auto runtime = std::make_unique<RuntimeState>();
    runtime->use_host_memory = accelerator.type == AcceleratorType::Cpu || enable_cpu_fallback;
    std::array<LiteRtEnvOption, 2> environment_options{};
    environment_options[0].tag = kLiteRtEnvOptionTagAutoRegisterAccelerators;
    environment_options[0].value.type = kLiteRtAnyTypeInt;
    environment_options[0].value.int_value = hardware;
    environment_options[1].tag = kLiteRtEnvOptionTagMinLoggerSeverity;
    environment_options[1].value.type = kLiteRtAnyTypeInt;
    environment_options[1].value.int_value = k_litert_warning_severity;
    check(
        LiteRtCreateEnvironment(environment_options.size(), environment_options.data(), &runtime->environment),
        "Create environment");
    check(
        LiteRtCreateModelFromFile(runtime->environment, model.model_path.c_str(), &runtime->model),
        "Load model");
    LiteRtParamIndex signatures = 0;
    check(LiteRtGetNumModelSignatures(runtime->model, &signatures), "Get signatures");
    if (signatures == 0) throw std::invalid_argument("LiteRT model has no signature");
    check(LiteRtGetModelSignature(runtime->model, k_signature, &runtime->signature), "Get signature");

    LiteRtOptions options = nullptr;
    check(LiteRtCreateOptions(&options), "Create compilation options");
    check(LiteRtSetOptionsHardwareAccelerators(options, hardware), "Select LiteRT accelerator");
    if (accelerator.type == AcceleratorType::Cpu) {
        const auto& cpu_options = std::get<CpuOptions>(accelerator.params);
        const bool enable_fp16 = cpu_options.enable_fp16 || model_has_fp16_weights(runtime->model) ||
            model_has_fp16_marker(model.model_path);
        configure_cpu(options, cpu_options, enable_fp16);
    } else if (enable_cpu_fallback) {
        for (const auto& fallback : model.acc_config.priorities) {
            if (fallback.type != AcceleratorType::Cpu) continue;
            const auto& cpu_options = std::get<CpuOptions>(fallback.params);
            const bool enable_fp16 = cpu_options.enable_fp16 || model_has_fp16_weights(runtime->model) ||
                model_has_fp16_marker(model.model_path);
            configure_cpu(options, cpu_options, enable_fp16);
            break;
        }
    }
    if (profiler.is_per_op_profiling_enabled()) configure_profiling(options);
    const auto compile_status =
        LiteRtCreateCompiledModel(runtime->environment, runtime->model, options, &runtime->compiled);
    LiteRtDestroyOptions(options);
    check(compile_status, "Compile model");
    if (profiler.is_per_op_profiling_enabled()) {
        check(LiteRtCompiledModelGetProfiler(runtime->compiled, &runtime->profiler), "Get profiler");
        if (!runtime->profiler) throw std::runtime_error("LiteRT profiler was not created");
    }

    LiteRtParamIndex input_count = 0;
    LiteRtParamIndex output_count = 0;
    check(LiteRtGetNumSignatureInputs(runtime->signature, &input_count), "Get input count");
    check(LiteRtGetNumSignatureOutputs(runtime->signature, &output_count), "Get output count");
    std::vector<TensorData> inputs(input_count);
    std::vector<TensorData> outputs(output_count);
    for (LiteRtParamIndex i = 0; i < input_count; ++i) {
        LiteRtTensor tensor;
        check(LiteRtGetSignatureInputTensorByIndex(runtime->signature, i, &tensor), "Get input");
        const auto type = ranked_type(tensor);
        inputs[i].format = tensor_format(type.element_type, type.layout);
        runtime->input_types.push_back(type.element_type);
        LiteRtLayout layout{};
        check(
            LiteRtGetCompiledModelInputTensorLayout(runtime->compiled, k_signature, i, &layout),
            "Get LiteRT input layout");
        runtime->inputs.push_back(create_buffer(
            runtime->environment,
            runtime->compiled,
            type.element_type,
            layout,
            runtime->use_host_memory,
            true,
            i));
    }
    std::vector<LiteRtLayout> output_layouts(output_count);
    check(
        LiteRtGetCompiledModelOutputTensorLayouts(
            runtime->compiled, k_signature, output_layouts.size(), output_layouts.data(), false),
        "Get LiteRT output layouts");
    for (LiteRtParamIndex i = 0; i < output_count; ++i) {
        LiteRtTensor tensor;
        check(LiteRtGetSignatureOutputTensorByIndex(runtime->signature, i, &tensor), "Get output");
        const auto type = ranked_type(tensor);
        outputs[i].format = tensor_format(type.element_type, type.layout);
        runtime->output_types.push_back(type.element_type);
        runtime->outputs.push_back(create_buffer(
            runtime->environment,
            runtime->compiled,
            type.element_type,
            output_layouts[i],
            runtime->use_host_memory,
            false,
            i));
    }
    runtime->input_initialized.assign(input_count, false);
    runtime->output_locked.assign(output_count, false);

    _runtime = std::move(runtime);
    _model_data = model;
    _model_name = model.model_name;
    _profiler = profiler;
    _inputs = std::move(inputs);
    _outputs = std::move(outputs);
    _states = model.states;
    _in_tensor_idx.clear();
    _out_tensor_idx.clear();
    for (const auto& [name, index] : model.inputs) map_input(name, static_cast<size_t>(index));
    for (const auto& [name, index] : model.outputs) map_output(name, static_cast<size_t>(index));
    for (const auto& [name, indices] : model.states) {
        map_input(name, static_cast<size_t>(indices.first));
        map_output(name, static_cast<size_t>(indices.second));
    }
}

void LiteRTRunner::run() {
    if (!_runtime) throw std::logic_error("LiteRT runner is not set up");
    if (_runtime->resized &&
        std::find(_runtime->input_initialized.begin(), _runtime->input_initialized.end(), false) !=
            _runtime->input_initialized.end()) {
        throw std::logic_error("All LiteRT inputs must be set after resizing a dynamic input");
    }
    unlock_outputs();
    auto model_scope = _profiler.measure_model("litert", _model_name);
    if (_runtime->profiler) {
        check(LiteRtResetProfiler(_runtime->profiler), "Reset profiler");
        check(LiteRtStartProfiler(_runtime->profiler), "Start profiler");
    }
    check(
        LiteRtRunCompiledModel(
            _runtime->compiled,
            k_signature,
            _runtime->inputs.size(),
            _runtime->inputs.data(),
            _runtime->outputs.size(),
            _runtime->outputs.data()),
        "Run model");
    _runtime->has_run = true;
    if (_runtime->profiler) {
        check(LiteRtStopProfiler(_runtime->profiler), "Stop profiler");
        int count = 0;
        check(LiteRtGetNumProfilerEvents(_runtime->profiler, &count), "Get profiler events");
        _runtime->profile_events.resize(static_cast<size_t>(count));
        if (count) {
            check(
                LiteRtGetProfilerEvents(_runtime->profiler, count, _runtime->profile_events.data()),
                "Read profiler events");
        }
        for (const auto& event : _runtime->profile_events) {
            if (!is_operator_profile_event(event.event_type)) continue;
            _profiler.add_nested_sample(
                "litert",
                _model_name,
                event.tag ? event.tag : "UNKNOWN",
                static_cast<long long>(event.elapsed_time_us) * 1'000);
        }
    }
    for (size_t i = 0; i < _outputs.size(); ++i) lock_output(i);
    for (const auto& [name, indices] : _states) {
        (void)indices;
        set_input(name, get_output(name));
    }
}

const std::string& LiteRTRunner::model_name() const { return _model_name; }

void LiteRTRunner::map_input(std::string_view key, size_t idx) {
    if (!_runtime || idx >= _inputs.size()) throw std::out_of_range("LiteRT input index out of range");
    if (!_in_tensor_idx.emplace(std::string(key), idx).second) throw std::invalid_argument("Duplicate input key");
}

void LiteRTRunner::map_output(std::string_view key, size_t idx) {
    if (!_runtime || idx >= _outputs.size()) throw std::out_of_range("LiteRT output index out of range");
    if (!_out_tensor_idx.emplace(std::string(key), idx).second) throw std::invalid_argument("Duplicate output key");
}

void LiteRTRunner::map_input(std::string_view key, std::string_view name) {
    map_input(key, find_input_idx_by_name(name));
}

void LiteRTRunner::map_output(std::string_view key, std::string_view name) {
    map_output(key, find_output_idx_by_name(name));
}

void LiteRTRunner::set_input(const std::string& key, const TensorData& tensor) {
    const auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) throw std::out_of_range("Input key not found: " + key);
    if (!tensor.data || tensor.format.has_dynamic_shape()) throw std::invalid_argument("Invalid input tensor");
    const size_t index = it->second;
    const TensorFormat& signature_format = _inputs[index].format;
    if (tensor.format.dt != signature_format.dt) throw std::invalid_argument("Input type mismatch: " + key);
    if (signature_format.has_dynamic_shape()) {
        if (tensor.format.shape.size() != signature_format.shape.size()) {
            throw std::invalid_argument("Input rank mismatch: " + key);
        }

        std::vector<int> dimensions;
        dimensions.reserve(tensor.format.shape.size());
        for (size_t i = 0; i < tensor.format.shape.size(); ++i) {
            const int64_t dimension = tensor.format.shape[i];
            if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("Invalid input dimension: " + key);
            }
            if (signature_format.shape[i] >= 0 && signature_format.shape[i] != dimension) {
                throw std::invalid_argument("Input shape mismatch: " + key);
            }
            dimensions.push_back(static_cast<int>(dimension));
        }

        const auto requires_resize = [&]() {
            LiteRtLayout current_layout{};
            check(
                LiteRtGetCompiledModelInputTensorLayout(
                    _runtime->compiled,
                    k_signature,
                    static_cast<LiteRtParamIndex>(index),
                    &current_layout),
                "Get LiteRT input layout");
            if (current_layout.rank != dimensions.size()) return true;
            for (size_t i = 0; i < dimensions.size(); ++i) {
                if (current_layout.dimensions[i] != dimensions[i]) return true;
            }
            return false;
        };

        bool resize = requires_resize();
        if (resize && _runtime->has_run) {
            // Delegates can retain allocations for the previous input shape after an invocation.
            // Recreate the compiled model before changing shape so those allocations stay valid.
            unlock_outputs();
            const ModelData model_data = _model_data;
            setup(model_data, _profiler);
            _runtime->resized = true;
            resize            = requires_resize();
        }
        if (resize) {
            std::vector<std::vector<uint8_t>> saved_inputs(_runtime->inputs.size());
            for (size_t i = 0; i < _runtime->inputs.size(); ++i) {
                if (i != index && _runtime->input_initialized[i]) {
                    saved_inputs[i] = read_buffer(_runtime->inputs[i]);
                }
            }

            unlock_outputs();
            check(
                LiteRtCompiledModelResizeInputTensor(
                    _runtime->compiled,
                    k_signature,
                    static_cast<LiteRtParamIndex>(index),
                    dimensions.data(),
                    dimensions.size()),
                "Resize LiteRT input");

            std::vector<LiteRtTensorBuffer> inputs;
            inputs.reserve(_runtime->inputs.size());
            for (size_t i = 0; i < _runtime->inputs.size(); ++i) {
                LiteRtLayout layout{};
                check(
                    LiteRtGetCompiledModelInputTensorLayout(
                        _runtime->compiled, k_signature, static_cast<LiteRtParamIndex>(i), &layout),
                    "Get resized LiteRT input layout");
                inputs.push_back(create_buffer(
                    _runtime->environment,
                    _runtime->compiled,
                    _runtime->input_types[i],
                    layout,
                    _runtime->use_host_memory,
                    true,
                    i));
            }

            std::vector<LiteRtLayout> output_layouts(_runtime->outputs.size());
            check(
                LiteRtGetCompiledModelOutputTensorLayouts(
                    _runtime->compiled,
                    k_signature,
                    output_layouts.size(),
                    output_layouts.data(),
                    true),
                "Update LiteRT output layouts");
            std::vector<LiteRtTensorBuffer> outputs;
            outputs.reserve(_runtime->outputs.size());
            for (size_t i = 0; i < _runtime->outputs.size(); ++i) {
                outputs.push_back(create_buffer(
                    _runtime->environment,
                    _runtime->compiled,
                    _runtime->output_types[i],
                    output_layouts[i],
                    _runtime->use_host_memory,
                    false,
                    i));
            }

            for (auto buffer : _runtime->inputs) LiteRtDestroyTensorBuffer(buffer);
            for (auto buffer : _runtime->outputs) LiteRtDestroyTensorBuffer(buffer);
            _runtime->inputs  = std::move(inputs);
            _runtime->outputs = std::move(outputs);
            for (size_t i = 0; i < saved_inputs.size(); ++i) {
                if (!saved_inputs[i].empty()) {
                    write_buffer(_runtime->inputs[i], saved_inputs[i].data(), saved_inputs[i].size());
                }
            }
            for (size_t i = 0; i < _outputs.size(); ++i) {
                _outputs[i].format = tensor_format(_runtime->output_types[i], output_layouts[i]);
            }
            _runtime->input_initialized[index] = false;
            _runtime->resized                  = true;
        }
    }

    const int64_t num_bytes = tensor.format.get_nbytes();
    if (num_bytes < 0) throw std::invalid_argument("Invalid input size: " + key);
    write_buffer(_runtime->inputs[index], tensor.data, static_cast<size_t>(num_bytes));
    _runtime->input_initialized[index] = true;
}

void LiteRTRunner::reset_states() {
    for (const auto& [name, indices] : _states) {
        (void)name;
        auto buffer = _runtime->inputs.at(static_cast<size_t>(indices.first));
        auto status = LiteRtClearTensorBuffer(buffer);
        if (status == kLiteRtStatusOk) {
            _runtime->input_initialized[static_cast<size_t>(indices.first)] = true;
            continue;
        }
        if (status != kLiteRtStatusErrorUnsupported) check(status, "Clear state");
        size_t size = 0;
        void* data = nullptr;
        check(LiteRtGetTensorBufferPackedSize(buffer, &size), "Get state size");
        check(LiteRtLockTensorBuffer(buffer, &data, kLiteRtTensorBufferLockModeWrite), "Lock state");
        std::memset(data, 0, size);
        check(LiteRtUnlockTensorBuffer(buffer), "Unlock state");
        _runtime->input_initialized[static_cast<size_t>(indices.first)] = true;
    }
}

TensorData LiteRTRunner::get_output(const std::string& key) {
    const auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) throw std::out_of_range("Output key not found: " + key);
    lock_output(it->second);
    return _outputs[it->second];
}

TensorFormat LiteRTRunner::get_input_format(const std::string& key) {
    return _inputs.at(_in_tensor_idx.at(key)).format;
}

TensorFormat LiteRTRunner::get_output_format(const std::string& key) {
    return _outputs.at(_out_tensor_idx.at(key)).format;
}

void LiteRTRunner::lock_output(size_t idx) {
    if (_runtime->output_locked[idx]) return;
    void* data = nullptr;
    check(LiteRtLockTensorBuffer(_runtime->outputs[idx], &data, kLiteRtTensorBufferLockModeRead), "Lock output");
    _runtime->output_locked[idx] = true;
    _outputs[idx].data = data;
}

void LiteRTRunner::unlock_outputs() {
    for (size_t i = 0; i < _runtime->outputs.size(); ++i) {
        if (!_runtime->output_locked[i]) continue;
        check(LiteRtUnlockTensorBuffer(_runtime->outputs[i]), "Unlock output");
        _runtime->output_locked[i] = false;
        _outputs[i].data = nullptr;
    }
}

size_t LiteRTRunner::find_input_idx_by_name(std::string_view name) const {
    LiteRtParamIndex count = 0;
    check(LiteRtGetNumSignatureInputs(_runtime->signature, &count), "Get inputs");
    for (LiteRtParamIndex i = 0; i < count; ++i) {
        const char* candidate = nullptr;
        check(LiteRtGetSignatureInputName(_runtime->signature, i, &candidate), "Get input name");
        if (candidate && name == candidate) return i;
    }
    throw std::invalid_argument("Input tensor not found");
}

size_t LiteRTRunner::find_output_idx_by_name(std::string_view name) const {
    LiteRtParamIndex count = 0;
    check(LiteRtGetNumSignatureOutputs(_runtime->signature, &count), "Get outputs");
    for (LiteRtParamIndex i = 0; i < count; ++i) {
        const char* candidate = nullptr;
        check(LiteRtGetSignatureOutputName(_runtime->signature, i, &candidate), "Get output name");
        if (candidate && name == candidate) return i;
    }
    throw std::invalid_argument("Output tensor not found");
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_LITERT
