//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2020 The TensorFlow Authors
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_LITERT

#include "runners/litert/litert_runner.h"

#include "compute_flow/check.h"

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
#include "tflite/c/common.h"

#if defined(__ANDROID__) && defined(COMPUTE_FLOW_HAS_LITERT_GPU)
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/runtime/accelerators/gpu_static_registry.h"
#include "tflite/delegates/gpu/delegate.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

// LiteRT 2.1.6 exports these interpreter symbols but does not stage their C API headers.
extern "C" {
struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteSignatureRunner;
struct TfLiteXNNPackDelegateWeightsCache;
struct TfLiteTelemetrySettings;

// LiteRT 2.1.6 omits the experimental telemetry C API headers from its package.
struct TfLiteTelemetryProfilerStruct {
    void* data;
    void (*ReportTelemetryEvent)(TfLiteTelemetryProfilerStruct*, const char*, uint64_t);
    void (*ReportTelemetryOpEvent)(TfLiteTelemetryProfilerStruct*, const char*, int64_t, int64_t, uint64_t);
    void (*ReportSettings)(TfLiteTelemetryProfilerStruct*, const char*, const TfLiteTelemetrySettings*);
    uint32_t (*ReportBeginOpInvokeEvent)(TfLiteTelemetryProfilerStruct*, const char*, int64_t, int64_t);
    void (*ReportEndOpInvokeEvent)(TfLiteTelemetryProfilerStruct*, uint32_t);
    void (*ReportOpInvokeEvent)(TfLiteTelemetryProfilerStruct*, const char*, uint64_t, int64_t, int64_t);
};

struct TfLiteXNNPackDelegateOptions {
    int32_t num_threads;
    uint32_t runtime_flags;
    uint32_t flags;
    TfLiteXNNPackDelegateWeightsCache* weights_cache;
    bool handle_variable_ops;
    const char* weight_cache_file_path;
    int weight_cache_file_descriptor;
    void* weight_cache_provider;
    bool weight_cache_lock_memory;
};

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
void TfLiteModelDelete(TfLiteModel* model);
TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate();
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options);
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options, int32_t num_threads);
void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* options, TfLiteOpaqueDelegate* delegate);
void TfLiteInterpreterOptionsSetTelemetryProfiler(TfLiteInterpreterOptions* options,
                                                  TfLiteTelemetryProfilerStruct* profiler);
TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();
TfLiteDelegate* TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions* options);
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);
TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteModel* model, const TfLiteInterpreterOptions* options);
void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
int32_t TfLiteInterpreterGetSignatureCount(const TfLiteInterpreter* interpreter);
const char* TfLiteInterpreterGetSignatureKey(const TfLiteInterpreter* interpreter, int32_t signature_index);
TfLiteSignatureRunner* TfLiteInterpreterGetSignatureRunner(const TfLiteInterpreter* interpreter,
                                                           const char* signature_key);
void TfLiteSignatureRunnerDelete(TfLiteSignatureRunner* signature_runner);
size_t TfLiteSignatureRunnerGetInputCount(const TfLiteSignatureRunner* signature_runner);
TfLiteTensor* TfLiteSignatureRunnerGetInputTensor(TfLiteSignatureRunner* signature_runner, const char* input_name);
size_t TfLiteSignatureRunnerGetOutputCount(const TfLiteSignatureRunner* signature_runner);
const char* TfLiteSignatureRunnerGetOutputName(const TfLiteSignatureRunner* signature_runner,
                                               int32_t output_index);
const TfLiteTensor* TfLiteSignatureRunnerGetOutputTensor(const TfLiteSignatureRunner* signature_runner,
                                                         const char* output_name);
TfLiteStatus TfLiteSignatureRunnerResizeInputTensor(TfLiteSignatureRunner* signature_runner,
                                                    const char* input_name,
                                                    const int* input_dims,
                                                    int32_t input_dims_size);
TfLiteStatus TfLiteSignatureRunnerAllocateTensors(TfLiteSignatureRunner* signature_runner);
TfLiteStatus TfLiteSignatureRunnerInvoke(TfLiteSignatureRunner* signature_runner);
TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);
int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);
int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index);
size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
void* TfLiteTensorData(const TfLiteTensor* tensor);
TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor, const void* input_data, size_t input_data_size);
}

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

struct InterpreterOperatorSample {
    std::string name;
    long long elapsed_ns;
};

struct InterpreterOperatorProfile {
    struct ActiveEvent {
        std::string name;
        std::chrono::steady_clock::time_point started;
    };

    std::mutex mutex;
    std::vector<ActiveEvent> active_events;
    std::vector<InterpreterOperatorSample> samples;
};

std::string interpreter_operator_name(const char* name, int64_t op_index, int64_t subgraph_index) {
    return std::string(name ? name : "UNKNOWN") + " [" + std::to_string(subgraph_index) + ":" +
        std::to_string(op_index) + "]";
}

void ignore_telemetry_event(TfLiteTelemetryProfilerStruct*, const char*, uint64_t) {}
void ignore_telemetry_op_event(TfLiteTelemetryProfilerStruct*, const char*, int64_t, int64_t, uint64_t) {}
void ignore_telemetry_settings(TfLiteTelemetryProfilerStruct*, const char*, const TfLiteTelemetrySettings*) {}

uint32_t begin_interpreter_operator(TfLiteTelemetryProfilerStruct* profiler,
                                    const char* name,
                                    int64_t op_index,
                                    int64_t subgraph_index) {
    auto& profile = *static_cast<InterpreterOperatorProfile*>(profiler->data);
    std::string label = interpreter_operator_name(name, op_index, subgraph_index);
    std::lock_guard<std::mutex> lock(profile.mutex);
    profile.active_events.push_back({std::move(label), std::chrono::steady_clock::now()});
    return static_cast<uint32_t>(profile.active_events.size());
}

void end_interpreter_operator(TfLiteTelemetryProfilerStruct* profiler, uint32_t handle) {
    const auto ended = std::chrono::steady_clock::now();
    auto& profile = *static_cast<InterpreterOperatorProfile*>(profiler->data);
    std::lock_guard<std::mutex> lock(profile.mutex);
    if (handle == 0 || handle > profile.active_events.size()) {
        return;
    }

    const auto& event = profile.active_events[handle - 1];
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(ended - event.started).count();
    profile.samples.push_back({event.name, elapsed_ns});
}

void record_interpreter_operator(TfLiteTelemetryProfilerStruct* profiler,
                                 const char* name,
                                 uint64_t elapsed_us,
                                 int64_t op_index,
                                 int64_t subgraph_index) {
    auto& profile = *static_cast<InterpreterOperatorProfile*>(profiler->data);
    std::string label = interpreter_operator_name(name, op_index, subgraph_index);
    std::lock_guard<std::mutex> lock(profile.mutex);
    profile.samples.push_back({std::move(label), static_cast<long long>(elapsed_us) * 1'000});
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

DataType convert_type_interpreter(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return DataType::F32;
        case kTfLiteFloat16: return DataType::F16;
        case kTfLiteBFloat16: return DataType::BF16;
        case kTfLiteFloat64: return DataType::F64;
        case kTfLiteInt64: return DataType::I64;
        case kTfLiteUInt64: return DataType::U64;
        case kTfLiteInt32: return DataType::I32;
        case kTfLiteUInt32: return DataType::U32;
        case kTfLiteInt8: return DataType::I8;
        case kTfLiteUInt8: return DataType::U8;
        case kTfLiteString: return DataType::String;
        case kTfLiteBool: return DataType::Bool;
        default: throw std::invalid_argument("Unsupported LiteRT interpreter element type");
    }
}

TensorFormat tensor_format_interpreter(const TfLiteTensor* tensor, bool signature_shape) {
    TensorFormat format;
    format.dt = convert_type_interpreter(TfLiteTensorType(tensor));
    const auto* dimensions = signature_shape && tensor->dims_signature && tensor->dims_signature->size
        ? tensor->dims_signature
        : tensor->dims;
    COMPUTE_FLOW_CHECK(dimensions != nullptr);
    for (int32_t i = 0; i < dimensions->size; ++i) {
        format.shape.push_back(dimensions->data[i]);
    }
    return format;
}

bool has_dynamic_output_interpreter(const std::string& model_path) {
    // Compiled output layouts can look fixed even when the serialized shape signature is dynamic.
    std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model(
        TfLiteModelCreateFromFile(model_path.c_str()), TfLiteModelDelete);
    if (!model) {
        return false;
    }

    std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)> interpreter(
        TfLiteInterpreterCreate(model.get(), nullptr), TfLiteInterpreterDelete);
    if (!interpreter || TfLiteInterpreterGetSignatureCount(interpreter.get()) == 0) {
        return false;
    }

    const char* signature_key = TfLiteInterpreterGetSignatureKey(interpreter.get(), k_signature);
    if (!signature_key) {
        return false;
    }

    std::unique_ptr<TfLiteSignatureRunner, decltype(&TfLiteSignatureRunnerDelete)> runner(
        TfLiteInterpreterGetSignatureRunner(interpreter.get(), signature_key), TfLiteSignatureRunnerDelete);
    if (!runner) {
        return false;
    }

    const size_t num_outputs = TfLiteSignatureRunnerGetOutputCount(runner.get());
    for (size_t i = 0; i < num_outputs; ++i) {
        const char* name = TfLiteSignatureRunnerGetOutputName(runner.get(), static_cast<int32_t>(i));
        COMPUTE_FLOW_CHECK(name != nullptr);
        const TfLiteTensor* tensor = TfLiteSignatureRunnerGetOutputTensor(runner.get(), name);
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        if (tensor->allocation_type == kTfLiteDynamic) {
            return true;
        }

        const auto* dimensions = tensor->dims_signature;
        if (!dimensions) {
            continue;
        }
        for (int32_t dim = 0; dim < dimensions->size; ++dim) {
            if (dimensions->data[dim] < 0) {
                return true;
            }
        }
    }
    return false;
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

struct LiteRTRunner::CompiledModelState {
    LiteRtEnvironment environment {nullptr};
    LiteRtModel model {nullptr};
    LiteRtSignature signature {nullptr};
    LiteRtCompiledModel compiled {nullptr};
    // LiteRT registrations retain custom-op name pointers from these options.
    LiteRtOptions options {nullptr};
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

    ~CompiledModelState() {
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (output_locked[i]) LiteRtUnlockTensorBuffer(outputs[i]);
        }
        if (compiled) LiteRtDestroyCompiledModel(compiled);
        if (options) LiteRtDestroyOptions(options);
        for (auto buffer : inputs) LiteRtDestroyTensorBuffer(buffer);
        for (auto buffer : outputs) LiteRtDestroyTensorBuffer(buffer);
        if (model) LiteRtDestroyModel(model);
        if (environment) LiteRtDestroyEnvironment(environment);
    }
};

struct LiteRTRunner::InterpreterState {
    TfLiteModel* model {nullptr};
    TfLiteInterpreter* interpreter {nullptr};
    TfLiteSignatureRunner* signature_runner {nullptr};
    TfLiteDelegate* xnnpack_delegate {nullptr};
    TfLiteTelemetryProfilerStruct telemetry {};
    InterpreterOperatorProfile operator_profile;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<uint8_t>> inputs;
    std::vector<std::vector<int32_t>> input_shapes;
    std::vector<bool> input_initialized;
    bool profile_operators {false};
    bool allocated {false};

    ~InterpreterState() {
        if (signature_runner) {
            TfLiteSignatureRunnerDelete(signature_runner);
        }
        if (interpreter) {
            TfLiteInterpreterDelete(interpreter);
        }
        if (xnnpack_delegate) {
            TfLiteXNNPackDelegateDelete(xnnpack_delegate);
        }
        if (model) {
            TfLiteModelDelete(model);
        }
    }
};

LiteRTRunner::LiteRTRunner() = default;
LiteRTRunner::~LiteRTRunner() = default;
LiteRTRunner::LiteRTRunner(LiteRTRunner&&) noexcept = default;
LiteRTRunner& LiteRTRunner::operator=(LiteRTRunner&&) noexcept = default;

void LiteRTRunner::setup(const ModelData& model, ProfilerView profiler) {
    // Dynamic outputs need their final shape after Invoke; fixed outputs keep the compiled path.
    _execution_mode = has_dynamic_output_interpreter(model.model_path) ? ExecutionMode::Interpreter
                                                                 : ExecutionMode::CompiledModel;
    switch (_execution_mode) {
        case ExecutionMode::CompiledModel:
            setup_compiled_model(model, profiler);
            break;
        case ExecutionMode::Interpreter:
            setup_interpreter(model, profiler);
            break;
    }
}

void LiteRTRunner::setup_compiled_model(const ModelData& model, ProfilerView profiler) {
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
    auto compiled_model = std::make_unique<CompiledModelState>();
    compiled_model->use_host_memory = accelerator.type == AcceleratorType::Cpu || enable_cpu_fallback;
    std::array<LiteRtEnvOption, 2> environment_options{};
    environment_options[0].tag = kLiteRtEnvOptionTagAutoRegisterAccelerators;
    environment_options[0].value.type = kLiteRtAnyTypeInt;
    environment_options[0].value.int_value = hardware;
    environment_options[1].tag = kLiteRtEnvOptionTagMinLoggerSeverity;
    environment_options[1].value.type = kLiteRtAnyTypeInt;
    environment_options[1].value.int_value = k_litert_warning_severity;
    check(
        LiteRtCreateEnvironment(environment_options.size(), environment_options.data(), &compiled_model->environment),
        "Create environment");
    check(
        LiteRtCreateModelFromFile(compiled_model->environment, model.model_path.c_str(), &compiled_model->model),
        "Load model");
    LiteRtParamIndex signatures = 0;
    check(LiteRtGetNumModelSignatures(compiled_model->model, &signatures), "Get signatures");
    if (signatures == 0) throw std::invalid_argument("LiteRT model has no signature");
    check(LiteRtGetModelSignature(compiled_model->model, k_signature, &compiled_model->signature), "Get signature");

    LiteRtOptions& options = compiled_model->options;
    check(LiteRtCreateOptions(&options), "Create compilation options");
    check(LiteRtSetOptionsHardwareAccelerators(options, hardware), "Select LiteRT accelerator");
    if (accelerator.type == AcceleratorType::Cpu) {
        const auto& cpu_options = std::get<CpuOptions>(accelerator.params);
        const bool enable_fp16 = cpu_options.enable_fp16 || model_has_fp16_weights(compiled_model->model) ||
            model_has_fp16_marker(model.model_path);
        configure_cpu(options, cpu_options, enable_fp16);
    } else if (enable_cpu_fallback) {
        for (const auto& fallback : model.acc_config.priorities) {
            if (fallback.type != AcceleratorType::Cpu) continue;
            const auto& cpu_options = std::get<CpuOptions>(fallback.params);
            const bool enable_fp16 = cpu_options.enable_fp16 || model_has_fp16_weights(compiled_model->model) ||
                model_has_fp16_marker(model.model_path);
            configure_cpu(options, cpu_options, enable_fp16);
            break;
        }
    }
    if (profiler.is_per_op_profiling_enabled()) configure_profiling(options);
    const auto compile_status =
        LiteRtCreateCompiledModel(compiled_model->environment, compiled_model->model, options, &compiled_model->compiled);
    check(compile_status, "Compile model");
    if (profiler.is_per_op_profiling_enabled()) {
        check(LiteRtCompiledModelGetProfiler(compiled_model->compiled, &compiled_model->profiler), "Get profiler");
        if (!compiled_model->profiler) throw std::runtime_error("LiteRT profiler was not created");
    }

    LiteRtParamIndex input_count = 0;
    LiteRtParamIndex output_count = 0;
    check(LiteRtGetNumSignatureInputs(compiled_model->signature, &input_count), "Get input count");
    check(LiteRtGetNumSignatureOutputs(compiled_model->signature, &output_count), "Get output count");
    std::vector<TensorData> inputs(input_count);
    std::vector<TensorData> outputs(output_count);
    for (LiteRtParamIndex i = 0; i < input_count; ++i) {
        LiteRtTensor tensor;
        check(LiteRtGetSignatureInputTensorByIndex(compiled_model->signature, i, &tensor), "Get input");
        const auto type = ranked_type(tensor);
        inputs[i].format = tensor_format(type.element_type, type.layout);
        compiled_model->input_types.push_back(type.element_type);
        LiteRtLayout layout{};
        check(
            LiteRtGetCompiledModelInputTensorLayout(compiled_model->compiled, k_signature, i, &layout),
            "Get LiteRT input layout");
        compiled_model->inputs.push_back(create_buffer(
            compiled_model->environment,
            compiled_model->compiled,
            type.element_type,
            layout,
            compiled_model->use_host_memory,
            true,
            i));
    }
    std::vector<LiteRtLayout> output_layouts(output_count);
    check(
        LiteRtGetCompiledModelOutputTensorLayouts(
            compiled_model->compiled, k_signature, output_layouts.size(), output_layouts.data(), false),
        "Get LiteRT output layouts");
    for (LiteRtParamIndex i = 0; i < output_count; ++i) {
        LiteRtTensor tensor;
        check(LiteRtGetSignatureOutputTensorByIndex(compiled_model->signature, i, &tensor), "Get output");
        const auto type = ranked_type(tensor);
        outputs[i].format = tensor_format(type.element_type, type.layout);
        compiled_model->output_types.push_back(type.element_type);
        compiled_model->outputs.push_back(create_buffer(
            compiled_model->environment,
            compiled_model->compiled,
            type.element_type,
            output_layouts[i],
            compiled_model->use_host_memory,
            false,
            i));
    }
    compiled_model->input_initialized.assign(input_count, false);
    compiled_model->output_locked.assign(output_count, false);

    _interpreter.reset();
    _compiled_model = std::move(compiled_model);
    _compiled_model_data = model;
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

void LiteRTRunner::setup_interpreter(const ModelData& model, ProfilerView profiler) {
    if (model.acc_config.priorities.empty()) {
        throw std::invalid_argument("LiteRT requires an accelerator");
    }
    const auto& accelerator = model.acc_config.priorities.front();
    if (accelerator.type != AcceleratorType::Cpu) {
        throw std::invalid_argument("LiteRT interpreter requires CPU acceleration");
    }

    const auto& cpu_options = std::get<CpuOptions>(accelerator.params);
    if (cpu_options.num_threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("LiteRT CPU thread count is too large");
    }

    auto interpreter = std::make_unique<InterpreterState>();
    interpreter->model = TfLiteModelCreateFromFile(model.model_path.c_str());
    COMPUTE_FLOW_CHECK(interpreter->model != nullptr);

    // ModelData indices use LiteRT's signature order, which can differ from interpreter order.
    // The same model view also identifies FP16 weights for CPU delegate configuration.
    LiteRtEnvOption logger_option{};
    logger_option.tag = kLiteRtEnvOptionTagMinLoggerSeverity;
    logger_option.value.type = kLiteRtAnyTypeInt;
    logger_option.value.int_value = k_litert_warning_severity;
    LiteRtEnvironment signature_environment = nullptr;
    check(LiteRtCreateEnvironment(1, &logger_option, &signature_environment), "Create signature environment");
    std::unique_ptr<LiteRtEnvironmentT, decltype(&LiteRtDestroyEnvironment)> environment(
        signature_environment, LiteRtDestroyEnvironment);
    LiteRtModel signature_model = nullptr;
    check(
        LiteRtCreateModelFromFile(environment.get(), model.model_path.c_str(), &signature_model),
        "Load signature model");
    std::unique_ptr<LiteRtModelT, decltype(&LiteRtDestroyModel)> signature_source(
        signature_model, LiteRtDestroyModel);
    LiteRtSignature signature = nullptr;
    check(LiteRtGetModelSignature(signature_source.get(), k_signature, &signature), "Get signature");

    // Match the compiled path's FP16 policy for the XNNPACK delegate.
    const bool enable_fp16 = cpu_options.enable_fp16 || model_has_fp16_weights(signature_source.get()) ||
        model_has_fp16_marker(model.model_path);
    std::unique_ptr<TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)> options(
        TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
    COMPUTE_FLOW_CHECK(options != nullptr);
    if (cpu_options.num_threads != 0) {
        TfLiteInterpreterOptionsSetNumThreads(options.get(), static_cast<int32_t>(cpu_options.num_threads));
    }

    if (profiler.is_enabled() && profiler.is_per_op_profiling_enabled()) {
        interpreter->telemetry = {
            &interpreter->operator_profile,
            ignore_telemetry_event,
            ignore_telemetry_op_event,
            ignore_telemetry_settings,
            begin_interpreter_operator,
            end_interpreter_operator,
            record_interpreter_operator,
        };
        TfLiteInterpreterOptionsSetTelemetryProfiler(options.get(), &interpreter->telemetry);
        interpreter->profile_operators = true;
    }

    auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
    delegate_options.flags |= k_xnnpack_flags | (enable_fp16 ? k_xnnpack_force_fp16 : 0);
    if (cpu_options.num_threads != 0) {
        delegate_options.num_threads = static_cast<int32_t>(cpu_options.num_threads);
    }
    interpreter->xnnpack_delegate = TfLiteXNNPackDelegateCreate(&delegate_options);
    COMPUTE_FLOW_CHECK(interpreter->xnnpack_delegate != nullptr);
    TfLiteInterpreterOptionsAddDelegate(options.get(), interpreter->xnnpack_delegate);
    interpreter->interpreter = TfLiteInterpreterCreate(interpreter->model, options.get());
    COMPUTE_FLOW_CHECK(interpreter->interpreter != nullptr);
    COMPUTE_FLOW_CHECK(TfLiteInterpreterGetSignatureCount(interpreter->interpreter) > 0);

    const char* signature_key = TfLiteInterpreterGetSignatureKey(interpreter->interpreter, k_signature);
    COMPUTE_FLOW_CHECK(signature_key != nullptr);
    interpreter->signature_runner = TfLiteInterpreterGetSignatureRunner(interpreter->interpreter, signature_key);
    COMPUTE_FLOW_CHECK(interpreter->signature_runner != nullptr);

    LiteRtParamIndex num_inputs = 0;
    LiteRtParamIndex num_outputs = 0;
    check(LiteRtGetNumSignatureInputs(signature, &num_inputs), "Get input count");
    check(LiteRtGetNumSignatureOutputs(signature, &num_outputs), "Get output count");
    COMPUTE_FLOW_CHECK(num_inputs == TfLiteSignatureRunnerGetInputCount(interpreter->signature_runner));
    COMPUTE_FLOW_CHECK(num_outputs == TfLiteSignatureRunnerGetOutputCount(interpreter->signature_runner));

    std::vector<TensorData> inputs(num_inputs);
    std::vector<TensorData> outputs(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
        const char* name = nullptr;
        check(LiteRtGetSignatureInputName(signature, i, &name), "Get input name");
        COMPUTE_FLOW_CHECK(name != nullptr);
        interpreter->input_names.emplace_back(name);
        auto* tensor = TfLiteSignatureRunnerGetInputTensor(interpreter->signature_runner, name);
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        inputs[i].format = tensor_format_interpreter(tensor, true);
    }
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
        const char* name = nullptr;
        check(LiteRtGetSignatureOutputName(signature, i, &name), "Get output name");
        COMPUTE_FLOW_CHECK(name != nullptr);
        interpreter->output_names.emplace_back(name);
        const auto* tensor = TfLiteSignatureRunnerGetOutputTensor(interpreter->signature_runner, name);
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        outputs[i].format = tensor_format_interpreter(tensor, true);
    }
    interpreter->inputs.resize(num_inputs);
    interpreter->input_shapes.resize(num_inputs);
    interpreter->input_initialized.assign(num_inputs, false);

    _compiled_model.reset();
    _interpreter = std::move(interpreter);
    _model_name = model.model_name;
    _profiler = profiler;
    _inputs = std::move(inputs);
    _outputs = std::move(outputs);
    _states = model.states;
    _in_tensor_idx.clear();
    _out_tensor_idx.clear();
    for (const auto& [name, index] : model.inputs) {
        map_input(name, static_cast<size_t>(index));
    }
    for (const auto& [name, index] : model.outputs) {
        map_output(name, static_cast<size_t>(index));
    }
    for (const auto& [name, indices] : model.states) {
        map_input(name, static_cast<size_t>(indices.first));
        map_output(name, static_cast<size_t>(indices.second));
    }
}

void LiteRTRunner::run() {
    switch (_execution_mode) {
        case ExecutionMode::CompiledModel:
            run_compiled_model();
            break;
        case ExecutionMode::Interpreter:
            run_interpreter();
            break;
    }
}

void LiteRTRunner::run_compiled_model() {
    if (!_compiled_model) throw std::logic_error("LiteRT runner is not set up");
    if (_compiled_model->resized &&
        std::find(_compiled_model->input_initialized.begin(), _compiled_model->input_initialized.end(), false) !=
            _compiled_model->input_initialized.end()) {
        throw std::logic_error("All LiteRT inputs must be set after resizing a dynamic input");
    }
    unlock_outputs_compiled_model();
    auto model_scope = _profiler.measure_model("litert", _model_name);
    if (_compiled_model->profiler) {
        check(LiteRtResetProfiler(_compiled_model->profiler), "Reset profiler");
        check(LiteRtStartProfiler(_compiled_model->profiler), "Start profiler");
    }
    check(
        LiteRtRunCompiledModel(
            _compiled_model->compiled,
            k_signature,
            _compiled_model->inputs.size(),
            _compiled_model->inputs.data(),
            _compiled_model->outputs.size(),
            _compiled_model->outputs.data()),
        "Run model");
    _compiled_model->has_run = true;
    if (_compiled_model->profiler) {
        check(LiteRtStopProfiler(_compiled_model->profiler), "Stop profiler");
        int count = 0;
        check(LiteRtGetNumProfilerEvents(_compiled_model->profiler, &count), "Get profiler events");
        _compiled_model->profile_events.resize(static_cast<size_t>(count));
        if (count) {
            check(
                LiteRtGetProfilerEvents(_compiled_model->profiler, count, _compiled_model->profile_events.data()),
                "Read profiler events");
        }
        for (const auto& event : _compiled_model->profile_events) {
            if (!is_operator_profile_event(event.event_type)) continue;
            _profiler.add_nested_sample(
                "litert",
                _model_name,
                event.tag ? event.tag : "UNKNOWN",
                static_cast<long long>(event.elapsed_time_us) * 1'000);
        }
    }
    for (size_t i = 0; i < _outputs.size(); ++i) lock_output_compiled_model(i);
    for (const auto& [name, indices] : _states) {
        (void)indices;
        set_input(name, get_output(name));
    }
}

void LiteRTRunner::run_interpreter() {
    if (!_interpreter) {
        throw std::logic_error("LiteRT runner is not set up");
    }
    if (std::find(_interpreter->input_initialized.begin(), _interpreter->input_initialized.end(), false) !=
        _interpreter->input_initialized.end()) {
        throw std::logic_error("All LiteRT interpreter inputs must be set before running");
    }

    // Resizing can invalidate tensor storage, so clear old output pointers before preparing inputs.
    for (auto& output : _outputs) {
        output.data = nullptr;
    }
    bool resized = false;
    for (size_t i = 0; i < _inputs.size(); ++i) {
        auto* tensor = TfLiteSignatureRunnerGetInputTensor(
            _interpreter->signature_runner, _interpreter->input_names[i].c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        const auto& shape = _interpreter->input_shapes[i];
        bool shape_changed = TfLiteTensorNumDims(tensor) != static_cast<int32_t>(shape.size());
        for (size_t dim = 0; !shape_changed && dim < shape.size(); ++dim) {
            shape_changed = TfLiteTensorDim(tensor, static_cast<int32_t>(dim)) != shape[dim];
        }
        if (shape_changed) {
            COMPUTE_FLOW_CHECK(
                TfLiteSignatureRunnerResizeInputTensor(
                    _interpreter->signature_runner,
                    _interpreter->input_names[i].c_str(),
                    shape.data(),
                    static_cast<int32_t>(shape.size())) == kTfLiteOk);
            resized = true;
        }
    }
    if (resized || !_interpreter->allocated) {
        COMPUTE_FLOW_CHECK(TfLiteSignatureRunnerAllocateTensors(_interpreter->signature_runner) == kTfLiteOk);
        _interpreter->allocated = true;
    }
    for (size_t i = 0; i < _inputs.size(); ++i) {
        auto* tensor = TfLiteSignatureRunnerGetInputTensor(
            _interpreter->signature_runner, _interpreter->input_names[i].c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        const auto& data = _interpreter->inputs[i];
        COMPUTE_FLOW_CHECK(TfLiteTensorCopyFromBuffer(tensor, data.data(), data.size()) == kTfLiteOk);
    }

    auto model_scope = _profiler.measure_model("litert", _model_name);
    if (_interpreter->profile_operators) {
        auto& profile = _interpreter->operator_profile;
        std::lock_guard<std::mutex> lock(profile.mutex);
        profile.active_events.clear();
        profile.samples.clear();
    }
    COMPUTE_FLOW_CHECK(TfLiteSignatureRunnerInvoke(_interpreter->signature_runner) == kTfLiteOk);
    if (_interpreter->profile_operators) {
        auto& profile = _interpreter->operator_profile;
        std::vector<InterpreterOperatorSample> samples;
        {
            std::lock_guard<std::mutex> lock(profile.mutex);
            samples.swap(profile.samples);
        }
        for (const auto& sample : samples) {
            _profiler.add_nested_sample("litert", _model_name, sample.name, sample.elapsed_ns);
        }
    }
    // The interpreter owns output memory and reveals dynamic output dimensions after invocation.
    for (size_t i = 0; i < _outputs.size(); ++i) {
        const auto* tensor = TfLiteSignatureRunnerGetOutputTensor(
            _interpreter->signature_runner, _interpreter->output_names[i].c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        _outputs[i].format = tensor_format_interpreter(tensor, false);
        _outputs[i].data = TfLiteTensorData(tensor);
        COMPUTE_FLOW_CHECK(_outputs[i].data != nullptr || TfLiteTensorByteSize(tensor) == 0);
    }
    for (const auto& [name, indices] : _states) {
        (void)indices;
        set_input(name, get_output(name));
    }
}

const std::string& LiteRTRunner::model_name() const { return _model_name; }

void LiteRTRunner::map_input(std::string_view key, size_t idx) {
    if ((_execution_mode == ExecutionMode::CompiledModel && !_compiled_model) ||
        (_execution_mode == ExecutionMode::Interpreter && !_interpreter) || idx >= _inputs.size()) {
        throw std::out_of_range("LiteRT input index out of range");
    }
    if (!_in_tensor_idx.emplace(std::string(key), idx).second) throw std::invalid_argument("Duplicate input key");
}

void LiteRTRunner::map_output(std::string_view key, size_t idx) {
    if ((_execution_mode == ExecutionMode::CompiledModel && !_compiled_model) ||
        (_execution_mode == ExecutionMode::Interpreter && !_interpreter) || idx >= _outputs.size()) {
        throw std::out_of_range("LiteRT output index out of range");
    }
    if (!_out_tensor_idx.emplace(std::string(key), idx).second) throw std::invalid_argument("Duplicate output key");
}

void LiteRTRunner::map_input(std::string_view key, std::string_view name) {
    const size_t index = _execution_mode == ExecutionMode::CompiledModel
        ? find_input_idx_by_name_compiled_model(name)
        : find_input_idx_by_name_interpreter(name);
    map_input(key, index);
}

void LiteRTRunner::map_output(std::string_view key, std::string_view name) {
    const size_t index = _execution_mode == ExecutionMode::CompiledModel
        ? find_output_idx_by_name_compiled_model(name)
        : find_output_idx_by_name_interpreter(name);
    map_output(key, index);
}

void LiteRTRunner::set_input(const std::string& key, const TensorData& tensor) {
    switch (_execution_mode) {
        case ExecutionMode::CompiledModel: set_input_compiled_model(key, tensor); break;
        case ExecutionMode::Interpreter: set_input_interpreter(key, tensor); break;
    }
}

void LiteRTRunner::set_input_interpreter(const std::string& key, const TensorData& tensor) {
    const auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }
    if (!tensor.data || tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("Invalid input tensor");
    }
    const size_t index = it->second;
    const TensorFormat& signature_format = _inputs[index].format;
    if (tensor.format.dt != signature_format.dt) {
        throw std::invalid_argument("Input type mismatch: " + key);
    }
    if (tensor.format.shape.size() != signature_format.shape.size()) {
        throw std::invalid_argument("Input rank mismatch: " + key);
    }

    auto& shape = _interpreter->input_shapes[index];
    shape.clear();
    for (size_t i = 0; i < tensor.format.shape.size(); ++i) {
        const int64_t dimension = tensor.format.shape[i];
        if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("Invalid input dimension: " + key);
        }
        if (signature_format.shape[i] >= 0 && signature_format.shape[i] != dimension) {
            throw std::invalid_argument("Input shape mismatch: " + key);
        }
        shape.push_back(static_cast<int32_t>(dimension));
    }

    const int64_t num_bytes = tensor.format.get_nbytes();
    if (num_bytes < 0) {
        throw std::invalid_argument("Invalid input size: " + key);
    }
    auto& data = _interpreter->inputs[index];
    // Keep caller data across ResizeInputTensor and AllocateTensors, which may replace input storage.
    data.resize(static_cast<size_t>(num_bytes));
    std::memcpy(data.data(), tensor.data, data.size());
    _interpreter->input_initialized[index] = true;
}

void LiteRTRunner::set_input_compiled_model(const std::string& key, const TensorData& tensor) {
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
                    _compiled_model->compiled,
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
        if (resize && _compiled_model->has_run) {
            // Delegates can retain allocations for the previous input shape after an invocation.
            // Recreate the compiled model before changing shape so those allocations stay valid.
            unlock_outputs_compiled_model();
            const ModelData model_data = _compiled_model_data;
            setup_compiled_model(model_data, _profiler);
            _compiled_model->resized = true;
            resize            = requires_resize();
        }
        if (resize) {
            std::vector<std::vector<uint8_t>> saved_inputs(_compiled_model->inputs.size());
            for (size_t i = 0; i < _compiled_model->inputs.size(); ++i) {
                if (i != index && _compiled_model->input_initialized[i]) {
                    saved_inputs[i] = read_buffer(_compiled_model->inputs[i]);
                }
            }

            unlock_outputs_compiled_model();
            check(
                LiteRtCompiledModelResizeInputTensor(
                    _compiled_model->compiled,
                    k_signature,
                    static_cast<LiteRtParamIndex>(index),
                    dimensions.data(),
                    dimensions.size()),
                "Resize LiteRT input");

            std::vector<LiteRtTensorBuffer> inputs;
            inputs.reserve(_compiled_model->inputs.size());
            for (size_t i = 0; i < _compiled_model->inputs.size(); ++i) {
                LiteRtLayout layout{};
                check(
                    LiteRtGetCompiledModelInputTensorLayout(
                        _compiled_model->compiled, k_signature, static_cast<LiteRtParamIndex>(i), &layout),
                    "Get resized LiteRT input layout");
                inputs.push_back(create_buffer(
                    _compiled_model->environment,
                    _compiled_model->compiled,
                    _compiled_model->input_types[i],
                    layout,
                    _compiled_model->use_host_memory,
                    true,
                    i));
            }

            std::vector<LiteRtLayout> output_layouts(_compiled_model->outputs.size());
            check(
                LiteRtGetCompiledModelOutputTensorLayouts(
                    _compiled_model->compiled,
                    k_signature,
                    output_layouts.size(),
                    output_layouts.data(),
                    true),
                "Update LiteRT output layouts");
            std::vector<LiteRtTensorBuffer> outputs;
            outputs.reserve(_compiled_model->outputs.size());
            for (size_t i = 0; i < _compiled_model->outputs.size(); ++i) {
                outputs.push_back(create_buffer(
                    _compiled_model->environment,
                    _compiled_model->compiled,
                    _compiled_model->output_types[i],
                    output_layouts[i],
                    _compiled_model->use_host_memory,
                    false,
                    i));
            }

            for (auto buffer : _compiled_model->inputs) LiteRtDestroyTensorBuffer(buffer);
            for (auto buffer : _compiled_model->outputs) LiteRtDestroyTensorBuffer(buffer);
            _compiled_model->inputs  = std::move(inputs);
            _compiled_model->outputs = std::move(outputs);
            for (size_t i = 0; i < saved_inputs.size(); ++i) {
                if (!saved_inputs[i].empty()) {
                    write_buffer(_compiled_model->inputs[i], saved_inputs[i].data(), saved_inputs[i].size());
                }
            }
            for (size_t i = 0; i < _outputs.size(); ++i) {
                _outputs[i].format = tensor_format(_compiled_model->output_types[i], output_layouts[i]);
            }
            _compiled_model->input_initialized[index] = false;
            _compiled_model->resized                  = true;
        }
    }

    const int64_t num_bytes = tensor.format.get_nbytes();
    if (num_bytes < 0) throw std::invalid_argument("Invalid input size: " + key);
    write_buffer(_compiled_model->inputs[index], tensor.data, static_cast<size_t>(num_bytes));
    _compiled_model->input_initialized[index] = true;
}

void LiteRTRunner::reset_states() {
    switch (_execution_mode) {
        case ExecutionMode::CompiledModel: reset_states_compiled_model(); break;
        case ExecutionMode::Interpreter: reset_states_interpreter(); break;
    }
}

void LiteRTRunner::reset_states_interpreter() {
    for (const auto& [name, indices] : _states) {
        (void)name;
        const size_t index = static_cast<size_t>(indices.first);
        auto* tensor = TfLiteSignatureRunnerGetInputTensor(
            _interpreter->signature_runner, _interpreter->input_names.at(index).c_str());
        COMPUTE_FLOW_CHECK(tensor != nullptr);
        const auto format = tensor_format_interpreter(tensor, false);
        const int64_t num_bytes = format.get_nbytes();
        COMPUTE_FLOW_CHECK(num_bytes >= 0);
        _interpreter->inputs[index].assign(static_cast<size_t>(num_bytes), 0);
        auto& shape = _interpreter->input_shapes[index];
        shape.clear();
        for (int64_t dimension : format.shape) {
            shape.push_back(static_cast<int32_t>(dimension));
        }
        _interpreter->input_initialized[index] = true;
    }
}

void LiteRTRunner::reset_states_compiled_model() {
    for (const auto& [name, indices] : _states) {
        (void)name;
        auto buffer = _compiled_model->inputs.at(static_cast<size_t>(indices.first));
        auto status = LiteRtClearTensorBuffer(buffer);
        if (status == kLiteRtStatusOk) {
            _compiled_model->input_initialized[static_cast<size_t>(indices.first)] = true;
            continue;
        }
        if (status != kLiteRtStatusErrorUnsupported) check(status, "Clear state");
        size_t size = 0;
        void* data = nullptr;
        check(LiteRtGetTensorBufferPackedSize(buffer, &size), "Get state size");
        check(LiteRtLockTensorBuffer(buffer, &data, kLiteRtTensorBufferLockModeWrite), "Lock state");
        std::memset(data, 0, size);
        check(LiteRtUnlockTensorBuffer(buffer), "Unlock state");
        _compiled_model->input_initialized[static_cast<size_t>(indices.first)] = true;
    }
}

TensorData LiteRTRunner::get_output(const std::string& key) {
    switch (_execution_mode) {
        case ExecutionMode::CompiledModel: return get_output_compiled_model(key);
        case ExecutionMode::Interpreter: return get_output_interpreter(key);
    }
    throw std::logic_error("Invalid LiteRT execution mode");
}

TensorData LiteRTRunner::get_output_interpreter(const std::string& key) {
    const auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }
    return _outputs[it->second];
}

TensorData LiteRTRunner::get_output_compiled_model(const std::string& key) {
    const auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) throw std::out_of_range("Output key not found: " + key);
    lock_output_compiled_model(it->second);
    return _outputs[it->second];
}

TensorFormat LiteRTRunner::get_input_format(const std::string& key) {
    return _inputs.at(_in_tensor_idx.at(key)).format;
}

TensorFormat LiteRTRunner::get_output_format(const std::string& key) {
    return _outputs.at(_out_tensor_idx.at(key)).format;
}

void LiteRTRunner::lock_output_compiled_model(size_t idx) {
    if (_compiled_model->output_locked[idx]) return;
    void* data = nullptr;
    check(LiteRtLockTensorBuffer(_compiled_model->outputs[idx], &data, kLiteRtTensorBufferLockModeRead), "Lock output");
    _compiled_model->output_locked[idx] = true;
    _outputs[idx].data = data;
}

void LiteRTRunner::unlock_outputs_compiled_model() {
    for (size_t i = 0; i < _compiled_model->outputs.size(); ++i) {
        if (!_compiled_model->output_locked[i]) continue;
        check(LiteRtUnlockTensorBuffer(_compiled_model->outputs[i]), "Unlock output");
        _compiled_model->output_locked[i] = false;
        _outputs[i].data = nullptr;
    }
}

size_t LiteRTRunner::find_input_idx_by_name_compiled_model(std::string_view name) const {
    LiteRtParamIndex count = 0;
    check(LiteRtGetNumSignatureInputs(_compiled_model->signature, &count), "Get inputs");
    for (LiteRtParamIndex i = 0; i < count; ++i) {
        const char* candidate = nullptr;
        check(LiteRtGetSignatureInputName(_compiled_model->signature, i, &candidate), "Get input name");
        if (candidate && name == candidate) return i;
    }
    throw std::invalid_argument("Input tensor not found");
}

size_t LiteRTRunner::find_output_idx_by_name_compiled_model(std::string_view name) const {
    LiteRtParamIndex count = 0;
    check(LiteRtGetNumSignatureOutputs(_compiled_model->signature, &count), "Get outputs");
    for (LiteRtParamIndex i = 0; i < count; ++i) {
        const char* candidate = nullptr;
        check(LiteRtGetSignatureOutputName(_compiled_model->signature, i, &candidate), "Get output name");
        if (candidate && name == candidate) return i;
    }
    throw std::invalid_argument("Output tensor not found");
}

size_t LiteRTRunner::find_input_idx_by_name_interpreter(std::string_view name) const {
    for (size_t i = 0; i < _interpreter->input_names.size(); ++i) {
        if (name == _interpreter->input_names[i]) {
            return i;
        }
    }
    throw std::invalid_argument("Input tensor not found");
}

size_t LiteRTRunner::find_output_idx_by_name_interpreter(std::string_view name) const {
    for (size_t i = 0; i < _interpreter->output_names.size(); ++i) {
        if (name == _interpreter->output_names[i]) {
            return i;
        }
    }
    throw std::invalid_argument("Output tensor not found");
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_LITERT
