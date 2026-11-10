//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "runners/litert/custom_ops/v1/adapter.h"
#include "runners/litert/common/buffer_lock.h"
#include "runners/litert/common/utils.h"

#include "compute_flow/check.h"
#include "compute_flow/custom_ops/memory_manager.h"
#include "compute_flow/custom_ops/profile_label.h"
#include "compute_flow/custom_ops/thread_pool.h"
#include "compute_flow/custom_ops/v1/kernel.h"
#include "compute_flow/custom_ops/v1/thread_dispatch.h"

#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_tensor_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace compute_flow::custom_ops::v1 {
namespace {

/** Convert a LiteRT buffer layout to the v1 kernel's byte-strided tensor format. */
CF_TensorFormat tensor_format(LiteRtTensorBuffer buffer) {
    LiteRtRankedTensorType type{};
    litert::check_status(LiteRtGetTensorBufferTensorType(buffer, &type));
    const int64_t bytes_per_element = litert::element_size(type.element_type);
    COMPUTE_FLOW_CHECK(bytes_per_element != 0);
    COMPUTE_FLOW_CHECK(type.layout.rank <= CF_MAX_TENSOR_RANK);

    size_t size_bytes = 0;
    litert::check_status(LiteRtGetTensorBufferPackedSize(buffer, &size_bytes));

    CF_TensorFormat format{};
    format.rank = type.layout.rank;
    format.size_bytes = size_bytes;
    int64_t stride = bytes_per_element;
    for (size_t i = format.rank; i > 0; --i) {
        const int32_t dimension = type.layout.dimensions[i - 1];
        COMPUTE_FLOW_CHECK(dimension >= 0);
        format.dims[i - 1] = dimension;
        format.strides[i - 1] = stride;
        COMPUTE_FLOW_CHECK(dimension == 0 || stride <= std::numeric_limits<int64_t>::max() / dimension);
        stride *= dimension;
    }

    COMPUTE_FLOW_CHECK(static_cast<uint64_t>(stride) <= size_bytes);
    return format;
}

bool same_tensor_format(const CF_TensorFormat& lhs, const CF_TensorFormat& rhs) {
    if (lhs.rank != rhs.rank || lhs.size_bytes != rhs.size_bytes) {
        return false;
    }

    for (size_t index = 0; index < lhs.rank; ++index) {
        if (lhs.dims[index] != rhs.dims[index] || lhs.strides[index] != rhs.strides[index]) {
            return false;
        }
    }

    return true;
}

bool copy_input_shape(const LiteRtLayout& layout, CF_TensorShape& shape) {
    if (layout.rank == 0 || layout.rank > CF_MAX_TENSOR_RANK) {
        return false;
    }

    shape.rank = layout.rank;

    for (size_t i = 0; i < layout.rank; ++i) {
        if (layout.dimensions[i] < 0) {
            return false;
        }

        shape.dims[i] = layout.dimensions[i];
    }

    return true;
}

bool copy_output_layout(const CF_TensorShape& shape, uint32_t expected_rank, LiteRtLayout& layout) {
    if (shape.rank != expected_rank) {
        return false;
    }

    LiteRtLayout converted{};
    converted.rank = shape.rank;

    for (size_t i = 0; i < shape.rank; ++i) {
        if (shape.dims[i] < 0 || shape.dims[i] > std::numeric_limits<int32_t>::max()) {
            return false;
        }

        converted.dimensions[i] = static_cast<int32_t>(shape.dims[i]);
    }

    layout = converted;
    return true;
}

}  // namespace

/** One v1 kernel registration kept alive while LiteRT may invoke its callbacks. */
struct LiteRTAdapter::KernelState {
    struct PreparedNode {
        std::vector<CF_TensorFormat> formats;
        std::vector<const void*> setup_buffers;
        MemoryManager persistent;
    };

    Kernel kernel;
    // Output ranks belong to the operator schema; dimensions are calculated for each node.
    std::vector<uint32_t> output_ranks;
    ThreadPool* thread_pool = nullptr;  // Borrowed from the owning LiteRTAdapter.
    // Latch callback errors so consume_failure() can identify a custom-op failure after LiteRT returns.
    // LiteRT may invoke callbacks on different threads, so the latch is atomic.
    std::atomic<bool> failed{false};
    std::mutex prepared_mutex;
    std::unordered_map<size_t, std::vector<std::shared_ptr<PreparedNode>>> prepared_nodes;

    size_t setup_key(const CF_TensorArray& src, const std::vector<CF_TensorFormat>& formats) const {
        size_t key = kernel.setup_inputs.size();
        const auto combine = [&key](size_t value) { key ^= value + 0x9e3779b9U + (key << 6) + (key >> 2); };

        if (kernel.setup_inputs.empty()) {
            combine(formats.size());
            for (const auto& format : formats) {
                combine(format.rank);
                combine(format.size_bytes);
                for (size_t index = 0; index < format.rank; ++index) {
                    combine(std::hash<int64_t>{}(format.dims[index]));
                    combine(std::hash<int64_t>{}(format.strides[index]));
                }
            }
        }

        for (size_t index : kernel.setup_inputs) {
            COMPUTE_FLOW_CHECK(index < src.count);
            const auto& tensor = src.tensors[index];
            COMPUTE_FLOW_CHECK(tensor.data != nullptr || tensor.format.size_bytes == 0);
            combine(index);
            combine(std::hash<const void*>{}(tensor.data));
        }

        return key;
    }

    bool same_setup(const PreparedNode& prepared, const CF_TensorArray& src,
                    const std::vector<CF_TensorFormat>& formats) const {
        if (kernel.setup_inputs.empty()) {
            if (prepared.formats.size() != formats.size()) {
                return false;
            }

            for (size_t index = 0; index < formats.size(); ++index) {
                if (!same_tensor_format(prepared.formats[index], formats[index])) {
                    return false;
                }
            }
        }

        for (size_t index = 0; index < kernel.setup_inputs.size(); ++index) {
            if (prepared.setup_buffers[index] != src.tensors[kernel.setup_inputs[index]].data) {
                return false;
            }
        }

        return true;
    }

    // The adapter keeps this state alive for LiteRT's callbacks throughout model execution.
    static LiteRtStatus init(void*, const void*, size_t) { return kLiteRtStatusOk; }

    static LiteRtStatus get_output_layouts(
        void* user_data,
        size_t num_inputs,
        const LiteRtLayout* input_layouts,
        size_t num_outputs,
        LiteRtLayout* outputs) {
        auto& state = *static_cast<KernelState*>(user_data);
        const auto invalid = [&state] {
            state.failed.store(true);
            return kLiteRtStatusErrorInvalidArgument;
        };

        if (num_outputs != state.output_ranks.size() ||
            (num_inputs != 0 && input_layouts == nullptr) || (num_outputs != 0 && outputs == nullptr)) {
            return invalid();
        }

        // LiteRT calls this for each node; never reuse a previous node's dimensions.
        try {
            std::vector<CF_TensorShape> input_shapes(num_inputs);
            for (size_t i = 0; i < num_inputs; ++i) {
                if (!copy_input_shape(input_layouts[i], input_shapes[i])) {
                    return invalid();
                }
            }

            std::vector<CF_TensorShape> calculated_shapes(num_outputs);
            for (size_t i = 0; i < num_outputs; ++i) {
                calculated_shapes[i].rank = state.output_ranks[i];
                // The authored callback must write every output dimension.
                std::fill_n(calculated_shapes[i].dims, CF_MAX_TENSOR_RANK, -1);
            }

            if (state.kernel.calculate_output_shapes(
                    input_shapes.data(), num_inputs, calculated_shapes.data(), num_outputs) != 0) {
                return invalid();
            }

            for (size_t i = 0; i < num_outputs; ++i) {
                if (!copy_output_layout(calculated_shapes[i], state.output_ranks[i], outputs[i])) {
                    return invalid();
                }
            }

            return kLiteRtStatusOk;
        } catch (...) {
            state.failed.store(true);
            return kLiteRtStatusErrorRuntimeFailure;
        }
    }

    static LiteRtStatus run(
        void* user_data,
        size_t num_inputs,
        const LiteRtTensorBuffer* inputs,
        size_t num_outputs,
        LiteRtTensorBuffer* outputs) {
        auto& state = *static_cast<KernelState*>(user_data);

        try {
            state.invoke(num_inputs, inputs, num_outputs, outputs);
            return kLiteRtStatusOk;
        } catch (...) {
            state.failed.store(true);
            return kLiteRtStatusErrorRuntimeFailure;
        }
    }

    static LiteRtStatus destroy(void*) { return kLiteRtStatusOk; }

    void invoke(
        size_t num_inputs,
        const LiteRtTensorBuffer* inputs,
        size_t num_outputs,
        LiteRtTensorBuffer* outputs) {
        // Keep every tensor locked until the kernel finishes using its data pointers.
        std::vector<std::unique_ptr<litert::LiteRTBufferLock>> input_locks;
        std::vector<std::unique_ptr<litert::LiteRTBufferLock>> output_locks;
        std::vector<CF_Tensor> src_tensors;
        std::vector<CF_Tensor> dst_tensors;
        std::vector<CF_TensorFormat> src_formats;
        std::vector<CF_TensorFormat> dst_formats;
        input_locks.reserve(num_inputs);
        output_locks.reserve(num_outputs);
        src_tensors.reserve(num_inputs);
        dst_tensors.reserve(num_outputs);
        src_formats.reserve(num_inputs);
        dst_formats.reserve(num_outputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            input_locks.push_back(std::make_unique<litert::LiteRTBufferLock>(inputs[i], kLiteRtTensorBufferLockModeRead));
            const CF_TensorFormat format = tensor_format(inputs[i]);
            src_tensors.push_back({input_locks.back()->data(), format});
            src_formats.push_back(format);
        }

        for (size_t i = 0; i < num_outputs; ++i) {
            output_locks.push_back(std::make_unique<litert::LiteRTBufferLock>(outputs[i], kLiteRtTensorBufferLockModeWrite));
            const CF_TensorFormat format = tensor_format(outputs[i]);
            dst_tensors.push_back({output_locks.back()->data(), format});
            dst_formats.push_back(format);
        }

        CF_TensorArray src{src_tensors.data(), src_tensors.size()};
        CF_TensorArray dst{dst_tensors.data(), dst_tensors.size()};
        const CF_TensorFormatArray src_shape{src_formats.data(), src_formats.size()};
        const CF_TensorFormatArray dst_shape{dst_formats.data(), dst_formats.size()};

        // LiteRT identifies registrations, not nodes. Constant setup buffers identify prepared data.
        std::shared_ptr<PreparedNode> prepared;
        const size_t key = setup_key(src, src_formats);
        {
            std::lock_guard<std::mutex> lock(prepared_mutex);

            for (const auto& candidate : prepared_nodes[key]) {
                if (same_setup(*candidate, src, src_formats)) {
                    prepared = candidate;
                    break;
                }
            }

            if (prepared == nullptr) {
                prepared = std::make_shared<PreparedNode>();
                if (kernel.setup_inputs.empty()) {
                    prepared->formats = src_formats;
                }
                for (size_t index : kernel.setup_inputs) {
                    prepared->setup_buffers.push_back(src.tensors[index].data);
                }
                CF_MemoryRequirement setup_requirement{};
                COMPUTE_FLOW_CHECK(kernel.get_persistent_memory_requirement(&src_shape, &setup_requirement) == 0);
                prepared->persistent.allocate(setup_requirement.size, setup_requirement.alignment);
                CF_Workspace workspace{prepared->persistent.data(), prepared->persistent.size()};
                COMPUTE_FLOW_CHECK(kernel.setup(&src, &workspace) == 0);
                prepared_nodes[key].push_back(prepared);
            }
        }

        MemoryManager scratch;
        CF_MemoryRequirement requirement{};
        COMPUTE_FLOW_CHECK(kernel.get_scratch_memory_requirement(&src_shape, &dst_shape, &requirement) == 0);
        scratch.allocate(requirement.size, requirement.alignment);
        CF_Workspace persistent_workspace{prepared->persistent.data(), prepared->persistent.size()};
        CF_Workspace scratch_workspace{scratch.data(), scratch.size()};

        COMPUTE_FLOW_CHECK(dispatch_threads(
                               *thread_pool, kernel, src, dst, persistent_workspace, scratch_workspace,
                               thread_pool->capacity()) == 0);
    }
};

LiteRTAdapter::LiteRTAdapter(const ::compute_flow::custom_ops::Registry& registry)
    : _registry(registry) {}

LiteRTAdapter::~LiteRTAdapter() = default;

std::map<std::string, std::string> LiteRTAdapter::register_cpu_model_custom_ops(
    LiteRtModel model, LiteRtOptions options, size_t num_threads) {
    COMPUTE_FLOW_CHECK(_kernels.empty());

    LiteRtParamIndex num_subgraphs = 0;
    litert::check_status(LiteRtGetNumModelSubgraphs(model, &num_subgraphs));
    const std::vector<CpuFeature> available_features = detect_cpu_features();

    if (_thread_pool == nullptr) {
        _thread_pool = std::make_unique<ThreadPool>(std::max<size_t>(1, num_threads));
    }

    std::map<std::string, std::string> profile_labels;

    for (LiteRtParamIndex subgraph_index = 0; subgraph_index < num_subgraphs; ++subgraph_index) {
        LiteRtSubgraph subgraph = nullptr;
        litert::check_status(LiteRtGetModelSubgraph(model, subgraph_index, &subgraph));
        LiteRtParamIndex num_ops = 0;
        litert::check_status(LiteRtGetNumSubgraphOps(subgraph, &num_ops));

        for (LiteRtParamIndex op_index = 0; op_index < num_ops; ++op_index) {
            LiteRtOp op = nullptr;
            LiteRtOpCode code{};
            litert::check_status(LiteRtGetSubgraphOp(subgraph, op_index, &op));
            litert::check_status(LiteRtGetOpCode(op, &code));
            if (code != kLiteRtOpCodeTflCustom) {
                continue;
            }

            const char* name = nullptr;
            litert::check_status(LiteRtGetCustomCode(op, &name));
            COMPUTE_FLOW_CHECK(name != nullptr);
            const std::string operator_name(name);

            LiteRtParamIndex num_outputs = 0;
            litert::check_status(LiteRtGetNumOpOutputs(op, &num_outputs));
            std::vector<uint32_t> output_ranks;
            output_ranks.reserve(num_outputs);

            for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
                LiteRtTensor tensor = nullptr;
                LiteRtRankedTensorType type{};
                litert::check_status(LiteRtGetOpOutput(op, i, &tensor));
                litert::check_status(LiteRtGetRankedTensorType(tensor, &type));
                COMPUTE_FLOW_CHECK(litert::element_size(type.element_type) != 0);
                COMPUTE_FLOW_CHECK(type.layout.rank != 0 && type.layout.rank <= CF_MAX_TENSOR_RANK);
                output_ranks.push_back(type.layout.rank);
            }

            if (profile_labels.count(operator_name) != 0) {
                continue;
            }

            // The required reference is last, so one compatible implementation is always available.
            const KernelRegistration* selected = select_kernel<Kernel>(_registry, operator_name, available_features);

            if (selected == nullptr) {
                throw std::invalid_argument(std::string("No valid kernel implementation for LiteRT custom operator ") + name);
            }

            const Kernel* kernel = selected->as<Kernel>();
            COMPUTE_FLOW_CHECK(kernel != nullptr);
            COMPUTE_FLOW_CHECK(kernel->calculate_output_shapes != nullptr);

            auto state = std::make_unique<KernelState>();
            state->kernel = *kernel;
            state->output_ranks = output_ranks;
            state->thread_pool = _thread_pool.get();
            const LiteRtCustomOpKernel callbacks{
                KernelState::init,
                KernelState::get_output_layouts,
                KernelState::run,
                KernelState::destroy,
            };

            litert::check_status(LiteRtAddCustomOpKernelOption(
                options,
                state->kernel.operator_name.c_str(),
                static_cast<int>(state->kernel.schema_version),
                &callbacks,
                state.get()));
            // LiteRT borrows the callback state until the compiled model is destroyed.
            _kernels.push_back(std::move(state));
            profile_labels.emplace(
                operator_name,
                "c_op:" + operator_name + get_custom_op_profile_suffix(selected->kernel_name, selected->required_features));
        }
    }

    return profile_labels;
}

bool LiteRTAdapter::consume_failure() {
    bool failed = false;

    for (const auto& state : _kernels) {
        failed = state->failed.exchange(false) || failed;
    }

    return failed;
}

}  // namespace compute_flow::custom_ops::v1
