/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_V1_KERNEL_H
#define COMPUTE_FLOW_CUSTOM_OPS_V1_KERNEL_H

#include "compute_flow/custom_ops/registry_types.h"
#include "compute_flow/custom_ops/v1/types.h"

#include <string>
#include <vector>

namespace compute_flow::custom_ops::v1 {

/** One compiled implementation for a versioned custom operator. */
struct Kernel {
    static constexpr uint64_t k_interface_version = 1;

    /** Name of the custom operator in the model. */
    std::string operator_name;
    /** Schema version implemented by this kernel. */
    uint64_t schema_version;
    /** Hardware features required to run this implementation; empty for the portable reference. */
    std::vector<::compute_flow::custom_ops::CpuFeature> required_features;
    /** Query the persistent workspace requirement. */
    int64_t (*get_persistent_memory_requirement)(
        const CF_TensorFormatArray* src_formats,
        CF_MemoryRequirement* requirements);
    /** Query the scratch workspace requirement. */
    int64_t (*get_scratch_memory_requirement)(
        const CF_TensorFormatArray* src_formats,
        const CF_TensorFormatArray* dst_formats,
        CF_MemoryRequirement* requirements);
    /** Initialize persistent workspace once per constant setup buffer. Read changing input values
     *  in run, not setup. */
    int64_t (*setup)(const CF_TensorArray* src_tensors, CF_Workspace* persistent_mem);
    /** Execute this kernel. */
    int64_t (*run)(
        const CF_TensorArray* src_tensors,
        CF_TensorArray* dst_tensors,
        const CF_Workspace* persistent_mem,
        CF_Workspace* scratch_mem,
        const CF_ThreadInfo* thread_info);
    /** Calculate output dimensions from current input shapes before output allocation. */
    int64_t (*calculate_output_shapes)(
        const CF_TensorShape* input_shapes,
        size_t num_inputs,
        CF_TensorShape* output_shapes,
        size_t num_outputs) = nullptr;
    /** Constant input indices read by setup or used for persistent memory size. Their shapes and
     *  contents must stay immutable. Empty means all input formats determine setup. */
    std::vector<size_t> setup_inputs = {};
};

}  // namespace compute_flow::custom_ops::v1

#endif
