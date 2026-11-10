/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reference_kernels.h"

int64_t custom_op_binary_output_shapes(
    const CF_TensorShape* inputs,
    size_t num_inputs,
    CF_TensorShape* outputs,
    size_t num_outputs) {
    if (num_inputs != 2 || num_outputs != 1 || inputs[0].rank != 1 || inputs[1].rank != 1 ||
        inputs[0].dims[0] != inputs[1].dims[0]) {
        return 1;
    }

    outputs[0].rank = 1;
    outputs[0].dims[0] = inputs[0].dims[0];
    return 0;
}

static int64_t no_workspace(CF_MemoryRequirement* requirement) {
    requirement->size = 0;
    requirement->alignment = 1;
    return 0;
}

static int64_t run_binary(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_ThreadInfo* thread,
    int subtract) {
    if (src->count != 2 || dst->count != 1 || thread->num_threads == 0 ||
        src->tensors[0].format.rank != 1 || src->tensors[1].format.rank != 1 ||
        dst->tensors[0].format.rank != 1 || src->tensors[0].format.strides[0] != sizeof(float) ||
        src->tensors[1].format.strides[0] != sizeof(float) || dst->tensors[0].format.strides[0] != sizeof(float)) {
        return 1;
    }

    const float* lhs = (const float*)src->tensors[0].data;
    const float* rhs = (const float*)src->tensors[1].data;
    float* output = (float*)dst->tensors[0].data;
    const int64_t num_elements = dst->tensors[0].format.dims[0];

    for (int64_t i = (int64_t)thread->thread_id; i < num_elements; i += (int64_t)thread->num_threads) {
        output[i] = subtract ? lhs[i] - rhs[i] : lhs[i] + rhs[i];
    }

    return 0;
}

static int64_t run_binary_i8(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_ThreadInfo* thread,
    int subtract) {
    if (src->count != 2 || dst->count != 1 || thread->num_threads == 0 ||
        src->tensors[0].format.rank != 1 || src->tensors[1].format.rank != 1 ||
        dst->tensors[0].format.rank != 1 || src->tensors[0].format.strides[0] != 1 ||
        src->tensors[1].format.strides[0] != 1 || dst->tensors[0].format.strides[0] != 1) {
        return 1;
    }

    const int8_t* lhs = (const int8_t*)src->tensors[0].data;
    const int8_t* rhs = (const int8_t*)src->tensors[1].data;
    int8_t* output = (int8_t*)dst->tensors[0].data;
    const int64_t num_elements = dst->tensors[0].format.dims[0];

    for (int64_t i = (int64_t)thread->thread_id; i < num_elements; i += (int64_t)thread->num_threads) {
        output[i] = (int8_t)(subtract ? lhs[i] - rhs[i] : lhs[i] + rhs[i]);
    }

    return 0;
}

int64_t custom_op_add_persistent(const CF_TensorFormatArray* src, CF_MemoryRequirement* requirement) {
    (void)src;
    return no_workspace(requirement);
}

int64_t custom_op_add_scratch(
    const CF_TensorFormatArray* src,
    const CF_TensorFormatArray* dst,
    CF_MemoryRequirement* requirement) {
    (void)src;
    (void)dst;
    return no_workspace(requirement);
}

int64_t custom_op_add_setup(const CF_TensorArray* src, CF_Workspace* workspace) {
    (void)src;
    (void)workspace;
    return 0;
}

int64_t custom_op_add_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    (void)persistent;
    (void)scratch;
    return run_binary(src, dst, thread, 0);
}

int64_t custom_op_add_i8_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    (void)persistent;
    (void)scratch;
    return run_binary_i8(src, dst, thread, 0);
}

int64_t custom_op_sub_persistent(const CF_TensorFormatArray* src, CF_MemoryRequirement* requirement) {
    (void)src;
    return no_workspace(requirement);
}

int64_t custom_op_sub_scratch(
    const CF_TensorFormatArray* src,
    const CF_TensorFormatArray* dst,
    CF_MemoryRequirement* requirement) {
    (void)src;
    (void)dst;
    return no_workspace(requirement);
}

int64_t custom_op_sub_setup(const CF_TensorArray* src, CF_Workspace* workspace) {
    (void)src;
    (void)workspace;
    return 0;
}

int64_t custom_op_sub_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    (void)persistent;
    (void)scratch;
    return run_binary(src, dst, thread, 1);
}

int64_t custom_op_sub_i8_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    (void)persistent;
    (void)scratch;
    return run_binary_i8(src, dst, thread, 1);
}
