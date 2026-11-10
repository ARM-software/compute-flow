/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_TEST_CUSTOM_OP_REFERENCE_KERNELS_H
#define COMPUTE_FLOW_TEST_CUSTOM_OP_REFERENCE_KERNELS_H

#include "compute_flow/custom_ops/v1/types.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t custom_op_binary_output_shapes(const CF_TensorShape*, size_t, CF_TensorShape*, size_t);

int64_t custom_op_add_persistent(const CF_TensorFormatArray*, CF_MemoryRequirement*);
int64_t custom_op_add_scratch(const CF_TensorFormatArray*, const CF_TensorFormatArray*, CF_MemoryRequirement*);
int64_t custom_op_add_setup(const CF_TensorArray*, CF_Workspace*);
int64_t custom_op_add_run(const CF_TensorArray*, CF_TensorArray*, const CF_Workspace*, CF_Workspace*, const CF_ThreadInfo*);
int64_t custom_op_add_i8_run(const CF_TensorArray*, CF_TensorArray*, const CF_Workspace*, CF_Workspace*, const CF_ThreadInfo*);

int64_t custom_op_sub_persistent(const CF_TensorFormatArray*, CF_MemoryRequirement*);
int64_t custom_op_sub_scratch(const CF_TensorFormatArray*, const CF_TensorFormatArray*, CF_MemoryRequirement*);
int64_t custom_op_sub_setup(const CF_TensorArray*, CF_Workspace*);
int64_t custom_op_sub_run(const CF_TensorArray*, CF_TensorArray*, const CF_Workspace*, CF_Workspace*, const CF_ThreadInfo*);
int64_t custom_op_sub_i8_run(const CF_TensorArray*, CF_TensorArray*, const CF_Workspace*, CF_Workspace*, const CF_ThreadInfo*);

#ifdef __cplusplus
}
#endif

#endif
