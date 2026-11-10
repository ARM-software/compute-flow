/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_V1_TYPES_H
#define COMPUTE_FLOW_CUSTOM_OPS_V1_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define CF_MAX_TENSOR_RANK 8

/* Shape and byte layout of one tensor. dims follow schema shape order. For
 * contiguous data, the last dimension varies fastest: its stride is the element
 * size in bytes, and each earlier stride is the next stride multiplied by the
 * next dimension. Kernels must use the supplied byte strides for other layouts.
 * Only the first rank entries are valid.
 */
typedef struct {
    int64_t dims[CF_MAX_TENSOR_RANK];    /* Extent of each dimension. */
    int64_t strides[CF_MAX_TENSOR_RANK]; /* Byte stride of each dimension. */
    uint32_t rank;                       /* Number of valid dims and strides. */
    uint64_t size_bytes;                 /* Size of the tensor buffer in bytes. */
} CF_TensorFormat;

/* Tensor dimensions without a buffer layout. Rank is fixed by the operator schema. */
typedef struct {
    int64_t dims[CF_MAX_TENSOR_RANK]; /* Extent of each dimension. */
    uint32_t rank;                    /* Number of valid dimensions. */
} CF_TensorShape;

/* Tensor data paired with its shape and byte layout. */
typedef struct {
    void* data;             /* Tensor buffer. */
    CF_TensorFormat format; /* Shape and byte layout of the buffer. */
} CF_Tensor;

/* Ordered tensors passed to setup and run. */
typedef struct {
    CF_Tensor* tensors; /* Tensors in schema order. */
    size_t count;       /* Number of tensors. */
} CF_TensorArray;

/* Ordered tensor formats used to calculate memory requirements. */
typedef struct {
    const CF_TensorFormat* formats; /* Formats in schema order. */
    size_t count;                   /* Number of formats. */
} CF_TensorFormatArray;

/* Caller-owned memory available to the operator. */
typedef struct {
    void* data;  /* Caller-owned buffer. */
    size_t size; /* Available bytes. */
} CF_Workspace;

/* Size and alignment requested from the caller. */
typedef struct {
    size_t size;      /* Required bytes. */
    size_t alignment; /* Required byte alignment. */
} CF_MemoryRequirement;

/* Identity and count of threads participating in one run. */
typedef struct {
    size_t thread_id;   /* Index of the current thread. */
    size_t num_threads; /* Number of threads in the run. */
} CF_ThreadInfo;

#endif
