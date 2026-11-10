/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_V1_THREAD_DISPATCH_H
#define COMPUTE_FLOW_CUSTOM_OPS_V1_THREAD_DISPATCH_H

#include "compute_flow/custom_ops/v1/kernel.h"
#include "compute_flow/custom_ops/thread_pool.h"

#include <cstddef>
#include <cstdint>

namespace compute_flow::custom_ops::v1 {

/**
 * @brief Run one selected custom operator kernel on a reusable thread pool.
 *
 * Thread zero runs on the calling thread. Other threads use the pool's background threads.
 * Every thread receives one distinct ID in [0, num_threads), the same num_threads value,
 * and the same tensor and workspace buffers. All threads finish before this function returns.
 * The kernel must partition writes to output and shared scratch storage by thread ID.
 *
 * @param thread_pool Pool shared across custom operator calls.
 * @param kernel Selected kernel with a non-null run entry point.
 * @param src_tensors Input tensors for every thread.
 * @param dst_tensors Output tensors shared by all threads.
 * @param persistent_mem Setup-prepared data shared by all threads.
 * @param scratch_mem One scratch buffer shared by all threads.
 * @param num_threads Total threads, including the caller; must be at least one.
 * @return Zero if every thread succeeds; otherwise the first nonzero status in thread ID order.
 * @throws std::runtime_error if the kernel has no run entry point or num_threads is outside
 * [1, thread_pool.capacity()].
 * Exceptions raised by a thread are rethrown after all threads have finished.
 */
int64_t dispatch_threads(
    ThreadPool& thread_pool,
    const Kernel& kernel,
    const CF_TensorArray& src_tensors,
    CF_TensorArray& dst_tensors,
    const CF_Workspace& persistent_mem,
    CF_Workspace& scratch_mem,
    size_t num_threads);

}  // namespace compute_flow::custom_ops::v1

#endif
