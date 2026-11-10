/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/v1/thread_dispatch.h"

#include "compute_flow/check.h"

#include <vector>

namespace compute_flow::custom_ops::v1 {

int64_t dispatch_threads(
    ThreadPool& thread_pool,
    const Kernel& kernel,
    const CF_TensorArray& src_tensors,
    CF_TensorArray& dst_tensors,
    const CF_Workspace& persistent_mem,
    CF_Workspace& scratch_mem,
    size_t num_threads) {
    COMPUTE_FLOW_CHECK(kernel.run != nullptr);
    COMPUTE_FLOW_CHECK(num_threads > 0 && num_threads <= thread_pool.capacity());

    std::vector<int64_t> statuses(num_threads, 0);

    thread_pool.run(num_threads, [&](size_t thread_id, size_t thread_count) {
        const CF_ThreadInfo thread_info{thread_id, thread_count};

        statuses[thread_id] =
            kernel.run(&src_tensors, &dst_tensors, &persistent_mem, &scratch_mem, &thread_info);
    });

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        if (statuses[thread_id] != 0) {
            return statuses[thread_id];
        }
    }

    return 0;
}

}  // namespace compute_flow::custom_ops::v1
