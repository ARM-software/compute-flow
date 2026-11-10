/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/v1/thread_dispatch.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

constexpr size_t k_num_threads = 4;

struct Fixture {
    std::array<int32_t, k_num_threads> calls{};
    std::array<std::thread::id, k_num_threads> thread_ids{};
    std::array<int64_t, k_num_threads> statuses{};
    std::array<int32_t, k_num_threads> output{};
    std::mutex mutex;
    std::condition_variable condition;
    size_t entered = 0;
    size_t expected_threads = 1;
    size_t throwing_thread = k_num_threads;
    bool wait_for_all = false;
    uint8_t persistent_byte = 0;
    uint8_t scratch_byte = 0;
    CF_Tensor src_tensor{};
    CF_Tensor dst_tensor{};
    CF_TensorArray src_tensors{&src_tensor, 1};
    CF_TensorArray dst_tensors{&dst_tensor, 1};
    CF_Workspace persistent_mem{&persistent_byte, 1};
    CF_Workspace scratch_mem{&scratch_byte, 1};

    Fixture() {
        src_tensor.data = this;
        dst_tensor.data = output.data();
    }
};

int64_t record_run(
    const CF_TensorArray* src_tensors,
    CF_TensorArray* dst_tensors,
    const CF_Workspace* persistent_mem,
    CF_Workspace* scratch_mem,
    const CF_ThreadInfo* thread_info) {
    if (src_tensors == nullptr || src_tensors->count != 1 || thread_info == nullptr ||
        thread_info->thread_id >= k_num_threads) {
        return -100;
    }

    auto* fixture = static_cast<Fixture*>(src_tensors->tensors[0].data);
    const size_t thread_id = thread_info->thread_id;

    if (dst_tensors != &fixture->dst_tensors || persistent_mem != &fixture->persistent_mem ||
        scratch_mem != &fixture->scratch_mem || persistent_mem->data != &fixture->persistent_byte ||
        scratch_mem->data != &fixture->scratch_byte ||
        thread_info->num_threads != fixture->expected_threads) {
        return -101;
    }

    fixture->calls[thread_id] += 1;
    fixture->thread_ids[thread_id] = std::this_thread::get_id();
    fixture->output[thread_id] = static_cast<int32_t>(thread_id);

    if (fixture->wait_for_all) {
        std::unique_lock<std::mutex> lock(fixture->mutex);

        ++fixture->entered;
        fixture->condition.notify_all();

        if (!fixture->condition.wait_for(
                lock,
                std::chrono::seconds(5),
                [&]() { return fixture->entered == fixture->expected_threads; })) {
            return -102;
        }
    }

    if (thread_id == fixture->throwing_thread) {
        throw std::runtime_error("thread failed");
    }

    return fixture->statuses[thread_id];
}

compute_flow::custom_ops::v1::Kernel make_kernel() {
    using compute_flow::custom_ops::v1::Kernel;

    return {"test_op", 1, {}, nullptr, nullptr, nullptr, record_run};
}

}  // namespace

int main() {
    using compute_flow::custom_ops::ThreadPool;
    using compute_flow::custom_ops::v1::dispatch_threads;

    const auto kernel = make_kernel();
    const std::thread::id caller_id = std::this_thread::get_id();
    ThreadPool thread_pool(k_num_threads);
    Fixture serial;

    if (dispatch_threads(
            thread_pool,
            kernel,
            serial.src_tensors,
            serial.dst_tensors,
            serial.persistent_mem,
            serial.scratch_mem,
            1) != 0 ||
        serial.calls[0] != 1 || serial.thread_ids[0] != caller_id || serial.output[0] != 0) {
        return 1;
    }

    Fixture parallel;

    parallel.expected_threads = k_num_threads;
    parallel.wait_for_all = true;

    if (dispatch_threads(
            thread_pool,
            kernel,
            parallel.src_tensors,
            parallel.dst_tensors,
            parallel.persistent_mem,
            parallel.scratch_mem,
            k_num_threads) != 0) {
        return 2;
    }

    for (size_t thread_id = 0; thread_id < k_num_threads; ++thread_id) {
        if (parallel.calls[thread_id] != 1 ||
            parallel.output[thread_id] != static_cast<int32_t>(thread_id)) {
            return 3;
        }

        for (size_t other_id = 0; other_id < thread_id; ++other_id) {
            if (parallel.thread_ids[thread_id] == parallel.thread_ids[other_id]) {
                return 4;
            }
        }
    }

    if (parallel.thread_ids[0] != caller_id) {
        return 5;
    }

    Fixture repeat;

    repeat.expected_threads = k_num_threads;

    if (dispatch_threads(
            thread_pool,
            kernel,
            repeat.src_tensors,
            repeat.dst_tensors,
            repeat.persistent_mem,
            repeat.scratch_mem,
            k_num_threads) != 0) {
        return 12;
    }

    for (size_t thread_id = 1; thread_id < k_num_threads; ++thread_id) {
        if (repeat.thread_ids[thread_id] != parallel.thread_ids[thread_id]) {
            return 13;
        }
    }

    std::array<int32_t, k_num_threads> generic_calls{};

    thread_pool.run(2, [&](size_t thread_id, size_t num_threads) {
        if (num_threads == 2) {
            generic_calls[thread_id] += 1;
        }
    });

    if (generic_calls[0] != 1 || generic_calls[1] != 1 || generic_calls[2] != 0 ||
        generic_calls[3] != 0) {
        return 14;
    }

    std::array<std::array<int32_t, 2>, 2> concurrent_calls{};
    std::thread first_caller([&]() {
        thread_pool.run(2, [&](size_t thread_id, size_t num_threads) {
            if (num_threads == 2) {
                concurrent_calls[0][thread_id] += 1;
            }
        });
    });
    std::thread second_caller([&]() {
        thread_pool.run(2, [&](size_t thread_id, size_t num_threads) {
            if (num_threads == 2) {
                concurrent_calls[1][thread_id] += 1;
            }
        });
    });

    first_caller.join();
    second_caller.join();

    for (const auto& calls : concurrent_calls) {
        if (calls[0] != 1 || calls[1] != 1) {
            return 15;
        }
    }

    Fixture failures;

    failures.expected_threads = k_num_threads;
    failures.statuses[2] = 7;
    failures.statuses[3] = 11;

    if (dispatch_threads(
            thread_pool,
            kernel,
            failures.src_tensors,
            failures.dst_tensors,
            failures.persistent_mem,
            failures.scratch_mem,
            k_num_threads) != 7) {
        return 6;
    }

    for (int32_t count : failures.calls) {
        if (count != 1) {
            return 7;
        }
    }

    Fixture exception;

    exception.expected_threads = k_num_threads;
    exception.statuses[1] = 7;
    exception.throwing_thread = 2;
    bool caught_thread_exception = false;

    try {
        dispatch_threads(
            thread_pool,
            kernel,
            exception.src_tensors,
            exception.dst_tensors,
            exception.persistent_mem,
            exception.scratch_mem,
            k_num_threads);
    } catch (const std::runtime_error&) {
        caught_thread_exception = true;
    }

    if (!caught_thread_exception) {
        return 8;
    }

    for (int32_t count : exception.calls) {
        if (count != 1) {
            return 9;
        }
    }

    bool rejected_zero_threads = false;

    try {
        dispatch_threads(
            thread_pool,
            kernel,
            serial.src_tensors,
            serial.dst_tensors,
            serial.persistent_mem,
            serial.scratch_mem,
            0);
    } catch (const std::runtime_error&) {
        rejected_zero_threads = true;
    }

    if (!rejected_zero_threads) {
        return 10;
    }

    auto missing_run = kernel;

    missing_run.run = nullptr;
    bool rejected_missing_run = false;

    try {
        dispatch_threads(
            thread_pool,
            missing_run,
            serial.src_tensors,
            serial.dst_tensors,
            serial.persistent_mem,
            serial.scratch_mem,
            1);
    } catch (const std::runtime_error&) {
        rejected_missing_run = true;
    }

    return rejected_missing_run ? 0 : 11;
}
