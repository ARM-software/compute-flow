/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/thread_pool.h"

#include "compute_flow/check.h"

#include <algorithm>

namespace compute_flow::custom_ops {

ThreadPool::ThreadPool(size_t max_threads)
    : _max_threads(max_threads), _exceptions(max_threads) {
    COMPUTE_FLOW_CHECK(max_threads > 0);

    _threads.reserve(max_threads - 1);

    try {
        for (size_t thread_id = 1; thread_id < max_threads; ++thread_id) {
            _threads.emplace_back([this, thread_id]() { thread_loop(thread_id); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopping = true;
        }

        _work_condition.notify_all();

        for (std::thread& thread : _threads) {
            thread.join();
        }

        throw;
    }
}

ThreadPool::~ThreadPool() {
    std::lock_guard<std::mutex> run_lock(_run_mutex);

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopping = true;
    }

    _work_condition.notify_all();

    for (std::thread& thread : _threads) {
        thread.join();
    }
}

void ThreadPool::run(size_t num_threads, const std::function<void(size_t, size_t)>& callback) {
    COMPUTE_FLOW_CHECK(num_threads > 0 && num_threads <= _max_threads);
    COMPUTE_FLOW_CHECK(callback);

    std::lock_guard<std::mutex> run_lock(_run_mutex);

    if (num_threads == 1) {
        callback(0, 1);
        return;
    }

    std::fill(_exceptions.begin(), _exceptions.begin() + num_threads, std::exception_ptr{});

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _callback = &callback;
        _num_active_threads = num_threads;
        _num_completed_threads = 0;
        ++_generation;
    }

    _work_condition.notify_all();

    try {
        callback(0, num_threads);
    } catch (...) {
        _exceptions[0] = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(_mutex);
        _done_condition.wait(lock, [&]() { return _num_completed_threads == num_threads - 1; });
        _callback = nullptr;
    }

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        if (_exceptions[thread_id]) {
            std::rethrow_exception(_exceptions[thread_id]);
        }
    }
}

void ThreadPool::thread_loop(size_t thread_id) {
    size_t observed_generation = 0;

    while (true) {
        std::unique_lock<std::mutex> lock(_mutex);

        _work_condition.wait(lock, [&]() { return _stopping || _generation != observed_generation; });

        if (_stopping) {
            return;
        }

        observed_generation = _generation;
        const size_t num_threads = _num_active_threads;
        const auto* callback = _callback;

        lock.unlock();

        if (thread_id < num_threads) {
            try {
                (*callback)(thread_id, num_threads);
            } catch (...) {
                _exceptions[thread_id] = std::current_exception();
            }

            lock.lock();
            ++_num_completed_threads;
            lock.unlock();
            _done_condition.notify_one();
        }
    }
}

}  // namespace compute_flow::custom_ops
