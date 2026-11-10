/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_THREAD_POOL_H
#define COMPUTE_FLOW_CUSTOM_OPS_THREAD_POOL_H

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace compute_flow::custom_ops {

/**
 * @brief Reuse C++ threads to run one callback per thread in a synchronous call.
 *
 * The pool owns capacity() - 1 background threads. Thread zero runs on the calling thread.
 * Calls to run() on the same pool are serialized, so callers may share one pool across
 * operations. The pool has no knowledge of the work performed by the callback.
 */
class ThreadPool {
public:
    /**
     * @param max_threads Maximum number of threads, including the caller; must be at least one.
     * @throws std::runtime_error if max_threads is zero.
     * @throws std::system_error if a background thread cannot be created.
     */
    explicit ThreadPool(size_t max_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /** Maximum number of threads available to a call. */
    size_t capacity() const { return _max_threads; }

    /**
     * @brief Invoke callback once per thread and wait for all threads to finish.
     *
     * Thread IDs are distinct in [0, num_threads). Each callback receives its thread ID and
     * the same num_threads value. If callbacks throw, all threads finish before the first
     * exception in thread ID order is rethrown. A callback must not call run() on this pool.
     *
     * @param num_threads Threads for this call, including the caller; must be in [1, capacity()].
     * @param callback Work to run on each thread; must be nonempty and safe to call concurrently.
     * @throws std::runtime_error if num_threads is invalid or callback is empty.
     */
    void run(size_t num_threads, const std::function<void(size_t, size_t)>& callback);

private:
    void thread_loop(size_t thread_id);

    const size_t _max_threads;
    std::vector<std::thread> _threads;
    std::vector<std::exception_ptr> _exceptions;
    std::mutex _run_mutex;
    std::mutex _mutex;
    std::condition_variable _work_condition;
    std::condition_variable _done_condition;
    const std::function<void(size_t, size_t)>* _callback = nullptr;
    size_t _generation = 0;
    size_t _num_active_threads = 0;
    size_t _num_completed_threads = 0;
    bool _stopping = false;
};

}  // namespace compute_flow::custom_ops

#endif
