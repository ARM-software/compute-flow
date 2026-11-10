/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_MEMORY_MANAGER_H
#define COMPUTE_FLOW_CUSTOM_OPS_MEMORY_MANAGER_H

#include <cstddef>
#include <memory>

namespace compute_flow::custom_ops {

/**
 * @brief Own one aligned workspace buffer.
 *
 * Keep a separate manager for each custom operator node's persistent workspace. Share one scratch
 * manager across custom operator calls that do not overlap, allocating it before every run.
 * Concurrent calls need separate scratch managers. The manager does not call setup or run. Its
 * buffer remains valid until the next allocation, reset, or destruction.
 */
class MemoryManager {
public:
    /**
     * @brief Make a buffer available for a memory requirement supplied by the caller.
     *
     * @param num_bytes Required bytes. Zero makes the buffer inactive but retains its capacity.
     * @param alignment Required byte alignment. For nonzero sizes, this must be a nonzero power of
     * two.
     *
     * Repeated calls reuse the allocation when it meets both requirements. Otherwise the manager
     * replaces it with an allocation large enough for the current request and its previous
     * capacity. Requests are not added together. A failed allocation leaves the previous buffer
     * available.
     */
    void allocate(size_t num_bytes, size_t alignment);

    /** @return Current buffer pointer, or nullptr after a zero-byte request or reset. */
    void* data() const noexcept;

    /** @return Available bytes, or zero after a zero-byte request or reset. */
    size_t size() const noexcept;

    /** Release the workspace. */
    void reset() noexcept;

private:
    struct AlignedDeleter {
        size_t alignment = alignof(std::max_align_t);
        void operator()(void* data) const noexcept;
    };

    struct Buffer {
        std::unique_ptr<void, AlignedDeleter> storage{nullptr, AlignedDeleter{}};
        size_t capacity = 0;
        size_t alignment = 0;
        bool active = false;

        void allocate(size_t num_bytes, size_t alignment);
        void* data() const noexcept;
        size_t size() const noexcept;
        void reset() noexcept;
    };

    Buffer _buffer;
};

}  // namespace compute_flow::custom_ops

#endif
