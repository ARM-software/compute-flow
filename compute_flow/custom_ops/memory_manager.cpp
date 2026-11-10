/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/memory_manager.h"

#include "compute_flow/check.h"

#include <algorithm>
#include <new>
#include <utility>

namespace compute_flow::custom_ops {

void MemoryManager::AlignedDeleter::operator()(void* data) const noexcept {
    ::operator delete(data, std::align_val_t(alignment));
}

void MemoryManager::Buffer::allocate(size_t num_bytes, size_t required_alignment) {
    if (num_bytes == 0) {
        active = false;
        return;
    }

    COMPUTE_FLOW_CHECK(
        required_alignment != 0 && (required_alignment & (required_alignment - 1)) == 0);

    required_alignment = std::max(required_alignment, alignof(std::max_align_t));

    if (storage && num_bytes <= capacity && required_alignment <= alignment) {
        active = true;
        return;
    }

    const size_t new_capacity = std::max(num_bytes, capacity);
    std::unique_ptr<void, AlignedDeleter> new_storage(
        ::operator new(new_capacity, std::align_val_t(required_alignment)),
        AlignedDeleter{required_alignment});

    storage = std::move(new_storage);
    capacity = new_capacity;
    alignment = required_alignment;
    active = true;
}

void* MemoryManager::Buffer::data() const noexcept {
    return active ? storage.get() : nullptr;
}

size_t MemoryManager::Buffer::size() const noexcept {
    return data() == nullptr ? 0 : capacity;
}

void MemoryManager::Buffer::reset() noexcept {
    storage.reset();
    capacity = 0;
    alignment = 0;
    active = false;
}

void MemoryManager::allocate(size_t num_bytes, size_t alignment) {
    _buffer.allocate(num_bytes, alignment);
}

void* MemoryManager::data() const noexcept {
    return _buffer.data();
}

size_t MemoryManager::size() const noexcept {
    return _buffer.size();
}

void MemoryManager::reset() noexcept {
    _buffer.reset();
}

}  // namespace compute_flow::custom_ops
