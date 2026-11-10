/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/memory_manager.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

bool is_aligned(const void* data, size_t alignment) {
    return data != nullptr && reinterpret_cast<uintptr_t>(data) % alignment == 0;
}

}  // namespace

int main() {
    using compute_flow::custom_ops::MemoryManager;

    MemoryManager first_persistent;
    MemoryManager second_persistent;
    MemoryManager shared_scratch;

    if (first_persistent.data() != nullptr || first_persistent.size() != 0 ||
        shared_scratch.data() != nullptr || shared_scratch.size() != 0) {
        return 1;
    }

    first_persistent.allocate(37, 64);
    const void* first_data = first_persistent.data();

    if (first_persistent.size() < 37 || !is_aligned(first_data, 64)) {
        return 2;
    }

    second_persistent.allocate(19, 32);
    const void* second_data = second_persistent.data();

    if (second_persistent.size() < 19 || !is_aligned(second_data, 32) ||
        second_data == first_data || first_persistent.data() != first_data) {
        return 3;
    }

    shared_scratch.allocate(13, 128);
    const void* first_scratch = shared_scratch.data();

    if (shared_scratch.size() < 13 || !is_aligned(first_scratch, 128)) {
        return 4;
    }

    shared_scratch.allocate(9, 64);

    if (shared_scratch.data() != first_scratch || shared_scratch.size() < 13) {
        return 5;
    }

    shared_scratch.allocate(257, 256);
    const void* grown_scratch = shared_scratch.data();
    const size_t grown_size = shared_scratch.size();

    if (grown_size < 257 || !is_aligned(grown_scratch, 256) || grown_scratch == first_scratch) {
        return 6;
    }

    shared_scratch.allocate(20, 64);

    if (shared_scratch.data() != grown_scratch || shared_scratch.size() != grown_size) {
        return 7;
    }

    shared_scratch.allocate(20, 512);
    const void* realigned_scratch = shared_scratch.data();

    if (shared_scratch.size() != grown_size || !is_aligned(realigned_scratch, 512)) {
        return 8;
    }

    bool rejected_invalid_alignment = false;

    try {
        shared_scratch.allocate(20, 3);
    } catch (const std::runtime_error&) {
        rejected_invalid_alignment = true;
    }

    if (!rejected_invalid_alignment || shared_scratch.data() != realigned_scratch) {
        return 9;
    }

    shared_scratch.allocate(0, 0);

    if (shared_scratch.data() != nullptr || shared_scratch.size() != 0) {
        return 10;
    }

    shared_scratch.allocate(20, 64);

    if (shared_scratch.data() != realigned_scratch || shared_scratch.size() != grown_size) {
        return 11;
    }

    MemoryManager concurrent_scratch;

    concurrent_scratch.allocate(20, 64);

    if (concurrent_scratch.data() == shared_scratch.data() ||
        !is_aligned(concurrent_scratch.data(), 64)) {
        return 12;
    }

    MemoryManager moved = std::move(first_persistent);

    if (moved.data() != first_data || first_persistent.data() != nullptr ||
        first_persistent.size() != 0 || second_persistent.data() != second_data) {
        return 13;
    }

    moved.reset();
    shared_scratch.reset();

    if (moved.data() != nullptr || moved.size() != 0 || shared_scratch.data() != nullptr ||
        shared_scratch.size() != 0 || concurrent_scratch.data() == nullptr) {
        return 14;
    }

    return 0;
}
