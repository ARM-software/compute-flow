//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "compute_flow/tensor_dict.h"

namespace compute_flow {

class MemoryPlanner {
public:
    void allocate(const std::string& name, const TensorFormat& fmt);
    TensorData& get_tensor(const std::string& name);
    template <typename T> T* get_data_as(const std::string& name);
    size_t nbytes(const std::string& name) const;
    void release(const std::string& name);

private:
    template <typename T> static bool dtype_matches(DataType);

    static size_t get_num_elems(const std::vector<int64_t>& shape);

    // cross-platform aligned alloc wrapped in unique_ptr<void, Deleter>
    struct AlignedDeleter {
        void operator()(void* p) const noexcept;
    };

    std::unique_ptr<void, AlignedDeleter> allocate_aligned_bytes(size_t nbytes);

    // Keep buffers alive; unique_ptr owns the aligned memory
    std::unordered_map<std::string, std::unique_ptr<void, AlignedDeleter>> buffers_;
    // Lightweight views/metadata
    std::unordered_map<std::string, TensorData> tensors_;
};

template <> bool MemoryPlanner::dtype_matches<float>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<int64_t>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<uint64_t>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<double>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<int8_t>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<uint8_t>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<int32_t>(DataType dt);
template <> bool MemoryPlanner::dtype_matches<char>(DataType dt);

}  // namespace compute_flow
