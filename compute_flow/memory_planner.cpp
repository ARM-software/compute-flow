//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/memory_planner.h"

namespace compute_flow {

void MemoryPlanner::allocate(const std::string& name, const TensorFormat& fmt) {
    TensorData td;
    td.format.dt    = fmt.dt;
    td.format.shape = fmt.shape;
    td.user_hosted  = true;

    auto buf = allocate_aligned_bytes(td.format.get_nbytes());
    td.data = buf.get();

    // store
    buffers_[name] = std::move(buf);
    tensors_[name] = std::move(td);
}

// Get metadata + raw pointer
TensorData& MemoryPlanner::get_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

template <typename T>
T* MemoryPlanner::get_data_as(const std::string& name) {
    auto& t = get_tensor(name);
    if (!dtype_matches<T>(t.format.dt)) {
        throw std::runtime_error("dtype/view mismatch for tensor '" + name + "'");
    }
    return reinterpret_cast<T*>(t.data);
}

template double* MemoryPlanner::get_data_as(const std::string& name);
template float* MemoryPlanner::get_data_as(const std::string& name);
template int64_t* MemoryPlanner::get_data_as(const std::string& name);
template uint64_t* MemoryPlanner::get_data_as(const std::string& name);
template int32_t* MemoryPlanner::get_data_as(const std::string& name);
template uint8_t* MemoryPlanner::get_data_as(const std::string& name);
template int8_t* MemoryPlanner::get_data_as(const std::string& name);
template char* MemoryPlanner::get_data_as(const std::string& name);

size_t MemoryPlanner::nbytes(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) throw std::runtime_error("Tensor not found: " + name);
    // no stored size, recompute from metadata
    const auto& t = tensors_.at(name);
    return t.format.get_nbytes();
}

template <> bool MemoryPlanner::dtype_matches<double>(DataType dt) {
    return dt == DataType::F64;
}

template <>
bool MemoryPlanner::dtype_matches<float>(DataType dt) {
    return dt == DataType::F32;
}

template <> bool MemoryPlanner::dtype_matches<int64_t>(DataType dt) {
    return dt == DataType::I64;
}

template <> bool MemoryPlanner::dtype_matches<uint64_t>(DataType dt) {
    return dt == DataType::U64;
}

template <> bool MemoryPlanner::dtype_matches<int32_t>(DataType dt) {
    return dt == DataType::I32;
}

template <> bool MemoryPlanner::dtype_matches<uint8_t>(DataType dt) {
    return dt == DataType::U8;
}

template <> bool MemoryPlanner::dtype_matches<int8_t>(DataType dt) {
    return dt == DataType::I8;
}

template <> bool MemoryPlanner::dtype_matches<char>(DataType dt) {
    return dt == DataType::String;
}

size_t MemoryPlanner::get_num_elems(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto s : shape) {
        n *= static_cast<size_t>(s);
    }
    return n;
}

#ifdef _MSC_VER
void MemoryPlanner::AlignedDeleter::operator()(void* p) const noexcept {
    if (p) {
        _aligned_free(p);
    }
}
#else
void MemoryPlanner::AlignedDeleter::operator()(void* p) const noexcept {
    std::free(p);
}
#endif

std::unique_ptr<void, MemoryPlanner::AlignedDeleter> MemoryPlanner::allocate_aligned_bytes(size_t nbytes) {
    constexpr std::size_t kAlign = alignof(std::max_align_t);
    void* p = nullptr;
#ifdef _MSC_VER
    p = _aligned_malloc(nbytes, kAlign);
    if (!p) {
        throw std::bad_alloc();
    }
#elif defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
    // C11 aligned_alloc requires size to be multiple of alignment
    size_t rounded = (nbytes + (kAlign - 1)) & ~(kAlign - 1);
    p = std::aligned_alloc(kAlign, rounded);
    if (!p) {
        throw std::bad_alloc();
    }
#else
    // POSIX fallback
    if (posix_memalign(&p, kAlign, nbytes) != 0) {
        throw std::bad_alloc();
    }
#endif
    return std::unique_ptr<void, AlignedDeleter>(p);
}

void MemoryPlanner::release(const std::string& name) {
    // Remove metadata first (so we don't keep a dangling TensorData
    // that points to memory we're about to free)
    auto t_it = tensors_.find(name);
    auto b_it = buffers_.find(name);

    if (t_it == tensors_.end() && b_it == buffers_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    if (t_it != tensors_.end()) {
        tensors_.erase(t_it);
    }
    if (b_it != buffers_.end()) {
        buffers_.erase(b_it);  // unique_ptr destructor frees the buffer
    }
}

}  // namespace compute_flow
