//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/tensor_dict.h"

#include <stdexcept>

namespace compute_flow {

namespace {
int64_t get_dt_size(DataType t) {
    switch (t) {
        case DataType::BF16: return 2;
        case DataType::F32: return 4;
        case DataType::F16: return 2;
        case DataType::F64: return 8;
        case DataType::I32: return 4;
        case DataType::I64: return 8;
        case DataType::I8: return 1;
        case DataType::String: return 1;
        case DataType::U64: return 8;
        case DataType::U32: return 4;
        case DataType::U8: return 1;
        case DataType::Bool: return 1;
        default:
            throw std::invalid_argument("Unsupported DataType: " + std::to_string(static_cast<int>(t)));
    }
}
}

bool TensorFormat::has_dynamic_shape() const {
    for (int64_t dim : shape) {
        if (dim == -1) {
            return true;
        }
    }
    return false;
}

int64_t TensorFormat::get_nbytes() const {
    if (has_dynamic_shape()) {
        return -1;
    }

    return get_num_elems() * get_dt_size(dt);
}

int64_t TensorFormat::get_num_elems() const {
    if (has_dynamic_shape()) {
        return -1;
    }

    int64_t result = 1;
    for (int64_t v : shape) {
        result *= v;
    }
    return result;
}

TensorData& TensorDict::declare_tensor(std::string_view key) {
    std::string k(key);
    if (_idx.count(k)) {
        return at(key);
    } else {
        _idx.emplace(k, _items.size());
        _items.emplace_back();
        return _items.back();
    }
}

TensorData& TensorDict::operator[](std::string_view key) {
    return at(key);
}

const TensorData& TensorDict::operator[](std::string_view key) const {
    return at(key);
}

TensorData& TensorDict::at(std::string_view key) {
    auto it = _idx.find(std::string(key));
    if (it == _idx.end()) throw std::out_of_range("Unknown key: " + std::string(key));
    return _items[it->second];
}

const TensorData& TensorDict::at(std::string_view key) const {
    auto it = _idx.find(std::string(key));
    if (it == _idx.end()) throw std::out_of_range("Unknown key: " + std::string(key));
    return _items[it->second];
}

size_t TensorDict::size() const noexcept {
    return _items.size();
}

std::vector<std::string> TensorDict::keys() const {
    std::vector<std::string> result;
    result.reserve(_idx.size());
    for (const auto& [key, _] : _idx)
        result.push_back(key);
    return result;
}

}  // namespace compute_flow
