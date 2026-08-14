//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace compute_flow {

enum class DataType : uint8_t {
    F32,
    F16,
    BF16,
    F64,
    I64,
    U64,
    I32,
    U32,
    I8,
    U8,
    String,
    Bool
};

struct TensorFormat {
    DataType             dt{};
    std::vector<int64_t> shape;
    bool has_dynamic_shape() const;
    int64_t get_nbytes() const;
    int64_t get_num_elems() const;
};

struct TensorData {
    // In theory, we should also have the strides to access the elements of the tensor properly
    // I have skipped this member for simplicity
    // The strides is needed both if user_hosted = true and user_hosted = false
    // TensorData should include TensorFormat and the pointer
    TensorFormat          format;
    void*                 data{};
    bool                  user_hosted {false};
};

struct TensorDictExternalState {
    virtual ~TensorDictExternalState() = default;
};

class TensorDict {
public:
    TensorData& declare_tensor(std::string_view key);
    TensorData& operator[](std::string_view key);
    const TensorData& operator[](std::string_view key) const;
    TensorData& at(std::string_view key);
    const TensorData& at(std::string_view key) const;
    size_t size() const noexcept;
    std::vector<std::string> keys() const;

    void set_external_state(std::shared_ptr<TensorDictExternalState> state) {
        _external_state = std::move(state);
    }

    std::shared_ptr<TensorDictExternalState> get_external_state() const {
        return _external_state;
    }

private:
    std::vector<TensorData>                 _items;
    std::unordered_map<std::string, size_t> _idx;
    std::shared_ptr<TensorDictExternalState> _external_state;
};

}  // namespace compute_flow
