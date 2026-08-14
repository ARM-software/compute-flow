//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/model_data.h"

#include <stdexcept>

namespace compute_flow {

ModelData& ModelData::set_input(std::string name, int64_t id) {
    inputs[std::move(name)] = id;       // overwrite if exists
    return *this;
}

ModelData& ModelData::set_output(std::string name, int64_t id) {
    outputs[std::move(name)] = id;      // overwrite if exists
    return *this;
}

ModelData& ModelData::set_state(std::string name, int64_t id_in, int64_t id_out) {
    states[std::move(name)] = std::make_pair(id_in, id_out);
    return *this;
}

bool ModelData::has_input(std::string_view name) const noexcept {
    return inputs.find(std::string(name)) != inputs.end();
}

bool ModelData::has_output(std::string_view name) const noexcept {
    return outputs.find(std::string(name)) != outputs.end();
}

bool ModelData::has_state(std::string_view name) const noexcept {
    return states.find(std::string(name)) != states.end();
}

int64_t ModelData::input(std::string_view name) const {
    auto it = inputs.find(std::string(name));
    if (it == inputs.end()) {
        throw std::out_of_range("Unknown input: " + std::string(name));
    }
    return it->second;
}

int64_t ModelData::output(std::string_view name) const {
    auto it = outputs.find(std::string(name));
    if (it == outputs.end()) {
        throw std::out_of_range("Unknown output: " + std::string(name));
    }
    return it->second;
}

std::pair<int64_t, int64_t> ModelData::state(std::string_view name) const {
    auto it = states.find(std::string(name));
    if (it == states.end()) throw std::out_of_range("Unknown state: " + std::string(name));
    return it->second;
}

}  // namespace compute_flow
