//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/manifest.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>
#include <memory>

namespace compute_flow {

namespace {

std::string manifest_value_to_string(const ConfigValue& value) {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return v;
        } else if constexpr (std::is_same_v<T, bool>) {
            return v ? "true" : "false";
        } else {
            return std::to_string(v);
        }
    }, value);
}

std::string trim(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(start, end - start));
}

} // namespace

ComputeFlowManifest& ComputeFlowManifest::set_brick_info(const BrickInfo& brick_info) {
    _brick_info = brick_info;
    return *this;
}

const BrickInfo& ComputeFlowManifest::get_brick_info() const noexcept{
    return _brick_info;
}

ModelData& ComputeFlowManifest::declare_module(std::string_view key) {
    std::string k(key);
    if (_idx.count(k)) {
        throw std::invalid_argument("Key already exists: " + k);
    }
    _idx.emplace(k, _items.size());
    _items.emplace_back();
    auto& model = _items.back();
    model.model_name = k;
    return model;
}

ModelData& ComputeFlowManifest::operator[](std::string_view key) {
    return at(key);
}

const ModelData& ComputeFlowManifest::operator[](std::string_view key) const {
    return at(key);
}

ModelData& ComputeFlowManifest::at(std::string_view key) {
    auto it = _idx.find(std::string(key));
    if (it == _idx.end()) throw std::out_of_range("Unknown key: " + std::string(key));
    return _items[it->second];
}

const ModelData& ComputeFlowManifest::at(std::string_view key) const {
    auto it = _idx.find(std::string(key));
    if (it == _idx.end()) throw std::out_of_range("Unknown key: " + std::string(key));
    return _items[it->second];
}

template <class T>
ComputeFlowManifest& ComputeFlowManifest::set_constant(std::string key, T value) {
    _constants.emplace(std::move(key), ConstantValue(std::move(value)));
    return *this;
}

template <typename T>
T ComputeFlowManifest::get_constant(std::string_view key) const {
    const ConstantValue* v = get_constant_value(key);
    if (!v) {
        throw std::runtime_error("Missing constants field: " + std::string(key));
    }
    return std::get<T>(*v);
}

const ConstantValue* ComputeFlowManifest::get_constant_ptr(std::string_view key) const {
    auto it = _constants.find(std::string(key));
    return (it == _constants.end()) ? nullptr : std::addressof(it->second);
}

template <class T>
ComputeFlowManifest& ComputeFlowManifest::set_config(std::string key, T value) {
    _configs[std::move(key)] = ConfigValue(std::move(value));
    return *this;
}

template <typename T>
T ComputeFlowManifest::get_config(std::string_view key) const {
    return std::get<T>(get_config_value(key));
}

ConfigValue ComputeFlowManifest::get_config_value(std::string_view key) const {
    auto it = _configs.find(std::string(key));
    if (it == _configs.end()) {
        throw std::runtime_error("Missing configs field: " + std::string(key));
    }

    if (std::holds_alternative<std::string>(it->second)) {
        std::unordered_map<std::string, std::string> cache;
        std::vector<std::string> stack;
        return ConfigValue(resolve_config_string(it->first, cache, stack));
    }
    return it->second;
}

const ConfigValue* ComputeFlowManifest::get_config_ptr(std::string_view key) const {
    auto it = _configs.find(std::string(key));
    return (it == _configs.end()) ? nullptr : std::addressof(it->second);
}

ComputeFlowManifest& ComputeFlowManifest::set_asset(std::string key, std::string value)
{
    _assets[std::move(key)] = std::move(value);
    return *this;
}

const std::string* ComputeFlowManifest::get_asset(std::string_view key) const
{
    auto it = _assets.find(std::string(key));
    if (it == _assets.end()) {
        return nullptr;
    }
    return &it->second;
}

std::string* ComputeFlowManifest::get_asset(std::string_view key)
{
    auto it = _assets.find(std::string(key));
    if (it == _assets.end()) {
        return nullptr;
    }
    return &it->second;
}

bool ComputeFlowManifest::has_asset(std::string_view key) const
{
    return _assets.find(std::string(key)) != _assets.end();
}

size_t ComputeFlowManifest::num_modules() const noexcept {
    return _items.size();
}

bool ComputeFlowManifest::has_module(std::string_view key) const {
    return _idx.find(std::string(key)) != _idx.end();
}

bool ComputeFlowManifest::has_constant(std::string_view key) const {
    return _constants.find(std::string(key)) != _constants.end();
}

bool ComputeFlowManifest::has_config(std::string_view key) const {
    return _configs.find(std::string(key)) != _configs.end();
}

const std::vector<ModelData>& ComputeFlowManifest::models() const {
    return _items;
}

const ConstantValue* ComputeFlowManifest::get_constant_value(std::string_view key) const {
    auto it = _constants.find(std::string(key));
    return (it == _constants.end()) ? nullptr : std::addressof(it->second);
}

std::string ComputeFlowManifest::resolve_config_string(
    const std::string& key,
    std::unordered_map<std::string, std::string>& cache,
    std::vector<std::string>& stack) const {
    auto cached = cache.find(key);
    if (cached != cache.end()) {
        return cached->second;
    }

    if (std::find(stack.begin(), stack.end(), key) != stack.end()) {
        throw std::runtime_error("Circular configs placeholder reference: " + key);
    }
    stack.push_back(key);

    auto it = _configs.find(key);
    if (it == _configs.end()) {
        stack.pop_back();
        return "";
    }

    std::string resolved;
    if (std::holds_alternative<std::string>(it->second)) {
        resolved = render_config_template(std::get<std::string>(it->second), cache, stack);
    } else {
        resolved = manifest_value_to_string(it->second);
    }

    cache.emplace(key, resolved);
    stack.pop_back();
    return resolved;
}

std::string ComputeFlowManifest::render_config_template(
    const std::string& templ,
    std::unordered_map<std::string, std::string>& cache,
    std::vector<std::string>& stack) const {
    std::string out;
    out.reserve(templ.size());

    size_t pos = 0;
    while (pos < templ.size()) {
        const size_t start = templ.find("{{", pos);
        if (start == std::string::npos) {
            out.append(templ.substr(pos));
            break;
        }
        out.append(templ.substr(pos, start - pos));

        const size_t end = templ.find("}}", start + 2);
        if (end == std::string::npos) {
            out.append(templ.substr(start));
            break;
        }

        const std::string placeholder = trim(std::string_view(templ).substr(start + 2, end - (start + 2)));
        auto it = _configs.find(placeholder);
        if (it != _configs.end()) {
            if (std::holds_alternative<std::string>(it->second)) {
                out.append(resolve_config_string(placeholder, cache, stack));
            } else {
                out.append(manifest_value_to_string(it->second));
            }
        } else {
            out.append(templ.substr(start, end - start + 2));
        }

        pos = end + 2;
    }

    return out;
}


template ComputeFlowManifest& ComputeFlowManifest::set_constant<float>(std::string, float);
template ComputeFlowManifest& ComputeFlowManifest::set_constant<double>(std::string, double);
template ComputeFlowManifest& ComputeFlowManifest::set_constant<int32_t>(std::string, int32_t);
template ComputeFlowManifest& ComputeFlowManifest::set_constant<int64_t>(std::string, int64_t);
template ComputeFlowManifest& ComputeFlowManifest::set_constant<bool>(std::string, bool);
template ComputeFlowManifest& ComputeFlowManifest::set_constant<std::string>(std::string, std::string);

template float ComputeFlowManifest::get_constant<float>(std::string_view) const;
template double ComputeFlowManifest::get_constant<double>(std::string_view) const;
template int32_t ComputeFlowManifest::get_constant<int32_t>(std::string_view) const;
template int64_t ComputeFlowManifest::get_constant<int64_t>(std::string_view) const;
template bool ComputeFlowManifest::get_constant<bool>(std::string_view) const;
template std::string ComputeFlowManifest::get_constant<std::string>(std::string_view) const;

template ComputeFlowManifest& ComputeFlowManifest::set_config<float>(std::string, float);
template ComputeFlowManifest& ComputeFlowManifest::set_config<double>(std::string, double);
template ComputeFlowManifest& ComputeFlowManifest::set_config<int32_t>(std::string, int32_t);
template ComputeFlowManifest& ComputeFlowManifest::set_config<int64_t>(std::string, int64_t);
template ComputeFlowManifest& ComputeFlowManifest::set_config<bool>(std::string, bool);
template ComputeFlowManifest& ComputeFlowManifest::set_config<std::string>(std::string, std::string);

template float ComputeFlowManifest::get_config<float>(std::string_view) const;
template double ComputeFlowManifest::get_config<double>(std::string_view) const;
template int32_t ComputeFlowManifest::get_config<int32_t>(std::string_view) const;
template int64_t ComputeFlowManifest::get_config<int64_t>(std::string_view) const;
template bool ComputeFlowManifest::get_config<bool>(std::string_view) const;
template std::string ComputeFlowManifest::get_config<std::string>(std::string_view) const;

}  // namespace compute_flow
