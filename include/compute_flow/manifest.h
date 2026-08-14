//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/bricks/brick_version.h"
#include "compute_flow/model_data.h"

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace compute_flow {

class ComputeFlowManifest;

namespace bricks {
ComputeFlowManifest load_manifest(const std::string& path);
}

namespace detail {
void set_manifest_constants(
    ComputeFlowManifest& manifest,
    const std::string& key,
    const ConstantValue& value);
}

/**
 * @brief Brick metadata parsed from the manifest.
 */
struct BrickInfo {
    std::string type;    ///< Internal brick type (e.g., "audiogen").
    std::string name;    ///< Brick name from info.json (e.g., "StableAudioOpenSmall").
    bricks::BrickVersion version; ///< Brick manifest version (MAJOR.MINOR).
};

class ComputeFlowManifest {
public:
    ComputeFlowManifest& set_brick_info(const BrickInfo& brick_info);
    const BrickInfo& get_brick_info() const noexcept;
    ModelData& declare_module(std::string_view key);
    ModelData& operator[](std::string_view key);
    const ModelData& operator[](std::string_view key) const;
    ModelData& at(std::string_view key);
    const ModelData& at(std::string_view key) const;
    template <typename T> T get_constant(std::string_view key) const;
    const ConstantValue* get_constant_ptr(std::string_view key) const;
    template <typename T> T get_config(std::string_view key) const;
    ConfigValue get_config_value(std::string_view key) const;
    const ConfigValue* get_config_ptr(std::string_view key) const;
    template <class T> ComputeFlowManifest& set_config(std::string key, T value);
    ComputeFlowManifest& set_asset(std::string key, std::string value);
    const std::string* get_asset(std::string_view key) const;
    std::string* get_asset(std::string_view key);
    size_t num_modules() const noexcept;
    bool has_module(std::string_view key) const;
    bool has_constant(std::string_view key) const;
    bool has_config(std::string_view key) const;
    bool has_asset(std::string_view key) const;
    const std::vector<ModelData>& models() const;

private:
    friend ComputeFlowManifest bricks::load_manifest(const std::string& path);
    friend void detail::set_manifest_constants(
        ComputeFlowManifest& manifest,
        const std::string& key,
        const ConstantValue& value);

    template <class T> ComputeFlowManifest& set_constant(std::string key, T value);
    const ConstantValue* get_constant_value(std::string_view key) const;
    std::string resolve_config_string(
        const std::string& key,
        std::unordered_map<std::string, std::string>& cache,
        std::vector<std::string>& stack) const;
    std::string render_config_template(
        const std::string& templ,
        std::unordered_map<std::string, std::string>& cache,
        std::vector<std::string>& stack) const;

    BrickInfo                                    _brick_info;
    std::vector<ModelData>                       _items;
    std::unordered_map<std::string, size_t>      _idx;
    std::unordered_map<std::string, ConstantValue> _constants;
    std::unordered_map<std::string, ConfigValue> _configs;
    std::unordered_map<std::string, std::string> _assets;
};

}  // namespace compute_flow
