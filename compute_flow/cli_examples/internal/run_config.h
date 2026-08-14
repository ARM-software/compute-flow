//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "compute_flow/check.h"

namespace compute_flow::bricks::internal {

using JsonConfig = nlohmann::json;

/**
 * @brief Load and validate run settings from a JSON file.
 *
 * @param[in] path JSON file path.
 * @param[in] allowed_arguments Allowed CLI arguments.
 * @param[in] required_arguments Required CLI arguments.
 * @return Validated run settings.
 */
std::vector<JsonConfig> load_run_configs(const std::string& path, std::initializer_list<const char*> allowed_arguments,
                                         std::initializer_list<const char*> required_arguments);

/**
 * @brief Get an argument value for one run.
 *
 * @param[in] config Run settings.
 * @param[in] argument Argument name.
 * @param[in] default_value Value returned when the argument is absent.
 * @param[in] run_index Zero-based run index.
 * @return Argument value or default_value.
 */
template <typename T>
T get_run_value(const JsonConfig& config, const char* argument, T default_value, int64_t run_index) {
    (void)run_index;
    if (!config.contains(argument)) {
        return default_value;
    }

    try {
        const T value = config.at(argument).get<T>();

        if constexpr (std::is_same_v<T, std::string>) {
            COMPUTE_FLOW_CHECK(value.rfind("@sample", 0) != 0);
            COMPUTE_FLOW_CHECK(value.rfind("@manifest", 0) != 0);
        }

        return value;
    } catch (const nlohmann::json::exception&) {
        COMPUTE_FLOW_CHECK(false);
    }

    return default_value;
}

}  // namespace compute_flow::bricks::internal
