//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/cli_examples/internal/run_config.h"

#include <fstream>
#include <unordered_set>

namespace compute_flow::bricks::internal {

std::vector<JsonConfig> load_run_configs(const std::string& path, std::initializer_list<const char*> allowed_arguments,
                                         std::initializer_list<const char*> required_arguments) {
    std::ifstream stream(path);
    COMPUTE_FLOW_CHECK(stream.is_open());

    const JsonConfig root = JsonConfig::parse(stream, nullptr, false);
    COMPUTE_FLOW_CHECK(!root.is_discarded());
    COMPUTE_FLOW_CHECK(root.is_array() && !root.empty());

    std::unordered_set<std::string> allowed = {"id"};

    for (const char* argument : allowed_arguments) {
        allowed.emplace(argument);
    }

    std::vector<JsonConfig> configs;
    configs.reserve(root.size());

    for (int64_t run_index = 0; run_index < static_cast<int64_t>(root.size()); ++run_index) {
        const JsonConfig& config = root[static_cast<size_t>(run_index)];
        COMPUTE_FLOW_CHECK(config.is_object());

        for (const auto& item : config.items()) {
            COMPUTE_FLOW_CHECK(allowed.count(item.key()) != 0);
        }

        COMPUTE_FLOW_CHECK(config.contains("id"));
        COMPUTE_FLOW_CHECK(config["id"].is_string() && !config["id"].get<std::string>().empty());

        for (const char* argument : required_arguments) {
            COMPUTE_FLOW_CHECK(config.contains(argument));
        }

        configs.push_back(config);
    }

    return configs;
}

}  // namespace compute_flow::bricks::internal
