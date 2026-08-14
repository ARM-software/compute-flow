//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/cli_examples/internal/thread_config.h"

#include <sstream>

#include "compute_flow/accelerator.h"

namespace compute_flow::bricks::internal {

std::unordered_map<std::string, size_t> parse_hint_cpu_threads(const std::string& spec, size_t& default_threads) {
    std::unordered_map<std::string, size_t> model_threads;
    if (spec.empty()) {
        return model_threads;
    }

    std::stringstream spec_stream(spec);
    std::string item;
    bool first_item = true;

    while (std::getline(spec_stream, item, ',')) {
        const size_t separator = item.find(':');
        if (separator == std::string::npos) {
            if (first_item) {
                default_threads = static_cast<size_t>(std::stoull(item));
            }
        } else {
            const std::string model_name = item.substr(0, separator);
            const size_t num_threads = static_cast<size_t>(std::stoull(item.substr(separator + 1)));
            model_threads[model_name] = num_threads;
        }
        first_item = false;
    }

    return model_threads;
}

void set_cpu_threads(BrickManifest& manifest, size_t default_threads,
                     const std::unordered_map<std::string, size_t>& model_threads) {
    for (const auto& model : manifest.models()) {
        manifest[model.model_name].acc_config = AcceleratorConfig::cpu_only(default_threads);
    }

    for (const auto& [model_name, threads] : model_threads) {
        manifest[model_name].acc_config = AcceleratorConfig::cpu_only(threads);
    }
}

}  // namespace compute_flow::bricks::internal
