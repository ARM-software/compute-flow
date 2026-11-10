//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "runners/litert/custom_ops/registration.h"
#include "runners/litert/custom_ops/v1/adapter.h"

#include "compute_flow/custom_ops/registry.h"
#include "compute_flow/custom_ops/v1/kernel.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace compute_flow::custom_ops {

struct LiteRTCustomOps::Impl {
    explicit Impl(const Registry& compiled_registry)
        : registry(compiled_registry) {}

    const Registry& registry;
    std::vector<std::unique_ptr<LiteRTInterfaceAdapter>> adapters;
    std::map<std::string, std::string, std::less<>> profile_labels;
};

LiteRTCustomOps::LiteRTCustomOps(const Registry& registry)
    : _impl(std::make_unique<Impl>(registry)) {}

LiteRTCustomOps::~LiteRTCustomOps() = default;

void LiteRTCustomOps::register_cpu_model_custom_ops(LiteRtModel model, LiteRtOptions options, size_t num_threads) {
    std::set<uint64_t> registered_versions;

    for (size_t index = 0; index < _impl->registry.size(); ++index) {
        const auto& registration = _impl->registry.data()[index];

        if (!registered_versions.insert(registration.interface_version).second) {
            continue;
        }

        std::unique_ptr<LiteRTInterfaceAdapter> adapter;

        switch (registration.interface_version) {
            case v1::Kernel::k_interface_version:
                adapter = std::make_unique<v1::LiteRTAdapter>(_impl->registry);
                break;
            default:
                throw std::invalid_argument("Unsupported LiteRT custom operator interface version: " +
                                            std::to_string(registration.interface_version));
        }

        // Keep callback state alive even if registration fails after LiteRT borrows it.
        _impl->adapters.push_back(std::move(adapter));
        const auto labels = _impl->adapters.back()->register_cpu_model_custom_ops(model, options, num_threads);

        _impl->profile_labels.insert(labels.begin(), labels.end());
    }
}

bool LiteRTCustomOps::consume_failure() {
    bool failed = false;

    // Check every interface version that registered callbacks for this model.
    for (const auto& adapter : _impl->adapters) {
        failed = adapter->consume_failure() || failed;
    }

    return failed;
}

std::string_view LiteRTCustomOps::profile_label(std::string_view operator_name) const {
    const auto label = _impl->profile_labels.find(operator_name);

    if (label != _impl->profile_labels.end()) {
        return label->second;
    }

    return operator_name;
}

}  // namespace compute_flow::custom_ops
