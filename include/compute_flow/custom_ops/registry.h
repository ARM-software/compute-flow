/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_H
#define COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_H

#include "compute_flow/custom_ops/registry_types.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace compute_flow {
struct BrickInfo;
}

namespace compute_flow::custom_ops {

/** Shared metadata and a typed, interface-specific kernel implementation. */
struct KernelRegistration {
    /** Capture shared metadata and the concrete kernel interface type. */
    template <typename KernelT>
    explicit KernelRegistration(const KernelT& kernel, std::string_view kernel_name)
        : operator_name(kernel.operator_name),
          schema_version(kernel.schema_version),
          interface_version(KernelT::k_interface_version),
          required_features(kernel.required_features),
          kernel_name(kernel_name),
          _implementation(kernel) {}

    std::string operator_name;  ///< Name of the operator in the model.
    uint64_t schema_version;  ///< Operator schema version, independent of the kernel interface.
    uint64_t interface_version;  ///< Kernel interface version used to interpret the payload.
    std::vector<CpuFeature> required_features;  ///< Empty for the portable reference.
    std::string kernel_name;  ///< Implementation name for profiling, independent of its interface version.

    /** Return the implementation only when its C++ type matches the requested interface. */
    template <typename KernelT>
    const KernelT* as() const {
        return std::any_cast<KernelT>(&_implementation);
    }

private:
    std::any _implementation;
};

/** Brick-local collection of kernels from any supported custom operator interface. */
class Registry {
public:
    explicit Registry(const std::string& brick_variant_name)
        : _brick_variant_name(brick_variant_name) {}

    /** Owning brick identity in type:name form. */
    const std::string& brick_variant_name() const { return _brick_variant_name; }

    /** Append a typed implementation. Keep each operator's implementations contiguous and at one
     *  schema version. Within an interface version, append them in priority order with the required
     *  portable reference last.
     *  @param kernel_name Implementation name for profiling, or empty when unnamed. */
    template <typename KernelT>
    void add_kernel(const KernelT& kernel, std::string_view kernel_name = {}) {
        _kernel_implementations.emplace_back(kernel, kernel_name);
    }

    /** Append an existing registration with its interface-independent metadata intact. */
    void add_kernel(const KernelRegistration& registration) {
        _kernel_implementations.push_back(registration);
    }

    /** Pointer to the registered kernels. Invalidated when the registry is modified. */
    const KernelRegistration* data() const { return _kernel_implementations.data(); }
    /** Number of registered kernels. */
    size_t size() const { return _kernel_implementations.size(); }

private:
    std::string _brick_variant_name;
    std::vector<KernelRegistration> _kernel_implementations;
};

/** Return the linked registry for a brick type and name, or nullptr when none exists. */
const Registry* find_compiled_registry(const BrickInfo& brick_info);

/** A contiguous range of matching registrations owned by a registry. */
struct LookupResult {
    LookupStatus status;  ///< Outcome of the lookup.
    const KernelRegistration* kernels;  ///< First matching registration, or nullptr on failure.
    size_t num_kernels;  ///< Number of matching registrations.
};

/**
 * Find all compiled kernels for a brick, operator name, and schema version.
 * A successful result refers to registrations owned by the supplied registry vector.
 * The lookup does not allocate memory or invoke a kernel.
 */
LookupResult lookup_kernels(
    const std::vector<Registry>& registries,
    const std::string& brick_identity,
    const std::string& operator_name,
    uint64_t schema_version);

/** Detect the hardware features available to this process. */
std::vector<CpuFeature> detect_cpu_features();

/** Select the first compatible implementation of the requested interface in priority order.
 *  An operator has one schema version, and its final implementation must be a portable fallback. */
template <typename KernelT>
const KernelRegistration* select_kernel(
    const Registry& registry,
    const std::string& operator_name,
    const std::vector<CpuFeature>& available_features) {
    const KernelRegistration* selected = nullptr;
    const KernelRegistration* last_matching = nullptr;

    for (size_t index = 0; index < registry.size(); ++index) {
        const KernelRegistration* candidate = &registry.data()[index];

        if (candidate->operator_name != operator_name || candidate->as<KernelT>() == nullptr) {
            continue;
        }

        if (last_matching != nullptr && candidate->schema_version != last_matching->schema_version) {
            return nullptr;
        }

        last_matching = candidate;

        const bool compatible = std::all_of(
            candidate->required_features.begin(), candidate->required_features.end(),
            [&available_features](const CpuFeature& required) {
                return std::find(available_features.begin(), available_features.end(), required) !=
                       available_features.end();
            });

        if (selected == nullptr && compatible) {
            selected = candidate;
        }
    }

    if (last_matching == nullptr || !last_matching->required_features.empty()) {
        return nullptr;
    }

    return selected;
}

}  // namespace compute_flow::custom_ops

#endif
