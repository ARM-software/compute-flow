/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_flow/custom_ops/registry.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace compute_flow::custom_ops {
namespace {

#if defined(__aarch64__) && defined(__APPLE__)
bool sysctl_feature(const char* name) {
    int32_t enabled = 0;
    size_t size = sizeof(enabled);
    return sysctlbyname(name, &enabled, &size, nullptr, 0) == 0 && enabled != 0;
}
#endif

}  // namespace

LookupResult lookup_kernels(
    const std::vector<Registry>& registries,
    const std::string& brick_identity,
    const std::string& operator_name,
    uint64_t schema_version) {
    const Registry* matching_registry = nullptr;

    for (const Registry& registry : registries) {
        if (brick_identity != registry.brick_variant_name()) {
            continue;
        }

        if (matching_registry != nullptr) {
            return {LookupStatus::Ambiguous, nullptr, 0};
        }

        matching_registry = &registry;
    }

    if (matching_registry == nullptr) {
        return {LookupStatus::BrickNotFound, nullptr, 0};
    }

    size_t first_index = 0;
    size_t num_matches = 0;
    bool found_operator = false;

    for (size_t kernel_index = 0; kernel_index < matching_registry->size(); ++kernel_index) {
        const KernelRegistration* candidate = &matching_registry->data()[kernel_index];

        if (operator_name != candidate->operator_name) {
            continue;
        }

        found_operator = true;

        if (candidate->schema_version != schema_version) {
            continue;
        }

        if (num_matches == 0) {
            first_index = kernel_index;
        } else if (kernel_index != first_index + num_matches) {
            return {LookupStatus::InvalidArgument, nullptr, 0};
        }

        ++num_matches;
    }

    if (num_matches != 0) {
        return {LookupStatus::Ok, &matching_registry->data()[first_index], num_matches};
    }

    if (found_operator) {
        return {LookupStatus::SchemaMismatch, nullptr, 0};
    }

    return {LookupStatus::OperatorNotFound, nullptr, 0};
}

std::vector<CpuFeature> detect_cpu_features() {
    std::vector<CpuFeature> features;

#if defined(__aarch64__) && defined(__linux__)
    // Linux® AArch64 AT_HWCAP and AT_HWCAP2 bit positions are a stable userspace ABI.
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    const auto add_if = [&features](bool available, HardwareFeature::Arm feature) {
        if (available) {
            features.emplace_back(feature);
        }
    };

    add_if((hwcap & (1UL << 1)) != 0, HardwareFeature::Arm::Neon);
    add_if((hwcap & (1UL << 20)) != 0, HardwareFeature::Arm::NeonDotProd);
    add_if((hwcap & (1UL << 10)) != 0, HardwareFeature::Arm::NeonFp16);
    add_if((hwcap2 & (1UL << 13)) != 0, HardwareFeature::Arm::NeonI8mm);
    add_if((hwcap2 & (1UL << 23)) != 0, HardwareFeature::Arm::Sme);
    add_if((hwcap2 & (1UL << 37)) != 0, HardwareFeature::Arm::Sme2);
    add_if((hwcap & (1UL << 22)) != 0, HardwareFeature::Arm::Sve);
    add_if((hwcap2 & (1UL << 1)) != 0, HardwareFeature::Arm::Sve2);
#elif defined(__aarch64__) && defined(__APPLE__)
    const auto add_if = [&features](const char* name, HardwareFeature::Arm feature) {
        if (sysctl_feature(name)) {
            features.emplace_back(feature);
        }
    };

    add_if("hw.optional.neon", HardwareFeature::Arm::Neon);
    add_if("hw.optional.arm.FEAT_DotProd", HardwareFeature::Arm::NeonDotProd);
    add_if("hw.optional.arm.FEAT_FP16", HardwareFeature::Arm::NeonFp16);
    add_if("hw.optional.arm.FEAT_I8MM", HardwareFeature::Arm::NeonI8mm);
    add_if("hw.optional.arm.FEAT_SME", HardwareFeature::Arm::Sme);
    add_if("hw.optional.arm.FEAT_SME2", HardwareFeature::Arm::Sme2);
#endif

    return features;
}

}  // namespace compute_flow::custom_ops
