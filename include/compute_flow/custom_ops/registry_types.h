/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_TYPES_H
#define COMPUTE_FLOW_CUSTOM_OPS_REGISTRY_TYPES_H

#include <string_view>
#include <type_traits>
#include <variant>

namespace compute_flow::custom_ops {

/** Hardware features that a custom operator kernel can require. */
struct HardwareFeature {
    enum class Arm {
        Neon,
        NeonDotProd,
        NeonFp16,
        NeonI8mm,
        Sve,
        Sve2,
        Sme,
        Sme2,
    };

    // Add each new CPU target enum here and to the CpuFeature std::variant.

    using CpuFeature = std::variant<Arm>;

    /** Return a target-qualified name for a feature. */
    static std::string_view to_string(const CpuFeature& feature) {
        return std::visit([](auto value) -> std::string_view {
            using T = decltype(value);

            if constexpr (std::is_same_v<T, Arm>) {
                switch (value) {
                    case Arm::Neon: return "arm_neon";
                    case Arm::NeonDotProd: return "arm_neon_dotprod";
                    case Arm::NeonFp16: return "arm_neon_fp16";
                    case Arm::NeonI8mm: return "arm_neon_i8mm";
                    case Arm::Sve: return "arm_sve";
                    case Arm::Sve2: return "arm_sve2";
                    case Arm::Sme: return "arm_sme";
                    case Arm::Sme2: return "arm_sme2";
                }
            }

            return "unknown";
        }, feature);
    }
};

using CpuFeature = HardwareFeature::CpuFeature;

/** Result of a kernel lookup. */
enum class LookupStatus {
    Ok = 0,
    InvalidArgument,
    BrickNotFound,
    OperatorNotFound,
    SchemaMismatch,
    /** More than one registry has the selected brick identity. */
    Ambiguous,
};

}  // namespace compute_flow::custom_ops

#endif
