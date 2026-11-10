//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/check.h"
#include "compute_flow/custom_ops/registry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using compute_flow::custom_ops::CpuFeature;
using compute_flow::custom_ops::HardwareFeature;
using compute_flow::custom_ops::Registry;
using Arm = HardwareFeature::Arm;

struct TestKernel {
    static constexpr uint64_t k_interface_version = 1;
    std::string operator_name;
    uint64_t schema_version;
    std::vector<CpuFeature> required_features;
};

struct FutureKernel {
    static constexpr uint64_t k_interface_version = 2;
    std::string operator_name;
    uint64_t schema_version;
    std::vector<CpuFeature> required_features;
};

}  // namespace

int main() {
    Registry registry("tests:dispatch");
    registry.add_kernel(TestKernel{"add", 1, {Arm::Sme2}});
    registry.add_kernel(TestKernel{"add", 1, {Arm::NeonDotProd, Arm::NeonI8mm}});
    registry.add_kernel(TestKernel{"add", 1, {Arm::NeonDotProd}});
    registry.add_kernel(TestKernel{"add", 1, {}});  // Required portable reference.

    using compute_flow::custom_ops::select_kernel;

    const auto* sme2 = select_kernel<TestKernel>(registry, "add", {Arm::Sme2, Arm::NeonDotProd});
    const auto* i8mm = select_kernel<TestKernel>(
        registry, "add", {Arm::NeonDotProd, Arm::NeonI8mm});
    const auto* dotprod = select_kernel<TestKernel>(registry, "add", {Arm::NeonDotProd});
    const auto* reference = select_kernel<TestKernel>(registry, "add", {});

    COMPUTE_FLOW_CHECK(sme2 == &registry.data()[0]);
    COMPUTE_FLOW_CHECK(i8mm == &registry.data()[1]);
    COMPUTE_FLOW_CHECK(dotprod == &registry.data()[2]);
    COMPUTE_FLOW_CHECK(reference == &registry.data()[3]);
    COMPUTE_FLOW_CHECK(select_kernel<TestKernel>(registry, "missing", {}) == nullptr);
    COMPUTE_FLOW_CHECK(HardwareFeature::to_string(CpuFeature{Arm::Neon}) == "arm_neon");
    COMPUTE_FLOW_CHECK(HardwareFeature::to_string(CpuFeature{Arm::Sme2}) == "arm_sme2");

    Registry generic("tests:generic");
    generic.add_kernel(TestKernel{"add", 1, {}});  // Generic implementation has priority.
    generic.add_kernel(TestKernel{"add", 1, {}});  // Required portable reference.
    COMPUTE_FLOW_CHECK(select_kernel<TestKernel>(generic, "add", {}) == &generic.data()[0]);

    Registry missing_fallback("tests:missing-fallback");
    missing_fallback.add_kernel(TestKernel{"add", 1, {Arm::Sme2}});
    COMPUTE_FLOW_CHECK(select_kernel<TestKernel>(missing_fallback, "add", {Arm::Sme2}) == nullptr);

    Registry wrong_order("tests:wrong-order");
    wrong_order.add_kernel(TestKernel{"add", 1, {}});
    wrong_order.add_kernel(TestKernel{"add", 1, {Arm::Sme2}});
    COMPUTE_FLOW_CHECK(select_kernel<TestKernel>(wrong_order, "add", {Arm::Sme2}) == nullptr);

    Registry mixed_versions("tests:mixed-versions");
    mixed_versions.add_kernel(TestKernel{"add", 1, {}});
    mixed_versions.add_kernel(TestKernel{"add", 2, {}});
    COMPUTE_FLOW_CHECK(select_kernel<TestKernel>(mixed_versions, "add", {}) == nullptr);

    Registry mixed_interfaces("tests:mixed-interfaces");
    mixed_interfaces.add_kernel(FutureKernel{"add", 1, {}});
    mixed_interfaces.add_kernel(TestKernel{"add", 1, {}});
    COMPUTE_FLOW_CHECK(
        select_kernel<TestKernel>(mixed_interfaces, "add", {}) == &mixed_interfaces.data()[1]);
    return 0;
}
