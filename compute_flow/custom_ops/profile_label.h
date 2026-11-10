//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/custom_ops/registry_types.h"

#include <string>
#include <string_view>
#include <vector>

namespace compute_flow::custom_ops {

/** Format the selected kernel's profiler label suffix.
 *  @param kernel_name Selected kernel name; an empty name is shown as "unknown".
 *  @param required_features CPU features required by the kernel, in display order.
 *  @return A suffix such as " [kernel=reference; hw=none]". */
std::string get_custom_op_profile_suffix(
    std::string_view kernel_name,
    const std::vector<CpuFeature>& required_features);

}  // namespace compute_flow::custom_ops
