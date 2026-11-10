//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "profile_label.h"

#include <cstddef>
#include <string_view>

namespace compute_flow::custom_ops {

std::string get_custom_op_profile_suffix(
    std::string_view kernel_name,
    const std::vector<CpuFeature>& required_features) {
    const std::string_view display_name = kernel_name.empty() ? "unknown" : kernel_name;
    std::string label = " [kernel=";
    label += display_name;
    label += "; hw=";

    if (required_features.empty()) {
        label += "none";
    } else {
        for (size_t i = 0; i < required_features.size(); ++i) {
            if (i != 0) {
                label += '+';
            }

            label += HardwareFeature::to_string(required_features[i]);
        }
    }

    label += ']';
    return label;
}

}  // namespace compute_flow::custom_ops
