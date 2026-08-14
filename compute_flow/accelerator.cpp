//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/accelerator.h"

namespace compute_flow {

AcceleratorConfig AcceleratorConfig::cpu_only(size_t num_threads, bool enable_fp16) {
    return AcceleratorConfig{
        { AcceleratorPreference{AcceleratorType::Cpu, CpuOptions{num_threads, enable_fp16}} }
    };
}

bool AcceleratorConfig::has_accelerator(AcceleratorType t) const noexcept {
    for (const auto& p : priorities) {
        if (p.type == t) {
            return true;
        }
    }
    return false;
}

bool AcceleratorConfig::is_accelerator_first_priority(AcceleratorType t) const noexcept {
    return !priorities.empty() && priorities.front().type == t;
}

}  // namespace compute_flow
