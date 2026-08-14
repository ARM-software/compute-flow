//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <variant>
#include <vector>

namespace compute_flow {

enum class AcceleratorType {
    Cpu,
    Gpu
};

struct CpuOptions {
    size_t num_threads = 0;
    bool   enable_fp16 = 0;
};

struct GpuOptions {
    size_t device_index = 0;
};

using AcceleratorParams = std::variant<CpuOptions, GpuOptions>;

struct AcceleratorPreference {
    AcceleratorType type;
    AcceleratorParams params;
};

struct AcceleratorConfig {
    // Express priority by order: first wins if available; falls back otherwise.
    std::vector<AcceleratorPreference> priorities;

    // Convenience helpers to set CPU only acceleration
    static AcceleratorConfig cpu_only(size_t num_threads = 0, bool enable_fp16 = false);

    bool has_accelerator(AcceleratorType t) const noexcept;

    bool is_accelerator_first_priority(AcceleratorType t) const noexcept;
};

}  // namespace compute_flow
