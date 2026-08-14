//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "compute_flow/bricks/brick_manifest.h"

namespace compute_flow::bricks::internal {

/**
 * @brief Parse default and per-model CPU thread counts from a CLI thread specification.
 *
 * @param[in] spec Comma-separated specification containing an optional default followed by
 *                 model-specific overrides in @c model_name:num_threads form.
 * @param[in,out] default_threads Default thread count, updated when @p spec starts with one.
 * @return Model names mapped to their thread-count overrides.
 */
std::unordered_map<std::string, size_t> parse_hint_cpu_threads(const std::string& spec, size_t& default_threads);

/**
 * @brief Configure default and per-model CPU thread counts in a brick manifest.
 *
 * @param[in,out] manifest Brick manifest to configure.
 * @param[in] default_threads Default thread count for all models.
 * @param[in] model_threads Model names mapped to their thread-count overrides.
 */
void set_cpu_threads(BrickManifest& manifest, size_t default_threads,
                     const std::unordered_map<std::string, size_t>& model_threads);

}  // namespace compute_flow::bricks::internal
