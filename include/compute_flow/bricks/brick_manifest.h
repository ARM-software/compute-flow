//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "compute_flow/manifest.h"

namespace compute_flow::bricks {

using BrickManifest = ComputeFlowManifest;

/**
 * @brief Load a Compute Flow brick manifest from a JSON file.
 * @param path Path to the manifest JSON file.
 * @return User-facing brick manifest.
 */
BrickManifest load_manifest(const std::string& path);

}  // namespace compute_flow::bricks
