//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string_view>

#include "compute_flow/bricks/brick_manifest.h"
#include "compute_flow/bricks/brick_interface_v1.h"

namespace compute_flow::bricks {

using Brick = std::unique_ptr<IBrickV1>;

/**
 * @brief Create a Compute Flow brick from a loaded manifest.
 * @param manifest The manifest containing the brick type/name metadata.
 * @return A brick instance matching the manifest metadata.
 */
Brick load_brick(const BrickManifest& manifest);

}  // namespace compute_flow::bricks
