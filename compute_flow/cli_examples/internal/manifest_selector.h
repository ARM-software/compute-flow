//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace compute_flow::bricks::internal {

std::string get_manifest_path(const std::string& value, const std::string& brick_type);
std::string get_sample_path(const std::string& value, const std::string& manifest_path);

} // namespace compute_flow::bricks::internal
