//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string_view>

namespace compute_flow::bricks {

/**
 * @brief Parsed brick manifest version.
 */
struct BrickVersion {
    int32_t major = 0;
    int32_t minor = 0;
};

/**
 * @brief Parse a brick version string in MAJOR.MINOR form.
 * @param version The version string from the manifest.
 * @param out_version Parsed version fields.
 * @return True when the version string is valid.
 */
bool parse_brick_version(std::string_view version, BrickVersion& out_version);

}  // namespace compute_flow::bricks
