//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/bricks/brick_version.h"

#include <limits>

namespace compute_flow::bricks {

namespace {

bool parse_version_component(std::string_view token, int32_t& value) {
    if (token.empty()) {
        return false;
    }

    int64_t parsed = 0;
    for (char ch : token) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = (parsed * 10) + (ch - '0');
        if (parsed > std::numeric_limits<int32_t>::max()) {
            return false;
        }
    }

    value = static_cast<int32_t>(parsed);
    return true;
}

}  // namespace

bool parse_brick_version(std::string_view version, BrickVersion& out_version) {
    if (version.empty()) {
        return false;
    }

    const size_t first_dot = version.find('.');
    if (first_dot == std::string_view::npos) {
        return false;
    }

    if (version.find('.', first_dot + 1) != std::string_view::npos) {
        return false;
    }

    std::string_view major_token = version.substr(0, first_dot);
    std::string_view minor_token = version.substr(first_dot + 1);

    if (!parse_version_component(major_token, out_version.major)) {
        return false;
    }
    if (!parse_version_component(minor_token, out_version.minor)) {
        return false;
    }

    return true;
}

}  // namespace compute_flow::bricks
