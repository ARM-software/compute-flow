//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace compute_flow {

struct ValidationResult {
    bool        ok = true;
    std::string error;

    explicit operator bool() const {
        return ok;
    }
};

} // namespace compute_flow

/**
 * @brief Check an always-on runtime condition and throw when it fails.
 * @param expr Condition that must evaluate to true for execution to continue.
 */
#define COMPUTE_FLOW_CHECK(expr)                                        \
    do {                                                                \
        if (!(expr)) {                                                   \
            const std::string message =                                  \
                std::string("Check failed: ") + #expr +                  \
                " at " + __FILE__ + ":" + std::to_string(__LINE__);     \
            std::cerr << "Error: " << message << std::endl;              \
            throw std::runtime_error(message);                           \
        }                                                               \
    } while (false)

#define COMPUTE_FLOW_VALIDATE(expr)                                     \
    do {                                                               \
        if (!(expr)) {                                                  \
            return {                                                    \
                false,                                                  \
                std::string("Validation failed: ") + #expr +            \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__)  \
            };                                                         \
        }                                                              \
    } while (false)

#define COMPUTE_FLOW_THROW_IF_ERROR(result_expr, context)          \
    do {                                                          \
        const auto result = (result_expr);                         \
        if (!result) {                                             \
            const std::string message =                            \
                std::string(context) + ": " + result.error;        \
            std::cerr << "Error: " << message << std::endl;        \
            throw std::runtime_error(message);                     \
        }                                                         \
    } while (false)

#define COMPUTE_FLOW_ASSERT(expr) assert(expr)
