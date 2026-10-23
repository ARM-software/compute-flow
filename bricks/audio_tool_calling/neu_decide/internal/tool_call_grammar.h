//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace compute_flow::bricks::audio_tool_calling::neu_decide {

struct Tool {
    std::string name;
    std::vector<std::string> arguments;
};

struct PrefixResult {
    bool valid    = false;
    bool complete = false;
};

// Checks prefixes against NeuDecide's tool_call_grammar() rules: known tools and argument keys,
// with arbitrary JSON values. Source:
// https://github.com/neuphonic/neudecide/blob/548134f512f11dfe249acde3110c05c3a4774871/neudecide/grammar.py
PrefixResult match_tool_calls(std::string_view text, const std::vector<Tool>& tools);

}  // namespace compute_flow::bricks::audio_tool_calling::neu_decide
