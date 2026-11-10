//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace compute_flow::custom_ops {

class Registry;

/** Bridge one custom operator kernel interface version to LiteRT. */
class LiteRTInterfaceAdapter {
public:
    virtual ~LiteRTInterfaceAdapter() = default;

    virtual std::map<std::string, std::string> register_cpu_model_custom_ops(
        LiteRtModel model, LiteRtOptions options, size_t num_threads) = 0;
    virtual bool consume_failure() = 0;
};

/** Own the active interface adapters and their profile labels for one LiteRT model. */
class LiteRTCustomOps {
public:
    explicit LiteRTCustomOps(const Registry& registry);
    ~LiteRTCustomOps();

    LiteRTCustomOps(const LiteRTCustomOps&) = delete;
    LiteRTCustomOps& operator=(const LiteRTCustomOps&) = delete;

    /** Register the model's CPU custom operators before LiteRT compiles it. */
    void register_cpu_model_custom_ops(LiteRtModel model, LiteRtOptions options, size_t num_threads);
    /** Return whether any custom operator callback failed since the previous call. */
    bool consume_failure();
    /** Return the selected kernel's profile label, or the original operator name. */
    std::string_view profile_label(std::string_view operator_name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace compute_flow::custom_ops
