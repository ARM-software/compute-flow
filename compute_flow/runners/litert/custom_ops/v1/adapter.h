//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runners/litert/custom_ops/registration.h"

#include "compute_flow/custom_ops/registry.h"

#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace compute_flow::custom_ops {
class ThreadPool;
}

namespace compute_flow::custom_ops::v1 {

/** One adapter per compiled model; registers its custom operators and owns their callback state. */
class LiteRTAdapter final : public ::compute_flow::custom_ops::LiteRTInterfaceAdapter {
public:
    /** Borrow the compiled registry for this model. */
    explicit LiteRTAdapter(const ::compute_flow::custom_ops::Registry& registry);
    /** Release callback state after the compiled model has been destroyed. */
    ~LiteRTAdapter() override;

    LiteRTAdapter(const LiteRTAdapter&) = delete;
    LiteRTAdapter& operator=(const LiteRTAdapter&) = delete;

    /** Register each CPU operator name and schema version once before LiteRT compiles the model.
     *  @return Full profiler labels indexed by operator name. */
    std::map<std::string, std::string> register_cpu_model_custom_ops(
        LiteRtModel model, LiteRtOptions options, size_t num_threads) override;
    /** Return whether a callback failed since the previous call, then clear the failure. */
    bool consume_failure() override;

private:
    struct KernelState;
    std::vector<std::unique_ptr<KernelState>> _kernels;
    const ::compute_flow::custom_ops::Registry& _registry;
    // KernelState borrows this shared pool for its LiteRT callbacks.
    std::unique_ptr<ThreadPool> _thread_pool;
};

}  // namespace compute_flow::custom_ops::v1
