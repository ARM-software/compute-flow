//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/check.h"

#include "litert/c/litert_tensor_buffer.h"

namespace compute_flow::litert {

/** Owns a LiteRT tensor buffer lock until the current operation is finished. */
class LiteRTBufferLock {
public:
    LiteRTBufferLock(LiteRtTensorBuffer buffer, LiteRtTensorBufferLockMode mode)
        : _buffer(buffer) {
        COMPUTE_FLOW_CHECK(LiteRtLockTensorBuffer(buffer, &_data, mode) == kLiteRtStatusOk);
    }

    ~LiteRTBufferLock() { LiteRtUnlockTensorBuffer(_buffer); }

    LiteRTBufferLock(const LiteRTBufferLock&) = delete;
    LiteRTBufferLock& operator=(const LiteRTBufferLock&) = delete;

    void* data() const { return _data; }

private:
    LiteRtTensorBuffer _buffer;
    void* _data = nullptr;
};

}  // namespace compute_flow::litert
