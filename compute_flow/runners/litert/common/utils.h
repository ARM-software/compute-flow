//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compute_flow/check.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"

#include <cstdint>

namespace compute_flow::litert {

inline void check_status(LiteRtStatus status) {
    COMPUTE_FLOW_CHECK(status == kLiteRtStatusOk);
}

/** Byte width of a fixed-size LiteRT tensor element, or zero for an unsupported type. */
inline int64_t element_size(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeBool:
        case kLiteRtElementTypeInt8:
        case kLiteRtElementTypeUInt8: return 1;
        case kLiteRtElementTypeFloat16:
        case kLiteRtElementTypeBFloat16: return 2;
        case kLiteRtElementTypeFloat32:
        case kLiteRtElementTypeInt32:
        case kLiteRtElementTypeUInt32: return 4;
        case kLiteRtElementTypeFloat64:
        case kLiteRtElementTypeInt64:
        case kLiteRtElementTypeUInt64: return 8;
        default: return 0;
    }
}

}  // namespace compute_flow::litert
