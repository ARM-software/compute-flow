//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace compute_flow::bricks::internal {

/**
 * @brief Pixel conversion applied while loading an image.
 */
enum class ImageDecodeMode {
    /** Decode to 8-bit pixels, then normalize to the range [0, 1]. */
    NormalizedU8,

    /** Use STB's floating-point decode and conversion behavior. */
    StbFloat
};

/**
 * @brief Load an image as planar floating-point RGB data.
 *
 * @param[in] path Path to the input image.
 * @param[out] width Decoded image width in pixels.
 * @param[out] height Decoded image height in pixels.
 * @param[in] decode_mode Pixel conversion to apply while decoding.
 * @return RGB image data in CHW order.
 */
std::vector<float> load_rgb_image_chw(const std::string& path,
                                      int32_t& width,
                                      int32_t& height,
                                      ImageDecodeMode decode_mode);

} // namespace compute_flow::bricks::internal
