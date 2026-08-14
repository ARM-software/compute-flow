//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/cli_examples/internal/image.h"

#include "compute_flow/compute_flow.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace compute_flow::bricks::internal {
namespace {

constexpr int32_t k_num_channels = 3;
constexpr float k_pixel_scale = 1.0F / 255.0F;

template <typename PixelType>
std::vector<float> interleaved_rgb_to_chw(const PixelType* image, int32_t width, int32_t height, float scale) {
    const int64_t num_pixels = static_cast<int64_t>(height) * width;
    std::vector<float> planar(static_cast<std::size_t>(k_num_channels * num_pixels));
    for (int32_t channel = 0; channel < k_num_channels; ++channel) {
        for (int64_t pixel = 0; pixel < num_pixels; ++pixel) {
            const int64_t source_offset = (pixel * k_num_channels) + channel;
            const int64_t target_offset = (channel * num_pixels) + pixel;
            planar[static_cast<std::size_t>(target_offset)] =
                static_cast<float>(image[source_offset]) * scale;
        }
    }
    return planar;
}

} // namespace

std::vector<float> load_rgb_image_chw(const std::string& path,
                                      int32_t& width,
                                      int32_t& height,
                                      ImageDecodeMode decode_mode) {
    if (decode_mode == ImageDecodeMode::NormalizedU8) {
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> image(
            stbi_load(path.c_str(), &width, &height, nullptr, k_num_channels), stbi_image_free);
        COMPUTE_FLOW_CHECK(image != nullptr);
        return interleaved_rgb_to_chw(image.get(), width, height, k_pixel_scale);
    }

    COMPUTE_FLOW_CHECK(decode_mode == ImageDecodeMode::StbFloat);
    std::unique_ptr<float, decltype(&stbi_image_free)> image(
        stbi_loadf(path.c_str(), &width, &height, nullptr, k_num_channels), stbi_image_free);
    COMPUTE_FLOW_CHECK(image != nullptr);
    return interleaved_rgb_to_chw(image.get(), width, height, 1.0F);
}

} // namespace compute_flow::bricks::internal
