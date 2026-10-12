//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Yolo.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <cmath>
#include <variant>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static int64_t get_longest_string_in_bytes(const std::vector<std::string>& v) {
    std::size_t max_bytes = 0;

    for (const auto& s : v) {
        max_bytes = std::max(max_bytes, s.size());
    }

    return static_cast<int64_t>(max_bytes);
}

static std::vector<std::string> load_class_names_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    json j;
    f >> j;

    // Find maximum id
    int64_t max_id = -1;
    for (const auto& [token, id] : j.items()) {
        int64_t idx = id.get<int64_t>();
        if (idx < 0) {
            throw std::runtime_error("Negative class id found in JSON");
        }
        max_id = std::max(max_id, idx);
    }

    // Allocate vector large enough to hold highest id
    std::vector<std::string> vocab(static_cast<size_t>(max_id + 1));

    // Fill vector
    for (const auto& [token, id] : j.items()) {
        int64_t idx = id.get<int64_t>();
        vocab[static_cast<size_t>(idx)] = token;
    }

    return vocab;
}


static float iou_xyxy(const std::array<float,4>& a, const std::array<float,4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float areaA = std::max(0.0f, a[2]-a[0]) * std::max(0.0f, a[3]-a[1]);
    const float areaB = std::max(0.0f, b[2]-b[0]) * std::max(0.0f, b[3]-b[1]);
    const float uni = areaA + areaB - inter;
    if (uni <= 0.0f) {
        return 0.0f;
    }
    return inter / uni;
}

static std::vector<int64_t> nms_indices(const std::vector<std::array<float,4>>& boxes,
                                        const std::vector<float>& scores,
                                        const std::vector<int64_t>& idxs,
                                        float iou_thresh,
                                        int64_t max_detections) {
    std::vector<int64_t> order = idxs;
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b){
        if (scores[a] != scores[b]) {
            return scores[a] > scores[b];
        }
        return a < b;
    });

    std::vector<int64_t> keep;
    for (int64_t oi = 0; oi < order.size(); ++oi) {
        int64_t i = order[oi];
        bool keep_i = true;
        for (int64_t kept : keep) {
            if (iou_xyxy(boxes[i], boxes[kept]) > iou_thresh) {
                keep_i = false;
                break;
            }
        }
        if (keep_i) {
            keep.push_back(i);
            if (keep.size() >= max_detections) {
                break;
            }
        }
    }
    return keep;
}

namespace compute_flow::bricks::object_detection {

constexpr int32_t k_brick_major_version = 1;

ValidationResult Yolo::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "object_detection");
    COMPUTE_FLOW_VALIDATE(info.name == "Yolo");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    // Validate presence of the required model
    COMPUTE_FLOW_VALIDATE(manifest.has_module("yolo"));

    // Validate presence of the model path
    COMPUTE_FLOW_VALIDATE(manifest["yolo"].model_path != "");

    // Validate the target framework.
    COMPUTE_FLOW_VALIDATE(manifest["yolo"].framework_type == RunnerType::ONNXRT ||
               manifest["yolo"].framework_type == RunnerType::ExecuTorch ||
               manifest["yolo"].framework_type == RunnerType::LiteRT);

    // Check whether the models only have 1 accelerator type
    COMPUTE_FLOW_VALIDATE(manifest["yolo"].acc_config.priorities.size() == 1);

    // Only CPU is supported as accelerator type
    COMPUTE_FLOW_VALIDATE(manifest["yolo"].acc_config.has_accelerator(AcceleratorType::Cpu));

    // Validate presence of required metadata
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("version"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("is_nms_exported"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("confidence_threshold"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("nms_threshold"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_detections_per_image"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("are_coordinates_normalized"));

    // Validate presence of required asset
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("class_names"));
    return {};
}

void Yolo::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
    // TODO: Implement configuration updates in a separate MR.
}

void Yolo::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    RunnerType yolo_runner_type = manifest["yolo"].framework_type;

    _runner["yolo"] = make_runner(yolo_runner_type);
    _runner["yolo"]->setup(manifest["yolo"], profiler);
}

void Yolo::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _confidence_thr           = manifest.get_constant<float>("confidence_threshold");
    _nms_thr                  = manifest.get_constant<float>("nms_threshold");
    _max_detections_per_image = manifest.get_constant<int64_t>("max_detections_per_image");
    _version                  = manifest.get_constant<int64_t>("version");
    _is_nms_exported          = manifest.get_constant<bool>("is_nms_exported");
    _are_coordinates_normalized  = manifest.get_constant<bool>("are_coordinates_normalized");

    COMPUTE_FLOW_ASSERT(_version == 5 || _version == 26);
    COMPUTE_FLOW_ASSERT(_is_nms_exported == false);

    const std::string class_names_file = *manifest.get_asset("class_names");
    _class_names = load_class_names_from_json(class_names_file);

    _max_str_in_bytes = get_longest_string_in_bytes(_class_names);

    _out_boxes.reserve(_max_detections_per_image * 4);
    _out_scores.reserve(_max_detections_per_image);
    _out_class_ids.reserve(_max_detections_per_image);
    _out_class_names.reserve(_max_detections_per_image * _max_str_in_bytes);
    _src_model_shape.resize(2);
}

void Yolo::warmup_impl(ProfilerView profiler) {
    TensorDict input;

    input.declare_tensor("img");

    auto fmt = _runner["yolo"]->get_input_format("img");

    std::vector<uint8_t> mock_input(fmt.get_nbytes(), 0);

    input["img"].data   = (void *)mock_input.data();
    input["img"].format = fmt;

    // We can ignore the output
    run_impl(input, profiler);
}

void Yolo::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    _is_output_ready = false;
    _out_boxes.clear();
    _out_scores.clear();
    _out_class_ids.clear();
    _out_class_names.clear();
    _src_model_shape.assign(2, 0);
}

bool Yolo::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict Yolo::run_impl(const TensorDict& input, ProfilerView profiler) {
    // Validate number of input
    // Validate type of input
    // Validate shape


    auto img_model_fmt = _runner["yolo"]->get_input_format("img");
    auto img_user_fmt  = input["img"].format;

    const int64_t src_model_num_dims = img_model_fmt.shape.size();
    const int64_t src_user_num_dims  = img_user_fmt.shape.size();

    // The dynamic shape for the model's input tensor is not supported
    COMPUTE_FLOW_ASSERT(img_model_fmt.shape[src_model_num_dims - 1] > 0);
    COMPUTE_FLOW_ASSERT(img_model_fmt.shape[src_model_num_dims - 2] > 0);

    const int64_t src_model_w = img_model_fmt.shape[src_model_num_dims - 1];
    const int64_t src_model_h = img_model_fmt.shape[src_model_num_dims - 2];
    const int64_t src_input_w = img_user_fmt.shape[src_user_num_dims - 1];
    const int64_t src_input_h = img_user_fmt.shape[src_user_num_dims - 2];

    COMPUTE_FLOW_ASSERT(src_input_w > 0);
    COMPUTE_FLOW_ASSERT(src_input_h > 0);

    // Store the source image resolution. Output boxes are mapped back to this
    // coordinate system before they are returned to the caller.
    _src_model_shape[0] = src_input_w;
    _src_model_shape[1] = src_input_h;

    // Rescale the image if necessary to align with the expected model's input
    if((src_model_w != src_input_w) ||
       (src_model_h != src_input_h)) {

        // Scale/crop required. So, allocate tensor for the input image
        _internal_memory.allocate("img", img_model_fmt);

        auto img_scaled = _internal_memory.get_tensor("img");

        // Rescale the input image
        custom_op_scale_crop_bilinear_chw(input["img"], img_scaled);

        _runner["yolo"]->set_input("img", img_scaled);

        _runner["yolo"]->run();

        // We must release this intermediate tensor because at the next call
        // we may have a different shape
        _internal_memory.release("img");
    } else {
        _runner["yolo"]->set_input("img", input["img"]);

        _runner["yolo"]->run();
    }

    auto t_out = _runner["yolo"]->get_output("output");

    std::vector<std::array<float,4>> boxes; // x1,y1,x2,y2
    std::vector<float>               scores;
    std::vector<int64_t>             class_ids;

    COMPUTE_FLOW_ASSERT(boxes.size() == 0);
    COMPUTE_FLOW_ASSERT(scores.size() == 0);
    COMPUTE_FLOW_ASSERT(class_ids.size() == 0);

    std::vector<int64_t> final_keep_indices;
    if (_version == 5) {
        custom_op_postprocessing_v5s(t_out, boxes, scores, class_ids);

        // Group indices by class
        std::unordered_map<int64_t, std::vector<int64_t>> per_class_indices;
        for (int64_t i = 0; i < boxes.size(); ++i) {
            per_class_indices[class_ids[i]].push_back(i);
        }

        // Apply per-class NMS
        for (auto &kv : per_class_indices) {
            const std::vector<int64_t>& idxs = kv.second;
            std::vector<int64_t> kept = nms_indices(boxes, scores, idxs, _nms_thr, _max_detections_per_image);
            final_keep_indices.insert(final_keep_indices.end(), kept.begin(), kept.end());
        }
    } else if (_version == 26) {
        custom_op_postprocessing_v26(t_out, boxes, scores, class_ids);

        final_keep_indices.reserve(boxes.size());
        for (int64_t i = 0; i < boxes.size(); ++i) {
            final_keep_indices.push_back(i);
        }
    } else {
        COMPUTE_FLOW_ASSERT(false);
    }

    // Sort by score
    std::sort(final_keep_indices.begin(), final_keep_indices.end(), [&](int64_t a, int64_t b){
        if (scores[a] != scores[b]) {
            return scores[a] > scores[b];
        }
        return a < b;
    });

    if (final_keep_indices.size() > _max_detections_per_image) {
        final_keep_indices.resize(_max_detections_per_image);
    }

    const int64_t num_detections = static_cast<int64_t>(final_keep_indices.size());

    _out_boxes.clear();
    _out_scores.clear();
    _out_class_ids.clear();
    _out_class_names.clear();

    _out_boxes.resize(num_detections * 4);
    _out_scores.resize(num_detections);
    _out_class_ids.resize(num_detections);
    _out_class_names.resize(num_detections * _max_str_in_bytes);

    // Reset to 0 to avoid left over chars from previous runs
    memset(_out_class_names.data(), 0, num_detections * _max_str_in_bytes);

    const float box_scale_x = _are_coordinates_normalized
        ? static_cast<float>(_src_model_shape[0])
        : static_cast<float>(_src_model_shape[0]) / static_cast<float>(src_model_w);
    const float box_scale_y = _are_coordinates_normalized
        ? static_cast<float>(_src_model_shape[1])
        : static_cast<float>(_src_model_shape[1]) / static_cast<float>(src_model_h);
    const float max_x = static_cast<float>(_src_model_shape[0] - 1);
    const float max_y = static_cast<float>(_src_model_shape[1] - 1);

    auto scale_box_x = [&](float x) -> float {
        return std::clamp(x * box_scale_x, 0.0f, max_x);
    };
    auto scale_box_y = [&](float y) -> float {
        return std::clamp(y * box_scale_y, 0.0f, max_y);
    };

    for (int64_t i = 0; i < num_detections; ++i) {
        int64_t idx = final_keep_indices[i];
        const auto& b = boxes[idx];

        _out_boxes[static_cast<size_t>(i)*4 + 0] = scale_box_x(b[0]);
        _out_boxes[static_cast<size_t>(i)*4 + 1] = scale_box_y(b[1]);
        _out_boxes[static_cast<size_t>(i)*4 + 2] = scale_box_x(b[2]);
        _out_boxes[static_cast<size_t>(i)*4 + 3] = scale_box_y(b[3]);

        _out_scores[static_cast<size_t>(i)]    = scores[idx];
        _out_class_ids[static_cast<size_t>(i)] = class_ids[idx];

        const int64_t class_id    = class_ids[idx];
        const std::string& name   = _class_names[static_cast<size_t>(class_id)];
        const uint8_t* class_name = reinterpret_cast<const uint8_t*>(name.data());

        memcpy((uint8_t*)_out_class_names.data() + i * _max_str_in_bytes, name.data(), name.size());
    }

    TensorDict output;
    output.declare_tensor("id");                 // int64 [num_detections]
    output.declare_tensor("score");              // float32 [num_detections]
    output.declare_tensor("name");               // string [num_detections, max_string_len]
    output.declare_tensor("box");                // float32 [num_detections, x1, y1, x2, y2] in input image coordinates

    // We only return the class with the highest probability.
    // Therefore, #num_classes_reported = 1
    output["id"].data         = (void*)_out_class_ids.data();
    output["id"].format.dt    = DataType::I64;
    output["id"].format.shape = {static_cast<int64_t>(num_detections), 1};

    output["score"].data         = (void*)_out_scores.data();
    output["score"].format.dt    = DataType::F32;
    output["score"].format.shape = {static_cast<int64_t>(num_detections), 1};

    // The name is returned as U8. Therefore, on the host side, we should decode it as utf-8
    output["name"].data         = (void*)_out_class_names.data();
    output["name"].format.dt    = DataType::U8;
    output["name"].format.shape = {static_cast<int64_t>(num_detections), 1, _max_str_in_bytes};

    output["box"].data         = (void*)_out_boxes.data();
    output["box"].format.dt    = DataType::F32;
    output["box"].format.shape = {static_cast<int64_t>(num_detections), 4};

    _is_output_ready = true;

    return output;
}

void Yolo::custom_op_scale_crop_bilinear_chw(const TensorData& src, TensorData& dst) {
    COMPUTE_FLOW_ASSERT(src.data != nullptr);
    COMPUTE_FLOW_ASSERT(dst.data != nullptr);
    COMPUTE_FLOW_ASSERT(src.format.dt == dst.format.dt);
    COMPUTE_FLOW_ASSERT(src.format.dt == DataType::F32);

    const auto src_shape = src.format.shape;
    const auto dst_shape = dst.format.shape;

    const int64_t src_num_dims = src_shape.size();
    const int64_t dst_num_dims = dst_shape.size();

    const int64_t src_c = src_shape[src_num_dims - 3];
    const int64_t src_h = src_shape[src_num_dims - 2];
    const int64_t src_w = src_shape[src_num_dims - 1];

    const int64_t dst_c = dst_shape[dst_num_dims - 3];
    const int64_t dst_h = dst_shape[dst_num_dims - 2];
    const int64_t dst_w = dst_shape[dst_num_dims - 1];

    // Batching is not supported
    if(src_num_dims == 4) {
        COMPUTE_FLOW_ASSERT(src_shape[0] == 1);
    }

    if(dst_num_dims == 4) {
        COMPUTE_FLOW_ASSERT(dst_shape[0] == 1);
    }

    COMPUTE_FLOW_ASSERT(src_c == dst_c);
    COMPUTE_FLOW_ASSERT(src_h > 0 && src_w > 0);
    COMPUTE_FLOW_ASSERT(dst_h > 0 && dst_w > 0);

    const float* src_ptr = reinterpret_cast<const float*>(src.data);
    float* dst_ptr = reinterpret_cast<float*>(dst.data);

    // Match PIL.Image.resize(..., Image.BILINEAR): direct resize to target size
    const float scale_x = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float scale_y = static_cast<float>(dst_h) / static_cast<float>(src_h);

    auto img_index = [&](int64_t c, int64_t y, int64_t x, int64_t width, int64_t height) -> int64_t {
        return x + (y * width) + c * width * height;
    };

    for (int64_t y = 0; y < dst_h; ++y) {
        for (int64_t x = 0; x < dst_w; ++x) {
            float src_x = static_cast<float>(x) / scale_x;
            float src_y = static_cast<float>(y) / scale_y;

            src_x = std::max(0.0f, std::min(src_x, static_cast<float>(src_w - 1)));
            src_y = std::max(0.0f, std::min(src_y, static_cast<float>(src_h - 1)));

            const int64_t x0 = static_cast<int64_t>(src_x);
            const int64_t y0 = static_cast<int64_t>(src_y);
            const int64_t x1 = std::min<int64_t>(x0 + 1, src_w - 1);
            const int64_t y1 = std::min<int64_t>(y0 + 1, src_h - 1);

            const float dx = src_x - static_cast<float>(x0);
            const float dy = src_y - static_cast<float>(y0);

            for (int64_t c = 0; c < dst_c; ++c) {
                const float v00 = src_ptr[img_index(c, y0, x0, src_w, src_h)];
                const float v01 = src_ptr[img_index(c, y0, x1, src_w, src_h)];
                const float v10 = src_ptr[img_index(c, y1, x0, src_w, src_h)];
                const float v11 = src_ptr[img_index(c, y1, x1, src_w, src_h)];

                const float v0 = v00 + (v01 - v00) * dx;
                const float v1 = v10 + (v11 - v10) * dx;
                const float v = v0 + (v1 - v0) * dy;

                dst_ptr[img_index(c, y, x, dst_w, dst_h)] = v;
            }
        }
    }
}

void Yolo::custom_op_postprocessing_v5su(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids) {

    const auto output_shape = src.format.shape;

    const int64_t batch_size              = output_shape[0];
    const int64_t D                       = output_shape[1];
    const int64_t num_raw_candidate_boxes = output_shape[2];
    const int64_t num_classes             = D - 4;

    // We expect something like [1,84,8400]
    COMPUTE_FLOW_ASSERT(output_shape.size() == 3);
    COMPUTE_FLOW_ASSERT(batch_size == 1);
    COMPUTE_FLOW_ASSERT(num_raw_candidate_boxes == 8400);
    COMPUTE_FLOW_ASSERT(D == 84);
    COMPUTE_FLOW_ASSERT(num_classes == 80);

    // Reserve 2048
    boxes.reserve(1024);
    scores.reserve(1024);
    class_ids.reserve(1024);

    COMPUTE_FLOW_ASSERT(boxes.size() == 0);
    COMPUTE_FLOW_ASSERT(scores.size() == 0);
    COMPUTE_FLOW_ASSERT(class_ids.size() == 0);

    float* out_ptr = reinterpret_cast<float*>(src.data);

    for (int64_t i = 0; i < num_raw_candidate_boxes; ++i) {

        // YOLO layout: cx, cy, w, h, class_prob_0..class_prob_79
        const int64_t base = i;

        // cx, cy, w, h
        // The box is defined by its center point (cx, cy) and its size (w, h).
        const float cx = out_ptr[base + 0 * num_raw_candidate_boxes];
        const float cy = out_ptr[base + 1 * num_raw_candidate_boxes];
        const float w  = out_ptr[base + 2 * num_raw_candidate_boxes];
        const float h  = out_ptr[base + 3 * num_raw_candidate_boxes];

        // find best class prob and id
        int64_t best_class = 0;
        float best_prob = out_ptr[base + 4 * num_raw_candidate_boxes];
        for (int64_t c = 1; c < num_classes; ++c) {
            float p = out_ptr[base + (4 + c) * num_raw_candidate_boxes];

            if (p > best_prob) {
                best_prob = p;
                best_class = c;
            }
        }

        // If below the confidence threshold, discard it
        if (best_prob < _confidence_thr) {
            continue;
        }

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(best_prob);
        class_ids.push_back(static_cast<int64_t>(best_class));
    }
}

void Yolo::custom_op_postprocessing_v5s(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids) {
    const auto output_shape = src.format.shape;

    const int64_t batch_size              = output_shape[0];
    const int64_t num_raw_candidate_boxes = output_shape[1];
    const int64_t D                       = output_shape[2];
    const int64_t num_classes             = D - 5;

    // We expect something like [1,25200, 85]
    COMPUTE_FLOW_ASSERT(output_shape.size() == 3);
    COMPUTE_FLOW_ASSERT(batch_size == 1);
    COMPUTE_FLOW_ASSERT(num_raw_candidate_boxes == 25200);
    COMPUTE_FLOW_ASSERT(D == 85);
    COMPUTE_FLOW_ASSERT(num_classes == 80);

    // Reserve 2048
    boxes.reserve(1024);
    scores.reserve(1024);
    class_ids.reserve(1024);

    COMPUTE_FLOW_ASSERT(boxes.size() == 0);
    COMPUTE_FLOW_ASSERT(scores.size() == 0);
    COMPUTE_FLOW_ASSERT(class_ids.size() == 0);

    float* out_ptr = reinterpret_cast<float*>(src.data);

    for (int64_t i = 0; i < num_raw_candidate_boxes; ++i) {

        // YOLO layout: cx, cy, w, h, objectness, class_prob_0..class_prob_79
        const int64_t base = i * D;

        // cx, cy, w, h
        // The box is defined by its center point (cx, cy) and its size (w, h).
        const float cx = out_ptr[base + 0];
        const float cy = out_ptr[base + 1];
        const float w  = out_ptr[base + 2];
        const float h  = out_ptr[base + 3];

        // The objectness is the probability that a predicted box actually contains any object at all,
        // regardless of its class.
        // The objectness is used to calculate the final score: final_score = objectness × best_class_probability
        const float objectness = out_ptr[base + 4];

        // find best class prob and id
        int64_t best_class = 0;
        float best_prob = out_ptr[base + 5];
        for (int64_t c = 1; c < num_classes; ++c) {
            float p = out_ptr[base + 5 + c];
            if (p > best_prob) {
                best_prob = p;
                best_class = c;
            }
        }

        // The final score
        float final_score = objectness * best_prob;

        // If below the confidence threshold, discard it
        if (final_score < _confidence_thr) {
            continue;
        }

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(final_score);
        class_ids.push_back(static_cast<int64_t>(best_class));
    }
}

void Yolo::custom_op_postprocessing_v26(const TensorData& src, std::vector<std::array<float,4>>& boxes, std::vector<float>& scores, std::vector<int64_t>& class_ids) {
    const auto output_shape = src.format.shape;

    COMPUTE_FLOW_ASSERT(output_shape.size() == 3);
    COMPUTE_FLOW_ASSERT(output_shape[0] == 1);
    COMPUTE_FLOW_ASSERT(output_shape[1] > 0);
    COMPUTE_FLOW_ASSERT(output_shape[2] == 6);

    const int64_t num_raw_candidate_boxes = output_shape[1];
    const int64_t D                       = output_shape[2];

    boxes.reserve(static_cast<size_t>(std::min<int64_t>(num_raw_candidate_boxes, 1024)));
    scores.reserve(static_cast<size_t>(std::min<int64_t>(num_raw_candidate_boxes, 1024)));
    class_ids.reserve(static_cast<size_t>(std::min<int64_t>(num_raw_candidate_boxes, 1024)));

    COMPUTE_FLOW_ASSERT(boxes.size() == 0);
    COMPUTE_FLOW_ASSERT(scores.size() == 0);
    COMPUTE_FLOW_ASSERT(class_ids.size() == 0);

    float* out_ptr = reinterpret_cast<float*>(src.data);

    for (int64_t i = 0; i < num_raw_candidate_boxes; ++i) {
        const int64_t base = i * D;

        const float x1    = out_ptr[base + 0];
        const float y1    = out_ptr[base + 1];
        const float x2    = out_ptr[base + 2];
        const float y2    = out_ptr[base + 3];
        const float score = out_ptr[base + 4];

        if (score < _confidence_thr) {
            continue;
        }

        const float class_id_value = out_ptr[base + 5];
        const int64_t class_id     = static_cast<int64_t>(std::round(class_id_value));

        COMPUTE_FLOW_ASSERT(std::fabs(class_id_value - static_cast<float>(class_id)) < 1e-3f);
        COMPUTE_FLOW_ASSERT(class_id >= 0);
        COMPUTE_FLOW_ASSERT(static_cast<size_t>(class_id) < _class_names.size());

        boxes.push_back({x1, y1, x2, y2});
        scores.push_back(score);
        class_ids.push_back(class_id);
    }
}

}  // namespace compute_flow::bricks::object_detection
