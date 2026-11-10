//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_ops/profile_label.h"
#include "runners/litert/litert_runner.h"

#include "compute_flow/check.h"
#include "compute_flow/custom_ops/registry.h"
#include "compute_flow/custom_ops/v1/kernel.h"
#include "compute_flow/manifest.h"
#include "compute_flow/profiler/profiler.h"

#include "fixtures/reference_kernels.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int32_t num_optimized_add_runs = 0;
std::atomic<int32_t> num_tracked_add_setups{0};

int64_t reject_shapes(const CF_TensorShape*, size_t, CF_TensorShape*, size_t) {
    return 1;
}

int64_t optimized_add_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    ++num_optimized_add_runs;
    return custom_op_add_run(src, dst, persistent, scratch, thread);
}

int64_t tracked_add_persistent(const CF_TensorFormatArray*, CF_MemoryRequirement* requirement) {
    requirement->size = sizeof(float);
    requirement->alignment = alignof(float);
    return 0;
}

int64_t tracked_add_setup(const CF_TensorArray* src, CF_Workspace* persistent) {
    ++num_tracked_add_setups;
    *static_cast<float*>(persistent->data) = static_cast<const float*>(src->tensors[1].data)[0];
    return 0;
}

int64_t tracked_add_scratch(
    const CF_TensorFormatArray* src,
    const CF_TensorFormatArray* dst,
    CF_MemoryRequirement* requirement) {
    if (src->count != 2 || dst->count != 1 || dst->formats[0].rank != 1) {
        return 1;
    }

    requirement->size = static_cast<size_t>(dst->formats[0].dims[0]) * sizeof(float);
    requirement->alignment = alignof(float);
    return 0;
}

int64_t tracked_add_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    if (*static_cast<const float*>(persistent->data) != static_cast<const float*>(src->tensors[1].data)[0] ||
        scratch->size < static_cast<size_t>(dst->tensors[0].format.dims[0]) * sizeof(float)) {
        return 1;
    }

    return custom_op_add_run(src, dst, persistent, scratch, thread);
}

int64_t scalar_output_shapes(
    const CF_TensorShape* inputs,
    size_t num_inputs,
    CF_TensorShape* outputs,
    size_t num_outputs) {
    if (num_inputs != 2 || num_outputs != 1 || inputs[0].rank != 1 || inputs[1].rank != 1 ||
        inputs[1].dims[0] != 1) {
        return 1;
    }

    outputs[0].rank = 1;
    outputs[0].dims[0] = inputs[0].dims[0];
    return 0;
}

int64_t tracked_scalar_run(
    const CF_TensorArray* src,
    CF_TensorArray* dst,
    const CF_Workspace* persistent,
    CF_Workspace* scratch,
    const CF_ThreadInfo* thread) {
    if (src->count != 2 || dst->count != 1 || thread->num_threads == 0 ||
        scratch->size < static_cast<size_t>(dst->tensors[0].format.dims[0]) * sizeof(float)) {
        return 1;
    }

    const float scalar = *static_cast<const float*>(persistent->data);
    if (scalar != static_cast<const float*>(src->tensors[1].data)[0]) {
        return 1;
    }

    const auto* lhs = static_cast<const float*>(src->tensors[0].data);
    auto* output = static_cast<float*>(dst->tensors[0].data);
    const int64_t size = dst->tensors[0].format.dims[0];
    for (int64_t i = static_cast<int64_t>(thread->thread_id); i < size;
         i += static_cast<int64_t>(thread->num_threads)) {
        output[i] = lhs[i] + scalar;
    }
    return 0;
}

struct FutureKernel {
    static constexpr uint64_t k_interface_version = 2;
    std::string operator_name;
    uint64_t schema_version;
    std::vector<compute_flow::custom_ops::CpuFeature> required_features;
};

int64_t failing_run(
    const CF_TensorArray*,
    CF_TensorArray*,
    const CF_Workspace*,
    CF_Workspace*,
    const CF_ThreadInfo*) {
    return 7;
}

void append_add_sub_reference_kernels(compute_flow::custom_ops::Registry& registry, bool fail_add) {
    namespace custom_ops = compute_flow::custom_ops;
    registry.add_kernel(custom_ops::v1::Kernel{
        "add_fp32",
        1,
        {},
        custom_op_add_persistent,
        custom_op_add_scratch,
        custom_op_add_setup,
        fail_add ? failing_run : custom_op_add_run,
        custom_op_binary_output_shapes,
    }, "reference");
    registry.add_kernel(custom_ops::v1::Kernel{
        "sub_fp32",
        1,
        {},
        custom_op_sub_persistent,
        custom_op_sub_scratch,
        custom_op_sub_setup,
        custom_op_sub_run,
        custom_op_binary_output_shapes,
    }, "reference");
}

compute_flow::custom_ops::Registry make_registry(
    const std::string& brick_name = "AddSub",
    bool fail_add = false) {
    compute_flow::custom_ops::Registry registry("tests:" + brick_name);
    append_add_sub_reference_kernels(registry, fail_add);
    return registry;
}

compute_flow::custom_ops::Registry make_i8_registry() {
    namespace custom_ops = compute_flow::custom_ops;
    custom_ops::Registry registry("tests:AddSubI8");
    registry.add_kernel(custom_ops::v1::Kernel{
        "add_i8",
        1,
        {},
        custom_op_add_persistent,
        custom_op_add_scratch,
        custom_op_add_setup,
        custom_op_add_i8_run,
        custom_op_binary_output_shapes,
    }, "reference");
    registry.add_kernel(custom_ops::v1::Kernel{
        "sub_i8",
        1,
        {},
        custom_op_sub_persistent,
        custom_op_sub_scratch,
        custom_op_sub_setup,
        custom_op_sub_i8_run,
        custom_op_binary_output_shapes,
    }, "reference");
    return registry;
}

compute_flow::ModelData make_model(const std::string& model_path, size_t num_threads) {
    compute_flow::ModelData model;
    model.model_name = "custom-add-sub";
    model.model_path = model_path;
    model.acc_config = compute_flow::AcceleratorConfig::cpu_only(num_threads);
    model.set_input("lhs", 0).set_input("rhs", 1);
    model.set_output("sum", 0).set_output("difference", 1);
    return model;
}

void check_pretty_profile_op(
    const std::string& report,
    const std::string& op_name,
    const std::string& kernel_name) {
    const auto op_pos = report.find("  c_op:" + op_name + " ");

    COMPUTE_FLOW_CHECK(op_pos != std::string::npos);

    const auto row_end = report.find('\n', op_pos);

    COMPUTE_FLOW_CHECK(row_end != std::string::npos);

    const auto kernel_end = report.find('\n', row_end + 1);

    COMPUTE_FLOW_CHECK(kernel_end != std::string::npos);

    const auto hw_end = report.find('\n', kernel_end + 1);

    COMPUTE_FLOW_CHECK(hw_end != std::string::npos);

    const std::string kernel_details = report.substr(row_end + 1, kernel_end - row_end - 1);
    const std::string hw_details = report.substr(kernel_end + 1, hw_end - kernel_end - 1);
    const auto kernel_pos = kernel_details.find("kernel: ");

    COMPUTE_FLOW_CHECK(kernel_pos != std::string::npos);
    COMPUTE_FLOW_CHECK(kernel_details.substr(kernel_pos) == "kernel: " + kernel_name);
    COMPUTE_FLOW_CHECK(hw_details == kernel_details.substr(0, kernel_pos) + "hw: none");
    COMPUTE_FLOW_CHECK(report.find("[kernel=") == std::string::npos);
}

void run_model(
    const std::string& model_path,
    size_t num_threads,
    const std::string& brick_name = "AddSub",
    bool enable_op_profiling = true) {
    const compute_flow::ModelData model = make_model(model_path, num_threads);

    compute_flow::LiteRTRunner runner;
    compute_flow::Profiler profiler;
    compute_flow::ProfilingOptions profiling_options;
    profiling_options.enable_op_profiling = enable_op_profiling;
    profiler.configure(profiling_options);
    runner.setup(model, {"tests", brick_name, {2, 0}}, compute_flow::ProfilerView{profiler});

    std::array<float, 2> lhs{1.0f, 2.0f};
    std::array<float, 2> rhs{3.0f, 4.0f};
    const compute_flow::TensorData lhs_tensor{{compute_flow::DataType::F32, {2}}, lhs.data(), false};
    const compute_flow::TensorData rhs_tensor{{compute_flow::DataType::F32, {2}}, rhs.data(), false};
    runner.set_input("lhs", lhs_tensor);
    runner.set_input("rhs", rhs_tensor);

    {
        auto scope = profiler.measure("run");

        runner.run();
    }

    const std::string profile_report = profiler.report_csv();
    if (enable_op_profiling) {
        const std::string add_kernel_name = brick_name == "AddSubOptimized" ? "optimized_add" : "reference";
        const std::string add_label = "c_op:add_fp32 [kernel=" + add_kernel_name + "; hw=none]";
        COMPUTE_FLOW_CHECK(profile_report.find("," + add_label + ",") != std::string::npos);
        COMPUTE_FLOW_CHECK(profile_report.find(",c_op:sub_fp32 [kernel=reference; hw=none],") != std::string::npos);
        COMPUTE_FLOW_CHECK(profile_report.find(",add_fp32,") == std::string::npos);

        for (const auto& report : {profiler.report(), profiler.report_visual(16, 50, 30, 5, true)}) {
            check_pretty_profile_op(report, "add_fp32", add_kernel_name);
            check_pretty_profile_op(report, "sub_fp32", "reference");
        }
    } else {
        COMPUTE_FLOW_CHECK(profile_report.find("[kernel=") == std::string::npos);
    }

    const auto sum = runner.get_output("sum");
    const auto difference = runner.get_output("difference");
    const auto* sum_values = static_cast<const float*>(sum.data);
    const auto* difference_values = static_cast<const float*>(difference.data);
    COMPUTE_FLOW_CHECK(sum_values[0] == 4.0f && sum_values[1] == 6.0f);
    COMPUTE_FLOW_CHECK(difference_values[0] == -2.0f && difference_values[1] == -2.0f);
}

void run_three_element_model(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 2);
    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSub", {2, 0}});

    std::array<float, 3> lhs{1.0f, 2.0f, 3.0f};
    std::array<float, 3> rhs{4.0f, 5.0f, 6.0f};
    runner.set_input("lhs", {{compute_flow::DataType::F32, {3}}, lhs.data(), false});
    runner.set_input("rhs", {{compute_flow::DataType::F32, {3}}, rhs.data(), false});
    runner.run();

    const auto sum = runner.get_output("sum");
    const auto difference = runner.get_output("difference");
    COMPUTE_FLOW_CHECK(sum.format.shape == std::vector<int64_t>{3});
    COMPUTE_FLOW_CHECK(difference.format.shape == std::vector<int64_t>{3});
    const auto* sum_values = static_cast<const float*>(sum.data);
    const auto* difference_values = static_cast<const float*>(difference.data);
    COMPUTE_FLOW_CHECK(sum_values[2] == 9.0f);
    COMPUTE_FLOW_CHECK(difference_values[2] == -3.0f);
}

void run_repeated_model(const std::string& model_path, int64_t second_size, const std::string& brick_name) {
    compute_flow::ModelData model;
    model.model_name = "custom-add-repeated";
    model.model_path = model_path;
    model.acc_config = compute_flow::AcceleratorConfig::cpu_only(2);
    model.set_input("lhs_a", 0).set_input("rhs_a", 1).set_input("lhs_b", 2).set_input("rhs_b", 3);
    model.set_output("sum_a", 0).set_output("sum_b", 1);

    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", brick_name, {2, 0}});

    std::array<float, 2> lhs_a{1.0f, 2.0f};
    std::array<float, 2> rhs_a{3.0f, 4.0f};
    std::array<float, 3> lhs_b{5.0f, 6.0f, 7.0f};
    std::array<float, 3> rhs_b{10.0f, 20.0f, 30.0f};
    runner.set_input("lhs_a", {{compute_flow::DataType::F32, {2}}, lhs_a.data(), false});
    runner.set_input("rhs_a", {{compute_flow::DataType::F32, {2}}, rhs_a.data(), false});
    runner.set_input("lhs_b", {{compute_flow::DataType::F32, {second_size}}, lhs_b.data(), false});
    runner.set_input("rhs_b", {{compute_flow::DataType::F32, {second_size}}, rhs_b.data(), false});
    runner.run();

    const auto sum_a = runner.get_output("sum_a");
    const auto sum_b = runner.get_output("sum_b");
    const auto* values_a = static_cast<const float*>(sum_a.data);
    const auto* values_b = static_cast<const float*>(sum_b.data);
    COMPUTE_FLOW_CHECK(sum_a.format.shape == std::vector<int64_t>{2});
    COMPUTE_FLOW_CHECK(sum_b.format.shape == std::vector<int64_t>{second_size});
    COMPUTE_FLOW_CHECK(values_a[0] == 4.0f && values_a[1] == 6.0f);
    COMPUTE_FLOW_CHECK(values_b[0] == 15.0f && values_b[1] == 26.0f);

    if (second_size == 3) {
        COMPUTE_FLOW_CHECK(values_b[2] == 37.0f);
    }
}

void check_persistent_reuse(const std::string& model_path) {
    compute_flow::ModelData model;
    model.model_name = "custom-add-repeated";
    model.model_path = model_path;
    model.acc_config = compute_flow::AcceleratorConfig::cpu_only(2);
    model.set_input("lhs_a", 0).set_input("rhs_a", 1).set_input("lhs_b", 2).set_input("rhs_b", 3);
    model.set_output("sum_a", 0).set_output("sum_b", 1);

    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubRepeated", {2, 0}});

    std::array<float, 2> lhs_a{1.0f, 2.0f};
    std::array<float, 2> rhs_a{3.0f, 4.0f};
    std::array<float, 2> lhs_b{5.0f, 6.0f};
    std::array<float, 2> rhs_b{10.0f, 20.0f};
    runner.set_input("lhs_a", {{compute_flow::DataType::F32, {2}}, lhs_a.data(), false});
    runner.set_input("rhs_a", {{compute_flow::DataType::F32, {2}}, rhs_a.data(), false});
    runner.set_input("lhs_b", {{compute_flow::DataType::F32, {2}}, lhs_b.data(), false});
    runner.set_input("rhs_b", {{compute_flow::DataType::F32, {2}}, rhs_b.data(), false});

    const int32_t before = num_tracked_add_setups.load();
    runner.run();
    COMPUTE_FLOW_CHECK(num_tracked_add_setups.load() == before + 2);

    lhs_a = {7.0f, 8.0f};
    runner.set_input("lhs_a", {{compute_flow::DataType::F32, {2}}, lhs_a.data(), false});
    runner.run();
    COMPUTE_FLOW_CHECK(num_tracked_add_setups.load() == before + 2);
    const auto sum_a = runner.get_output("sum_a");
    const auto* values_a = static_cast<const float*>(sum_a.data);
    COMPUTE_FLOW_CHECK(values_a[0] == 10.0f && values_a[1] == 12.0f);
}

void check_setup_reused_across_data_shapes(const std::string& model_path) {
    compute_flow::ModelData model;
    model.model_name = "custom-add-shared-constant";
    model.model_path = model_path;
    model.acc_config = compute_flow::AcceleratorConfig::cpu_only(2);
    model.set_input("lhs_a", 0).set_input("lhs_b", 1).set_input("rhs", 2);
    model.set_output("sum_a", 0).set_output("sum_b", 1);

    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubRepeated", {2, 0}});

    std::array<float, 2> lhs_a{1.0f, 2.0f};
    std::array<float, 3> lhs_b{3.0f, 4.0f, 5.0f};
    std::array<float, 1> rhs{10.0f};
    runner.set_input("lhs_a", {{compute_flow::DataType::F32, {2}}, lhs_a.data(), false});
    runner.set_input("lhs_b", {{compute_flow::DataType::F32, {3}}, lhs_b.data(), false});
    runner.set_input("rhs", {{compute_flow::DataType::F32, {1}}, rhs.data(), false});

    const int32_t before = num_tracked_add_setups.load();
    runner.run();
    COMPUTE_FLOW_CHECK(num_tracked_add_setups.load() == before + 1);
    const auto sum_a = runner.get_output("sum_a");
    const auto sum_b = runner.get_output("sum_b");
    const auto* values_a = static_cast<const float*>(sum_a.data);
    const auto* values_b = static_cast<const float*>(sum_b.data);
    COMPUTE_FLOW_CHECK(values_a[0] == 11.0f && values_a[1] == 12.0f);
    COMPUTE_FLOW_CHECK(values_b[0] == 13.0f && values_b[1] == 14.0f && values_b[2] == 15.0f);

    lhs_a = {6.0f, 7.0f};
    runner.set_input("lhs_a", {{compute_flow::DataType::F32, {2}}, lhs_a.data(), false});
    runner.run();
    COMPUTE_FLOW_CHECK(num_tracked_add_setups.load() == before + 1);
    const auto updated_sum_a = runner.get_output("sum_a");
    const auto* updated_values_a = static_cast<const float*>(updated_sum_a.data);
    COMPUTE_FLOW_CHECK(updated_values_a[0] == 16.0f && updated_values_a[1] == 17.0f);
}

void run_subgraph_model(const std::string& model_path) {
    compute_flow::ModelData model;
    model.model_name = "custom-add-subgraphs";
    model.model_path = model_path;
    model.acc_config = compute_flow::AcceleratorConfig::cpu_only(2);
    model.set_input("lhs", 0).set_input("rhs", 1);
    model.set_output("sum", 0);

    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubRepeated", {2, 0}});

    std::array<float, 2> lhs{1.0f, 2.0f};
    std::array<float, 2> rhs{3.0f, 4.0f};
    runner.set_input("lhs", {{compute_flow::DataType::F32, {2}}, lhs.data(), false});
    runner.set_input("rhs", {{compute_flow::DataType::F32, {2}}, rhs.data(), false});
    runner.run();

    const auto sum = runner.get_output("sum");
    const auto* values = static_cast<const float*>(sum.data);
    COMPUTE_FLOW_CHECK(values[0] == 4.0f && values[1] == 6.0f);
}

void run_i8_model(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 2);
    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubI8", {1, 0}});

    std::array<int8_t, 2> lhs{1, 2};
    std::array<int8_t, 2> rhs{3, 4};
    const compute_flow::TensorData lhs_tensor{{compute_flow::DataType::I8, {2}}, lhs.data(), false};
    const compute_flow::TensorData rhs_tensor{{compute_flow::DataType::I8, {2}}, rhs.data(), false};
    runner.set_input("lhs", lhs_tensor);
    runner.set_input("rhs", rhs_tensor);
    runner.run();

    const auto sum = runner.get_output("sum");
    const auto difference = runner.get_output("difference");
    COMPUTE_FLOW_CHECK(sum.format.dt == compute_flow::DataType::I8);
    COMPUTE_FLOW_CHECK(difference.format.dt == compute_flow::DataType::I8);
    const auto* sum_values = static_cast<const int8_t*>(sum.data);
    const auto* difference_values = static_cast<const int8_t*>(difference.data);
    COMPUTE_FLOW_CHECK(sum_values[0] == 4 && sum_values[1] == 6);
    COMPUTE_FLOW_CHECK(difference_values[0] == -2 && difference_values[1] == -2);
}

void check_missing_registry(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 1);
    compute_flow::LiteRTRunner runner;
    bool reported = false;

    try {
        runner.setup(model, {"tests", "Missing", {1, 0}});
    } catch (const std::invalid_argument& error) {
        reported = std::string(error.what()).find("Missing compiled registry") != std::string::npos;
    }

    COMPUTE_FLOW_CHECK(reported);
}

void check_unsupported_interface(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 2);
    compute_flow::LiteRTRunner runner;
    bool reported = false;

    try {
        runner.setup(model, {"tests", "AddSubFuture", {1, 0}});
    } catch (const std::invalid_argument& error) {
        reported = std::string(error.what()).find("Unsupported LiteRT custom operator interface version: 2") !=
            std::string::npos;
    }

    COMPUTE_FLOW_CHECK(reported);
}

void check_kernel_failure(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 2);
    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubFail", {1, 0}});

    std::array<float, 2> lhs{1.0f, 2.0f};
    std::array<float, 2> rhs{3.0f, 4.0f};
    const compute_flow::TensorData lhs_tensor{{compute_flow::DataType::F32, {2}}, lhs.data(), false};
    const compute_flow::TensorData rhs_tensor{{compute_flow::DataType::F32, {2}}, rhs.data(), false};
    runner.set_input("lhs", lhs_tensor);
    runner.set_input("rhs", rhs_tensor);

    bool reported = false;
    try {
        runner.run();
    } catch (const std::runtime_error& error) {
        reported = std::string(error.what()).find("LiteRT custom operator") != std::string::npos;
    }

    COMPUTE_FLOW_CHECK(reported);
}

void check_shape_failure(const std::string& model_path) {
    const compute_flow::ModelData model = make_model(model_path, 1);
    compute_flow::LiteRTRunner runner;
    runner.setup(model, {"tests", "AddSubBadShapes", {2, 0}});

    std::array<float, 2> lhs{1.0f, 2.0f};
    std::array<float, 2> rhs{3.0f, 4.0f};
    runner.set_input("lhs", {{compute_flow::DataType::F32, {2}}, lhs.data(), false});
    runner.set_input("rhs", {{compute_flow::DataType::F32, {2}}, rhs.data(), false});

    bool reported = false;
    try {
        runner.run();
    } catch (const std::runtime_error& error) {
        reported = std::string(error.what()).find("LiteRT custom operator") != std::string::npos;
    }

    COMPUTE_FLOW_CHECK(reported);
}

void check_optimized_profile_label() {
    namespace custom_ops = compute_flow::custom_ops;
    const std::vector<custom_ops::CpuFeature> required_features{
        custom_ops::HardwareFeature::Arm::NeonI8mm,
        custom_ops::HardwareFeature::Arm::NeonFp16,
    };

    COMPUTE_FLOW_CHECK(
        custom_ops::get_custom_op_profile_suffix("arm/add_neon", required_features) ==
        " [kernel=arm/add_neon; hw=arm_neon_i8mm+arm_neon_fp16]");
}

}  // namespace

namespace compute_flow::custom_ops {

const Registry& test_add_sub_registry() {
    static const Registry k_registry = make_registry();
    return k_registry;
}

const Registry& test_add_sub_i8_registry() {
    static const Registry k_registry = make_i8_registry();
    return k_registry;
}

const Registry& test_add_sub_optimized_registry() {
    static const Registry k_registry = [] {
        Registry registry("tests:AddSubOptimized");
        registry.add_kernel(v1::Kernel{
            "add_fp32",
            1,
            {},
            custom_op_add_persistent,
            custom_op_add_scratch,
            custom_op_add_setup,
            optimized_add_run,
            custom_op_binary_output_shapes,
        }, "optimized_add");
        append_add_sub_reference_kernels(registry, false);
        return registry;
    }();
    return k_registry;
}

const Registry& test_add_sub_repeated_registry() {
    static const Registry k_registry = [] {
        Registry registry("tests:AddSubRepeated");

        registry.add_kernel(v1::Kernel{
            "add_fp32",
            1,
            {},
            tracked_add_persistent,
            tracked_add_scratch,
            tracked_add_setup,
            tracked_add_run,
            custom_op_binary_output_shapes,
            {1},
        }, "reference");

        registry.add_kernel(v1::Kernel{
            "add_scalar_fp32",
            1,
            {},
            tracked_add_persistent,
            tracked_add_scratch,
            tracked_add_setup,
            tracked_scalar_run,
            scalar_output_shapes,
            {1},
        }, "reference");

        return registry;
    }();
    return k_registry;
}

const Registry& test_add_sub_bad_shapes_registry() {
    static const Registry k_registry = [] {
        Registry registry("tests:AddSubBadShapes");
        registry.add_kernel(v1::Kernel{
            "add_fp32",
            1,
            {},
            custom_op_add_persistent,
            custom_op_add_scratch,
            custom_op_add_setup,
            custom_op_add_run,
            reject_shapes,
        }, "bad_shapes");
        append_add_sub_reference_kernels(registry, false);
        return registry;
    }();
    return k_registry;
}

const Registry& test_add_sub_future_registry() {
    static const Registry k_registry = [] {
        Registry registry("tests:AddSubFuture");
        registry.add_kernel(FutureKernel{"add_fp32", 1, {}});
        append_add_sub_reference_kernels(registry, false);
        return registry;
    }();
    return k_registry;
}

const Registry& test_add_sub_fail_registry() {
    static const Registry k_registry = make_registry("AddSubFail", true);
    return k_registry;
}

}  // namespace compute_flow::custom_ops

int main(int argc, char** argv) {
    if (argc != 8) {
        return 1;
    }

    run_model(argv[1], 1);
    run_model(argv[1], 2);
    run_model(argv[1], 1, "AddSubOptimized");
    COMPUTE_FLOW_CHECK(num_optimized_add_runs == 1);
    run_model(argv[1], 1, "AddSub", false);
    run_i8_model(argv[2]);
    run_three_element_model(argv[6]);
    run_repeated_model(argv[3], 2, "AddSubRepeated");
    check_persistent_reuse(argv[3]);
    check_setup_reused_across_data_shapes(argv[7]);
    run_repeated_model(argv[4], 3, "AddSubRepeated");
    run_subgraph_model(argv[5]);

    auto first = std::async(std::launch::async, run_repeated_model, argv[4], 3, "AddSubRepeated");
    auto second = std::async(std::launch::async, run_repeated_model, argv[4], 3, "AddSubRepeated");
    first.get();
    second.get();

    check_missing_registry(argv[1]);
    check_unsupported_interface(argv[1]);
    check_kernel_failure(argv[1]);
    check_shape_failure(argv[1]);
    check_optimized_profile_label();
    return 0;
}
