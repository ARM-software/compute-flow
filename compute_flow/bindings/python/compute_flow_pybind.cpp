//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "compute_flow/accelerator.h"  // AcceleratorConfig
#include "compute_flow/bricks/brick_interface_v1.h"
#include "compute_flow/bricks/brick_manifest.h"
#include "compute_flow/cli_examples/internal/manifest_selector.h"
#include "compute_flow/manifest.h"
#include "compute_flow/model_data.h"   // <- ensure this header exists
#include "compute_flow/tensor_dict.h"  // TensorDict, TensorData (assumed API)

namespace py = pybind11;

namespace compute_flow {

struct PyKeepAlive {
    std::vector<py::object> objs;
    void add(py::object o) { objs.emplace_back(std::move(o)); }
};

struct PyTensorDictState : TensorDictExternalState {
    PyKeepAlive keep;
};

static std::shared_ptr<PyTensorDictState> ensure_state(TensorDict& td) {
    if (auto existing = std::dynamic_pointer_cast<PyTensorDictState>(td.get_external_state())) {
        return existing;
    }
    auto state = std::make_shared<PyTensorDictState>();
    td.set_external_state(state);
    return state;
}

// -----------------------------
// Helpers: NumPy <-> TensorData
// -----------------------------
// NOTE: These helpers make a couple of *documented* assumptions about your
// TensorData/TensorDict API.
//
// Assumptions (adjust if different in your codebase):
//   - TensorData supports:
//       * std::vector<int64_t> shape() const;
//       * size_t nbytes() const;        // total byte size
//       * void* data();                 // raw pointer
//       * const void* data() const;
//       * enum DType { F32, I32, I64, U8, F16, BF16 } data_type() const;
//   - There is a way to construct a TensorData view from an external buffer:
//       * TensorData MakeView(void* data, size_t nbytes, const std::vector<int64_t>& shape, DType dt);
//     If your real API differs, replace the MakeView implementation below to match.
//   - TensorDict supports:
//       * std::vector<std::string> keys() const;
//       * const TensorData& at(const std::string& name) const;  // throws if missing
//       * void set(const std::string& name, const TensorData& td);
//     If your API differs (e.g., operator[]/emplace), tweak the adapters accordingly.

// Map NumPy dtype to your engine's dtype.
static inline DataType dtype_from_numpy(const py::array& arr) {
    if (py::isinstance<py::array_t<float>>(arr))   return DataType::F32;
    if (py::isinstance<py::array_t<int32_t>>(arr)) return DataType::I32;
    if (py::isinstance<py::array_t<int64_t>>(arr)) return DataType::I64;
    if (py::isinstance<py::array_t<uint64_t>>(arr))return DataType::U64;
    if (py::isinstance<py::array_t<uint8_t>>(arr)) return DataType::U8;
    throw std::invalid_argument("Unsupported NumPy dtype for TensorData");
}

static inline TensorData tensor_from_numpy(py::array arr, PyKeepAlive& keep) {
    // Ensure C-contiguous and keep the produced array alive
    if ((arr.flags() & py::array::c_style) == 0) {
        py::array ensured = py::array::ensure(arr, py::array::c_style);
        if (!ensured) throw std::invalid_argument("Expected a C-contiguous NumPy array (copy failed)");
        arr = ensured;
        keep.add(arr); // keep the ensured array alive
    } else {
        keep.add(arr); // keep the original array alive as well
    }

    // Shape
    std::vector<int64_t> shape(arr.ndim());
    for (ssize_t i = 0; i < arr.ndim(); ++i) shape[i] = static_cast<int64_t>(arr.shape(i));

    // Dtype + raw pointer
    const auto dt = dtype_from_numpy(arr);
    void* ptr = const_cast<void*>(arr.request().ptr);

    TensorData t_out;
    t_out.data = ptr;
    t_out.user_hosted = true;     // engine must not free Python memory
    t_out.format.dt = dt;
    t_out.format.shape = shape;
    return t_out;
}

// Convert TensorData -> NumPy (zero-copy view where possible)
static inline py::array numpy_from_tensor(const TensorData& td) {
    if (td.format.dt == DataType::String) {
        const auto& shape = td.format.shape;

        // Compute number of bytes
        ssize_t num_bytes = 1;
        for (ssize_t d : shape) {
            num_bytes *= d;
        }

        const char* data = static_cast<const char*>(td.data);

        // Return a Python str (attempt UTF-8 decode)
        // If decoding fails, fall back to replacement.
        try {
            return py::str(data, num_bytes);
        } catch (const std::exception&) {
            return py::str(py::bytes(data, num_bytes));   // decode in Python
        }
    }

    // Map dtype
    py::dtype np_dt; // default
    switch (td.format.dt) {
        case DataType::F32:  np_dt = py::dtype::of<float>();   break;
        case DataType::I32:  np_dt = py::dtype::of<int32_t>(); break;
        case DataType::I64:  np_dt = py::dtype::of<int64_t>(); break;
        case DataType::U8:   np_dt = py::dtype::of<uint8_t>(); break;
        default: throw std::invalid_argument("Unsupported TensorData dtype -> NumPy");
    }

    // Build shape/strides assuming compact row-major layout.
    const auto& shape = td.format.shape;
    std::vector<ssize_t> np_shape(shape.begin(), shape.end());

    // Compute compact strides from dtype itemsize.
    const ssize_t itemsize = np_dt.itemsize();
    std::vector<ssize_t> strides(np_shape.size());
    if (!np_shape.empty()) {
        strides.back() = itemsize;
        for (int i = static_cast<int>(np_shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * np_shape[i + 1];
        }
    }

    // Create a Python array that references the existing memory.
    // Keep a capsule to avoid double-free (engine owns memory).
    void* ptr = const_cast<void*>(td.data);
    py::capsule owner(ptr, [](void*) {/* no-op: memory owned by engine */});

    return py::array(np_dt, np_shape, strides, ptr, owner);
}

// Python dict[str, np.ndarray|str] -> TensorDict
static inline TensorDict tensordict_from_pydict(const py::dict& d, PyKeepAlive& keep) {
    TensorDict td;
    for (auto item : d) {
        const std::string name = py::cast<std::string>(item.first);

        if (py::isinstance<py::array>(item.second)) {
            py::array np = py::cast<py::array>(item.second);
            td.declare_tensor(name);
            TensorData t = tensor_from_numpy(std::move(np), keep);
            td[name] = std::move(t);
            continue;
        }

        if (py::isinstance<py::str>(item.second)) {
            // Materialize as Python bytes so we can safely expose a buffer
            py::bytes b = py::bytes(py::cast<py::str>(item.second));
            keep.add(b);  // ensure lifetime through run()

            // Extract raw buffer + size
            char* buf = nullptr;
            ssize_t size = 0;
            PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buf, &size); // safe for Python bytes

            td.declare_tensor(name);
            TensorData t;
            t.data = buf;
            t.user_hosted = true;        // engine must not free
            t.format.dt = DataType::String;
            // Choose the shape your engine expects for strings. If it expects a flat char vector:
            t.format.shape = { static_cast<int64_t>(size) };  // or { size, 1 } if that's your convention
            td[name] = std::move(t);
            continue;
        }

        throw std::invalid_argument("Unsupported Python value type for key: " + name);
    }
    return td;
}

// TensorDict -> Python dict[str, np.ndarray]
static inline py::dict pydict_from_tensordict(const TensorDict& td) {
    py::dict out;
    for (const auto& name : td.keys()) {
        out[py::str(name)] = numpy_from_tensor(td.at(name));
    }
    return out;
}

static py::object constant_to_py(const ConstantValue& v) {
    return std::visit([](const auto& value) -> py::object {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, bool>) {
            return py::int_(value);
        } else if constexpr (std::is_same_v<T, float> ||
                             std::is_same_v<T, double>) {
            return py::float_(value);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return py::str(value);
        } else {
            // Fallback: expose as string representation
            return py::str("<unsupported ConstantValue type>");
        }
    }, v);
}

static void set_config_from_py(
    ComputeFlowManifest& manifest,
    const std::string& key,
    const py::object& value) {
    const ConfigValue* current = manifest.get_config_ptr(key);

    if (py::isinstance<py::bool_>(value)) {
        manifest.set_config(key, py::cast<bool>(value));
    } else if (py::isinstance<py::int_>(value)) {
        if (current != nullptr && std::holds_alternative<int32_t>(*current)) {
            manifest.set_config(key, py::cast<int32_t>(value));
        } else {
            manifest.set_config(key, py::cast<int64_t>(value));
        }
    } else if (py::isinstance<py::float_>(value)) {
        if (current != nullptr && std::holds_alternative<float>(*current)) {
            manifest.set_config(key, py::cast<float>(value));
        } else {
            manifest.set_config(key, py::cast<double>(value));
        }
    } else if (py::isinstance<py::str>(value)) {
        manifest.set_config(key, py::cast<std::string>(value));
    } else {
        throw std::invalid_argument("Unsupported Python value type for config: " + key);
    }
}

void bind_compute_flow_py(py::module_& m) {
    m.doc() = "Python bindings for Compute Flow Core (pybind11)";

    m.def("load_manifest",
    [](const std::string& manifest_path) -> ComputeFlowManifest {
        return bricks::load_manifest(manifest_path);
    },
    py::arg("manifest_path"),
    "Load a ComputeFlowManifest by reading the JSON manifest at manifest_path.");

    m.def("get_manifest_path",
          &bricks::internal::get_manifest_path,
          py::arg("manifest"),
          py::arg("brick_type"),
          "Resolve a manifest path or typed @manifest selector to a manifest JSON path.");

    // ---- Accelerator API --------------------------------------------------------
    py::enum_<AcceleratorType>(m, "AcceleratorType")
        .value("Cpu", AcceleratorType::Cpu)
        .value("Gpu", AcceleratorType::Gpu)
        .export_values();

    py::class_<CpuOptions>(m, "CpuOptions")
        .def(py::init<>())
        .def_readwrite("num_threads", &CpuOptions::num_threads)
        .def_readwrite("enable_fp16", &CpuOptions::enable_fp16);

    py::class_<GpuOptions>(m, "GpuOptions")
        .def(py::init<>())
        .def_readwrite("device_index", &GpuOptions::device_index);

    // You don’t need to expose the std::variant to mirror your C++ usage,
    // but it's harmless to expose the preference struct:
    py::class_<AcceleratorPreference>(m, "AcceleratorPreference")
        .def(py::init<AcceleratorType, const AcceleratorParams&>(),
            py::arg("type"), py::arg("params"))
        .def_readwrite("type", &AcceleratorPreference::type)
        .def_readwrite("params", &AcceleratorPreference::params);

    // Bind the config; expose the static cpu_only(num_threads, enable_fp16)
    py::class_<AcceleratorConfig>(m, "AcceleratorConfig")
        .def(py::init<>())
        .def_readwrite("priorities", &AcceleratorConfig::priorities)
        .def_static("cpu_only",
            [](size_t num_threads, bool enable_fp16) {
                return AcceleratorConfig::cpu_only(num_threads, enable_fp16);
            },
            py::arg("num_threads") = 0, py::arg("enable_fp16") = false)
        .def("has_accelerator", &AcceleratorConfig::has_accelerator, py::arg("t"))
        .def("is_accelerator_first_priority", &AcceleratorConfig::is_accelerator_first_priority, py::arg("t"));

    py::class_<WeightsCachingOptions>(m, "WeightsCachingOptions")
        .def(py::init<>())
        .def_readwrite("enabled", &WeightsCachingOptions::enabled);

    py::class_<RuntimeOptions>(m, "RuntimeOptions")
        .def(py::init<>())
        .def_readwrite("enable_warmup", &RuntimeOptions::enable_warmup)
        .def_readwrite("weights_caching", &RuntimeOptions::weights_caching);

    py::enum_<ProfilingFormat>(m, "ProfilingFormat")
        .value("Pretty", ProfilingFormat::Pretty)
        .value("Csv", ProfilingFormat::Csv)
        .export_values();

    py::enum_<ProfilingUnit>(m, "ProfilingUnit")
        .value("Seconds", ProfilingUnit::Seconds)
        .value("Milliseconds", ProfilingUnit::Milliseconds)
        .value("Microseconds", ProfilingUnit::Microseconds)
        .value("Nanoseconds", ProfilingUnit::Nanoseconds)
        .export_values();

    py::class_<ProfilingOptions>(m, "ProfilingOptions")
        .def(py::init<>())
        .def_readwrite("enabled", &ProfilingOptions::enabled)
        .def_readwrite("enable_op_profiling", &ProfilingOptions::enable_op_profiling)
        .def_readwrite("enable_timeline_annotations", &ProfilingOptions::enable_timeline_annotations)
        .def_readwrite("format", &ProfilingOptions::format)
        .def_readwrite("unit", &ProfilingOptions::unit);

    py::class_<bricks::BrickOptions>(m, "BrickOptions")
        .def(py::init<>())
        .def_readwrite("runtime", &bricks::BrickOptions::runtime)
        .def_readwrite("profiling", &bricks::BrickOptions::profiling);

    py::class_<ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("ok", &ValidationResult::ok)
        .def_readwrite("error", &ValidationResult::error)
        .def("__bool__", [](const ValidationResult& result) {
            return static_cast<bool>(result);
        });

    py::class_<ModelData>(m, "ModelData")
        .def_readwrite("acc_config", &ModelData::acc_config)
        .def_readonly("model_name", &ModelData::model_name, "Logical model name from the manifest.");

    py::class_<bricks::BrickVersion>(m, "BrickVersion")
        .def(py::init<>())
        .def_readwrite("major", &bricks::BrickVersion::major)
        .def_readwrite("minor", &bricks::BrickVersion::minor);

    py::class_<BrickInfo>(m, "BrickInfo")
        .def(py::init<>())
        .def_readwrite("type", &BrickInfo::type)
        .def_readwrite("name", &BrickInfo::name)
        .def_readwrite("version", &BrickInfo::version);

    // ---- ComputeFlowManifest binding ----------------------------------------
    py::class_<ComputeFlowManifest>(m, "ComputeFlowManifest")

        // module accessors
        .def("declare_module",
             &ComputeFlowManifest::declare_module,
             py::arg("key"),
             py::return_value_policy::reference)
        .def("set_brick_info",
             &ComputeFlowManifest::set_brick_info,
             py::arg("brick_info"),
             py::return_value_policy::reference_internal,
             "Set the brick type/name metadata for this manifest.")
        .def("get_brick_info",
             &ComputeFlowManifest::get_brick_info,
             py::return_value_policy::reference_internal,
             "Get the brick type/name metadata for this manifest.")

        // __getitem__ -> operator[] (non-const)
        .def("__getitem__",
             (ModelData& (ComputeFlowManifest::*)(std::string_view))
                 &ComputeFlowManifest::operator[],
             py::arg("key"),
             py::return_value_policy::reference)

        // at(key) non-const
        .def("at",
             (ModelData& (ComputeFlowManifest::*)(std::string_view))
                 &ComputeFlowManifest::at,
             py::arg("key"),
             py::return_value_policy::reference)

        // const overloads if you need them in Python (rare)
        // .def("at_const",
        //      (const ModelData& (ComputeFlowManifest::*)(std::string_view) const)
        //          &ComputeFlowManifest::at,
        //      py::arg("key"),
        //      py::return_value_policy::reference)

        // info methods
        .def("num_modules", &ComputeFlowManifest::num_modules)
        .def("models",
             &ComputeFlowManifest::models,
             py::return_value_policy::reference_internal,
             "Return the models declared in manifest order.")
        .def("has_module",  &ComputeFlowManifest::has_module, py::arg("key"))
        .def("has_constant",    &ComputeFlowManifest::has_constant,   py::arg("key"))
        .def("has_config",  &ComputeFlowManifest::has_config, py::arg("key"))
        .def("has_asset",   &ComputeFlowManifest::has_asset,  py::arg("key"))

        .def("get_constant",
            [](ComputeFlowManifest& self, std::string_view key) -> py::object {
                if (auto* mv = self.get_constant_ptr(key)) {
                    return constant_to_py(*mv);   // convert ConstantValue -> py::object
                }
                return py::none();  // key not present
            },
            py::arg("key"))

        .def("get_config",
            [](const ComputeFlowManifest& self, std::string_view key) -> py::object {
                if (!self.has_config(key)) {
                    return py::none();
                }
                return constant_to_py(self.get_config_value(key));
            },
            py::arg("key"))

        .def("set_config",
            [](ComputeFlowManifest& self, std::string key, py::object value) -> ComputeFlowManifest& {
                set_config_from_py(self, key, value);
                return self;
            },
            py::arg("key"), py::arg("value"),
            py::return_value_policy::reference_internal)

        .def("set_asset",
            [](ComputeFlowManifest& self, std::string key, std::string value) -> ComputeFlowManifest& {
                self.set_asset(std::move(key), std::move(value));
                return self;
            },
            py::arg("key"), py::arg("value"),
            py::return_value_policy::reference_internal)

        .def("get_asset",
            [](ComputeFlowManifest& self, std::string_view key) -> py::object {
                if (auto* p = self.get_asset(key)) {
                    return py::str(*p);
                }
                return py::none();
            },
            py::arg("key"))

        .def("get_asset_const",
            [](const ComputeFlowManifest& self, std::string_view key) -> py::object {
                if (const auto* p = self.get_asset(key)) {
                    return py::str(*p);
                }
                return py::none();
            },
            py::arg("key"));


    py::class_<TensorDict>(m, "TensorDict")
        .def(py::init([]() {
            TensorDict td;
            td.set_external_state(std::make_shared<PyTensorDictState>());
            return td;
        }))

        // Return None instead of TensorData&, to avoid exposing TensorData
        .def("declare_tensor",
            [](TensorDict& t, const std::string& name) {
                t.declare_tensor(name);   // ignore the returned TensorData&
            },
            py::arg("name"),
            "Declare a tensor entry without assigning data yet.")

        .def("keys", &TensorDict::keys)

        .def("__getitem__", [](const TensorDict& t, const std::string& k) {
            return numpy_from_tensor(t.at(k));   // still fine
        })

        .def("__setitem__", [](TensorDict& t, const std::string& k, py::object v) {
            auto state = ensure_state(t);
            if (py::isinstance<py::array>(v)) {
                py::array np = py::cast<py::array>(v);
                t.declare_tensor(k);
                t[k] = tensor_from_numpy(np, state->keep);
            } else if (py::isinstance<py::str>(v)) {
                py::bytes b = py::bytes(py::cast<py::str>(v));
                state->keep.add(b);
                char* buf = nullptr; ssize_t size = 0;
                PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buf, &size);

                t.declare_tensor(k);
                TensorData td;
                td.data = buf;
                td.user_hosted = true;
                td.format.dt = DataType::String;
                td.format.shape = { static_cast<int64_t>(size) }; // or {size,1} if needed
                t[k] = std::move(td);
            } else {
                throw std::invalid_argument("Unsupported value type for TensorDict item '" + k + "'");
            }
        })

        .def("to_dict", &pydict_from_tensordict);
}

}  // namespace compute_flow

PYBIND11_MODULE(_compute_flow_py, m) {
    compute_flow::bind_compute_flow_py(m);
}
