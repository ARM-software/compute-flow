//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "compute_flow/bricks/brick_interface_v1.h"
#include "compute_flow/manifest.h"
#include "compute_flow/tensor_dict.h"

namespace compute_flow::pybind {

namespace py = pybind11;

struct PyKeepAlive {
    std::vector<py::object> objs;
    void add(py::object o) { objs.emplace_back(std::move(o)); }
};

static inline DataType dtype_from_numpy(const py::array& arr) {
    if (py::isinstance<py::array_t<float>>(arr)) {
        return DataType::F32;
    }
    if (py::isinstance<py::array_t<int32_t>>(arr)) {
        return DataType::I32;
    }
    if (py::isinstance<py::array_t<int64_t>>(arr)) {
        return DataType::I64;
    }
    if (py::isinstance<py::array_t<uint64_t>>(arr)) {
        return DataType::U64;
    }
    if (py::isinstance<py::array_t<uint8_t>>(arr)) {
        return DataType::U8;
    }
    throw std::invalid_argument("Unsupported NumPy dtype for TensorData");
}

static inline TensorData tensor_from_numpy(py::array arr, PyKeepAlive& keep) {
    if ((arr.flags() & py::array::c_style) == 0) {
        py::array ensured = py::array::ensure(arr, py::array::c_style);
        if (!ensured) {
            throw std::invalid_argument("Expected a C-contiguous NumPy array (copy failed)");
        }
        arr = ensured;
        keep.add(arr);
    } else {
        keep.add(arr);
    }

    std::vector<int64_t> shape(arr.ndim());
    for (ssize_t i = 0; i < arr.ndim(); ++i) {
        shape[i] = static_cast<int64_t>(arr.shape(i));
    }

    const auto dt = dtype_from_numpy(arr);
    void* ptr = const_cast<void*>(arr.request().ptr);

    TensorData t_out;
    t_out.data = ptr;
    t_out.user_hosted = true;
    t_out.format.dt = dt;
    t_out.format.shape = shape;
    return t_out;
}

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

    py::dtype np_dt;
    switch (td.format.dt) {
        case DataType::F32:
            np_dt = py::dtype::of<float>();
            break;
        case DataType::I32:
            np_dt = py::dtype::of<int32_t>();
            break;
        case DataType::I64:
            np_dt = py::dtype::of<int64_t>();
            break;
        case DataType::U64:
            np_dt = py::dtype::of<uint64_t>();
            break;
        case DataType::U8:
            np_dt = py::dtype::of<uint8_t>();
            break;
        default:
            throw std::invalid_argument("Unsupported TensorData dtype -> NumPy");
    }

    const auto& shape = td.format.shape;
    std::vector<ssize_t> np_shape(shape.begin(), shape.end());

    const ssize_t itemsize = np_dt.itemsize();
    std::vector<ssize_t> strides(np_shape.size());
    if (!np_shape.empty()) {
        strides.back() = itemsize;
        for (int i = static_cast<int>(np_shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * np_shape[i + 1];
        }
    }

    void* ptr = const_cast<void*>(td.data);
    py::capsule owner(ptr, [](void*) {});

    return py::array(np_dt, np_shape, strides, ptr, owner);
}

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
            py::bytes b = py::bytes(py::cast<py::str>(item.second));
            keep.add(b);

            char* buf = nullptr;
            ssize_t size = 0;
            PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buf, &size);

            td.declare_tensor(name);
            TensorData t;
            t.data = buf;
            t.user_hosted = true;
            t.format.dt = DataType::String;
            t.format.shape = { static_cast<int64_t>(size) };
            td[name] = std::move(t);
            continue;
        }

        throw std::invalid_argument("Unsupported Python value type for key: " + name);
    }
    return td;
}

static inline py::dict pydict_from_tensordict(const TensorDict& td) {
    py::dict out;
    for (const auto& name : td.keys()) {
        out[py::str(name)] = numpy_from_tensor(td.at(name));
    }
    return out;
}

template <typename BrickT>
class PyBrick {
public:
    PyBrick() = default;

    ValidationResult validate(const ComputeFlowManifest& manifest) {
        return static_cast<const bricks::IBrickV1&>(impl_).validate(manifest);
    }

    void setup(const ComputeFlowManifest& manifest, const bricks::BrickOptions& options = {}) {
        static_cast<bricks::IBrickV1&>(impl_).setup(manifest, options);
    }

    py::object run(py::object input) {
        PyKeepAlive keep;
        TensorDict in_td;

        if (py::isinstance<py::dict>(input)) {
            in_td = tensordict_from_pydict(py::cast<py::dict>(input), keep);
        } else if (py::isinstance<TensorDict>(input)) {
            in_td = py::cast<TensorDict>(input);
        } else {
            throw std::invalid_argument("run expects dict[str, np.ndarray|str] or TensorDict");
        }

        TensorDict out_td = impl_.run(in_td);
        return pydict_from_tensordict(out_td);
    }

    bool is_generation_complete() const {
        return static_cast<const bricks::IBrickV1&>(impl_).is_generation_complete();
    }

    bool is_output_ready() const { return is_generation_complete(); }

    std::string name() const { return impl_.name(); }

    std::string write_profile() const {
        std::ostringstream os;
        static_cast<const bricks::IBrickV1&>(impl_).write_profile(os);
        return os.str();
    }

    void display_profile(bool enable_op_profiling = false) const {
        (void)enable_op_profiling;
        static_cast<const bricks::IBrickV1&>(impl_).write_profile(std::cout);
    }

    void reset_states() { impl_.reset_states(); }

private:
    BrickT impl_{};
};

template <typename BrickT>
py::class_<PyBrick<BrickT>> bind_brick(py::module_& m, const char* class_name) {
    using PyBrickT = PyBrick<BrickT>;

    py::class_<PyBrickT> cls(m, class_name);
    cls.def(py::init<>())
        .def("validate", &PyBrickT::validate, py::arg("manifest"))
        .def(
            "setup",
            &PyBrickT::setup,
            py::arg("manifest"),
            py::arg_v("options", bricks::BrickOptions{}, "BrickOptions()"))
        .def("run", &PyBrickT::run, py::arg("input"))
        .def("is_generation_complete", &PyBrickT::is_generation_complete)
        .def("is_output_ready", &PyBrickT::is_output_ready)
        .def("name", &PyBrickT::name)
        .def("write_profile", &PyBrickT::write_profile)
        .def("display_profile", &PyBrickT::display_profile, py::arg("enable_op_profiling") = false)
        .def("reset_states", &PyBrickT::reset_states);

    return cls;
}

}  // namespace compute_flow::pybind
