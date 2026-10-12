//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/bindings/python/brick_pybind.h"

#include "Kokoro.h"

namespace py = pybind11;

namespace compute_flow::bricks::tts {

void bind_kokoro_py(py::module_& m) {
    m.doc() = "Python bindings for Kokoro (pybind11)";
    compute_flow::pybind::bind_brick<Kokoro>(m, "Kokoro");
}

}  // namespace compute_flow::bricks::tts

PYBIND11_MODULE(kokoro_py, m) {
    compute_flow::bricks::tts::bind_kokoro_py(m);
}
