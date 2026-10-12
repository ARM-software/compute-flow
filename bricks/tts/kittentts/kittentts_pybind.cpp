//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/bindings/python/brick_pybind.h"

#include "KittenTTS.h"

namespace py = pybind11;

namespace compute_flow::bricks::tts {

void bind_kittentts_py(py::module_& m) {
    m.doc() = "Python bindings for KittenTTS (pybind11)";
    compute_flow::pybind::bind_brick<KittenTTS>(m, "KittenTTS");
}

}  // namespace compute_flow::bricks::tts

PYBIND11_MODULE(kittentts_py, m) {
    compute_flow::bricks::tts::bind_kittentts_py(m);
}
