#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(onnxruntime_test C CXX)

find_package(onnxruntime CONFIG REQUIRED)

if(NOT TARGET onnxruntime::onnxruntime_session)
    message(FATAL_ERROR "Expected target 'onnxruntime::onnxruntime_session' from onnxruntimeConfig.cmake")
endif()
