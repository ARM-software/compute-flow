#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(executorch_test C CXX)

find_package(executorch CONFIG REQUIRED)

if(NOT TARGET executorch)
    message(FATAL_ERROR "Expected target 'executorch' from executorchConfig.cmake")
endif()
