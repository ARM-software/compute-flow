#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(mnn_test CXX)

find_package(MNN CONFIG REQUIRED)

if(NOT TARGET MNN::MNN)
    message(FATAL_ERROR "Expected target 'MNN::MNN' from MNNConfig.cmake")
endif()
