#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(protobuf_test C CXX)

find_package(Protobuf CONFIG REQUIRED)

if(NOT TARGET protobuf::libprotobuf)
    message(FATAL_ERROR "Expected target 'protobuf::libprotobuf' from Protobuf CMake config package")
endif()
