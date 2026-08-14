#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(litert_test C CXX)

find_package(litert CONFIG REQUIRED)

if(NOT TARGET litert::litert)
    message(FATAL_ERROR "Expected target 'litert::litert' from litertConfig.cmake")
endif()
