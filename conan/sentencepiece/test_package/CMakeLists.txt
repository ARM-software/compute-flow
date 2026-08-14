#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(sentencepiece_test C CXX)

find_package(sentencepiece CONFIG REQUIRED)

if(NOT TARGET sentencepiece-static AND NOT TARGET sentencepiece::sentencepiece-static AND NOT TARGET sentencepiece)
    message(FATAL_ERROR "Expected a SentencePiece CMake target (e.g. sentencepiece-static)")
endif()
