#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set(litert_cmake "${LITERT_SOURCE_DIR}/litert/CMakeLists.txt")
file(READ "${litert_cmake}" litert_cmake_text)
set(tensor_api_block
    "if(EXISTS \"\${CMAKE_CURRENT_SOURCE_DIR}/../tensor/CMakeLists.txt\")\n  add_subdirectory(\"\${CMAKE_CURRENT_SOURCE_DIR}/../tensor\"\n                   \"\${CMAKE_CURRENT_BINARY_DIR}/tensor\")\nendif()"
)

string(
    REPLACE "${tensor_api_block}"
    "if(LITERT_BUILD_TENSOR_API AND EXISTS \"\${CMAKE_CURRENT_SOURCE_DIR}/../tensor/CMakeLists.txt\")\n  add_subdirectory(\"\${CMAKE_CURRENT_SOURCE_DIR}/../tensor\"\n                   \"\${CMAKE_CURRENT_BINARY_DIR}/tensor\")\nendif()"
    litert_cmake_text
    "${litert_cmake_text}"
)
file(WRITE "${litert_cmake}" "${litert_cmake_text}")
