#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import shutil
import textwrap

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration


class MnnConan(ConanFile):
    name = "mnn"
    version = "3.6.1"
    settings = "os", "arch", "compiler", "build_type"

    def package_id(self):
        # This recipe wraps a prebuilt prefix; allow consumers to use any application build type.
        del self.info.settings.build_type

    def _deps_root(self):
        configured = self.conf.get("user.compute_flow_deps:deps_root", default=None)
        if not configured:
            return None
        if not os.path.isabs(configured):
            raise ConanInvalidConfiguration(
                f"user.compute_flow_deps:deps_root must be an absolute path.\nGot: '{configured}'"
            )
        return os.path.abspath(configured)

    def validate(self):
        deps_root = self._deps_root()
        if deps_root is None:
            return
        prefix = os.path.join(deps_root, "mnn")
        required = [
            os.path.join("lib", "libMNN.a"),
            os.path.join("include", "MNN", "Interpreter.hpp"),
            os.path.join("include", "MNN", "Tensor.hpp"),
        ]
        missing = [
            artifact
            for artifact in required
            if not os.path.exists(os.path.join(prefix, artifact))
        ]
        if missing:
            formatted = "\n  ".join(missing)
            raise ConanInvalidConfiguration(
                f"Expected an MNN prefix under '{prefix}'.\nMissing artifacts:\n  {formatted}"
            )

        if str(self.settings.arch) == "armv8":
            library = os.path.join(prefix, "lib", "libMNN.a")
            with open(library, "rb") as library_file:
                if b"KleidiAI is running!" not in library_file.read():
                    raise ConanInvalidConfiguration(
                        f"Expected Arm64 MNN library '{library}' to include KleidiAI."
                    )

    def package(self):
        deps_root = self._deps_root()
        if deps_root is None:
            raise ConanInvalidConfiguration(
                "No dependency root provided. Use export-pkg with:\n"
                "  -c user.compute_flow_deps:deps_root=/absolute/path/to/build/_deps/install"
            )
        shutil.copytree(
            os.path.join(deps_root, "mnn"), self.package_folder, dirs_exist_ok=True
        )
        self._write_cmake_config()

    def _write_cmake_config(self):
        config_dir = os.path.join(self.package_folder, "lib", "cmake", "MNN")
        os.makedirs(config_dir, exist_ok=True)
        with open(
            os.path.join(config_dir, "MNNConfig.cmake"), "w", encoding="utf-8"
        ) as config_file:
            config_file.write(
                textwrap.dedent(
                    """\
                    #
                    # SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
                    #
                    # SPDX-License-Identifier: Apache-2.0
                    #

                    include(CMakeFindDependencyMacro)
                    find_dependency(Threads REQUIRED)

                    get_filename_component(_mnn_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
                    set(_mnn_include_dir "${_mnn_prefix}/include")
                    set(_mnn_library "${_mnn_prefix}/lib/libMNN.a")

                    if(NOT TARGET MNN::MNN)
                        add_library(MNN::MNN STATIC IMPORTED)
                        set_target_properties(
                            MNN::MNN
                            PROPERTIES
                                IMPORTED_LOCATION "${_mnn_library}"
                                INTERFACE_INCLUDE_DIRECTORIES "${_mnn_include_dir}"
                        )
                        target_link_libraries(MNN::MNN INTERFACE Threads::Threads)
                        if(ANDROID)
                            target_link_libraries(MNN::MNN INTERFACE log m android)
                        elseif(UNIX AND NOT APPLE)
                            target_link_libraries(MNN::MNN INTERFACE dl)
                        endif()
                    endif()

                    set(MNN_FOUND TRUE)
                    """
                )
            )

    def package_info(self):
        # Upstream MNN does not install a CMake config package for this static build.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["."]
