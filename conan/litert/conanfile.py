#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import shutil

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration


class LiteRtConan(ConanFile):
    name = "litert"
    # Keep in sync with the superbuild (cmake/deps/litert/LiteRT.cmake).
    version = "2.1.6"
    settings = "os", "arch", "compiler", "build_type"

    def package_id(self):
        # This recipe wraps a prebuilt prefix; allow consumers to build their application
        # in Debug while depending on Release-built static libs.
        del self.info.settings.build_type

    def _deps_root(self):
        configured = self.conf.get("user.compute_flow_deps:deps_root", default=None)
        if not configured:
            # Consuming from cache: no need to validate an external staging folder.
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
        prefix = os.path.join(deps_root, "litert")
        if not os.path.isdir(prefix):
            raise ConanInvalidConfiguration(
                f"Expected LiteRT prefix at: '{prefix}'.\n"
                "Build the superbuild first, then re-run export-pkg."
            )

    def package(self):
        deps_root = self._deps_root()
        if deps_root is None:
            raise ConanInvalidConfiguration(
                "No deps root provided. This recipe packages prebuilt binaries; use export-pkg with:\n"
                "  -c user.compute_flow_deps:deps_root=/abs/path/to/build/_deps/install"
            )
        src = os.path.join(deps_root, "litert")
        shutil.copytree(src, self.package_folder, dirs_exist_ok=True)

    def package_info(self):
        # Prefer the dependency-provided CMake config package (litertConfig.cmake).
        # Consumers should use `find_package(litert CONFIG REQUIRED)`.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["."]
