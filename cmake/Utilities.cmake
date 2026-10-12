#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# ---- compute_flow_install_python_extension -------------------------------------------------
# Installs a Python extension module into the active Python environment's site-packages.
# Optionally updates a caller-visible variable with the full installation path.
#
# Usage:
#   compute_flow_install_python_extension(<target> [<subdir_var>])
#
function(compute_flow_install_python_extension target)
    if(CMAKE_CROSSCOMPILING)
        return()
    endif()

    if(NOT TARGET ${target})
        message(FATAL_ERROR "compute_flow_install_python_extension: target '${target}' does not exist.")
    endif()

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE site_packages
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(site_packages STREQUAL "")
        message(FATAL_ERROR "Python site-packages could not be determined for interpreter: ${Python3_EXECUTABLE}")
    endif()
    if(NOT IS_ABSOLUTE "${site_packages}")
        message(FATAL_ERROR "Python site-packages path must be absolute, got: '${site_packages}'")
    endif()

    message(STATUS "${target} install enabled: Python='${Python3_EXECUTABLE}', site-packages='${site_packages}'")

    if(ARGC GREATER 1)
        # Update ${site_packages} with the subdir
        set(subdir "${${ARGV1}}")
        set(site_packages "${site_packages}/${subdir}")
    endif()

    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${site_packages}"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:${target}>"
            "${site_packages}/$<TARGET_FILE_NAME:${target}>"
        COMMAND "${CMAKE_COMMAND}" -E echo "Installed ${target} to: ${site_packages}"
        VERBATIM
    )

    if(ARGC GREATER 1)
        set(${ARGV1} "${site_packages}" PARENT_SCOPE)
    endif()
endfunction()

# ---- compute_flow_extract_target_archives -------------------------------------------------
# Extracts archive paths from an IMPORTED target for monolithic archive declaration.
# Reads IMPORTED_LOCATION and collects .a/.lib paths from INTERFACE_LINK_LIBRARIES.
# Used by dep modules in the find_package (prebuilt) path to populate
# COMPUTE_FLOW_<DEP>_MONOLITHIC_ARCHIVES.
#
# Usage:
#   compute_flow_extract_target_archives(<out_var> <target> [<target2> ...])
#
function(compute_flow_extract_target_archives out_var)
    set(result "")
    foreach(target IN LISTS ARGN)
        if(NOT TARGET "${target}")
            continue()
        endif()

        # Collect IMPORTED_LOCATION (try RELEASE config first, then generic)
        get_target_property(loc "${target}" IMPORTED_LOCATION_RELEASE)
        if(NOT loc)
            get_target_property(loc "${target}" IMPORTED_LOCATION)
        endif()
        if(loc)
            list(APPEND result "${loc}")
        endif()

        # Collect archive paths from INTERFACE_LINK_LIBRARIES
        get_target_property(iface_libs "${target}" INTERFACE_LINK_LIBRARIES)
        if(iface_libs)
            foreach(iface_lib IN LISTS iface_libs)
                if(iface_lib MATCHES "\\.(a|lib)$")
                    list(APPEND result "${iface_lib}")
                endif()
            endforeach()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES result)
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# ---- compute_flow_collect_static_link_archives -------------------------------------------
# Collects the static archives in a target's transitive link closure. Core Compute Flow
# targets are excluded because applications link the packaged core archive separately.
function(compute_flow_collect_static_link_archives out_archives out_targets)
    set(todo ${ARGN})
    set(visited "")
    set(archives "")
    set(targets "")
    set(excluded_targets compute_flow compute_flow_core compute_flow_objs)

    while(todo)
        list(POP_FRONT todo current)

        if(current MATCHES "\\$<")
            continue()
        endif()

        if(TARGET "${current}")
            get_target_property(aliased_target "${current}" ALIASED_TARGET)
            if(aliased_target)
                set(current "${aliased_target}")
            endif()

            if(current IN_LIST visited OR current IN_LIST excluded_targets)
                continue()
            endif()
            list(APPEND visited "${current}")

            get_target_property(target_type "${current}" TYPE)
            if(target_type STREQUAL "STATIC_LIBRARY")
                list(APPEND archives "$<TARGET_FILE:${current}>")
                list(APPEND targets "${current}")
            endif()

            get_target_property(link_libraries "${current}" LINK_LIBRARIES)
            if(link_libraries AND NOT link_libraries STREQUAL "LINK_LIBRARIES-NOTFOUND")
                list(APPEND todo ${link_libraries})
            endif()

            get_target_property(interface_link_libraries "${current}" INTERFACE_LINK_LIBRARIES)
            if(interface_link_libraries AND NOT interface_link_libraries STREQUAL "INTERFACE_LINK_LIBRARIES-NOTFOUND")
                list(APPEND todo ${interface_link_libraries})
            endif()
        elseif(current MATCHES "\\.(a|lib)$")
            list(APPEND archives "${current}")
        endif()
    endwhile()

    list(REMOVE_DUPLICATES archives)
    list(REMOVE_DUPLICATES targets)
    set(${out_archives} "${archives}" PARENT_SCOPE)
    set(${out_targets} "${targets}" PARENT_SCOPE)
endfunction()

# ---- compute_flow_add_monolithic_archive -------------------------------------------------
# Creates a monolithic static archive by merging per-dependency archives at build time.
#
# Usage:
#   compute_flow_add_monolithic_archive(
#       <target> <out_path>
#       OBJECTS <object_lib>
#       PROVENANCES <name1> <name2> ...
#       PRIORITY_ORDER <dep1> <dep2> ...
#   )
# Each provenance is a name. Archives and glob dirs are looked up via a PREFIX_MAP:
#   MONOLITHIC_PREFIX_MAP_<name> -> <PREFIX>
#   COMPUTE_FLOW_<PREFIX>_MONOLITHIC_ARCHIVES -> archive list
#   COMPUTE_FLOW_<PREFIX>_MONOLITHIC_GLOB_DIR -> glob dir (optional)
# These must be set in the calling scope.
# PRIORITY_ORDER controls dedup when archives share a basename across provenances.
#
function(compute_flow_add_monolithic_archive target out_path)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "compute_flow_add_monolithic_archive: target '${target}' does not exist.")
    endif()

    cmake_parse_arguments(MONO "" "OBJECTS" "PROVENANCES;PRIORITY_ORDER" ${ARGN})

    if(NOT MONO_PROVENANCES)
        message(FATAL_ERROR "compute_flow_add_monolithic_archive: PROVENANCES is required.")
    endif()

    _compute_flow_monolithic_provenances(
        "${target}" "${out_path}" "${MONO_OBJECTS}"
        "${MONO_PROVENANCES}" "${MONO_PRIORITY_ORDER}"
    )
endfunction()

# ---- _compute_flow_monolithic_provenances ------------------------------------------------
# PROVENANCES mode: generates a config file and delegates to MonolithicArchive.cmake.
#
# provenances is a flat list of provenance names (e.g. "protobuf;tensorflow-lite;...").
# For each name, archives and glob dirs are looked up via a PREFIX_MAP:
#   MONOLITHIC_PREFIX_MAP_<name> → <PREFIX>
#   COMPUTE_FLOW_<PREFIX>_MONOLITHIC_ARCHIVES → archive list
#   COMPUTE_FLOW_<PREFIX>_MONOLITHIC_GLOB_DIR → glob dir (optional)
#
# These mapping variables must be set in the calling scope (CMakeLists.txt).
function(
    _compute_flow_monolithic_provenances
    target
    out_path
    objects_lib
    provenances
    priority_order
)
    set(config_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_monolithic_config.cmake")
    set(monolithic_filelist "${CMAKE_CURRENT_BINARY_DIR}/${target}_monolithic.filelist")
    set(monolithic_log "${CMAKE_CURRENT_BINARY_DIR}/${target}_monolithic_archive_manifest.log")
    set(monolithic_mri "${CMAKE_CURRENT_BINARY_DIR}/${target}_monolithic.mri")
    set(monolithic_archive_script "${CMAKE_SOURCE_DIR}/cmake/MonolithicArchive.cmake")

    set(config_content "# Auto-generated monolithic archive config — do not edit\n")

    # Handle OBJECTS lib: resolve to object files via generator expression, written to a
    # separate filelist since file(GENERATE) supports genex but file(WRITE) does not.
    set(objects_filelist "")
    if(objects_lib AND TARGET "${objects_lib}")
        set(objects_filelist "${CMAKE_CURRENT_BINARY_DIR}/${target}_monolithic_objects.filelist")
        file(GENERATE OUTPUT "${objects_filelist}" CONTENT "$<JOIN:$<TARGET_OBJECTS:${objects_lib}>,\n>\n")
        # Also add as link dependency
        set_property(TARGET "${target}" APPEND PROPERTY LINK_DEPENDS "$<TARGET_OBJECTS:${objects_lib}>")
    endif()

    # Write provenance entries by looking up the prefix map
    foreach(prov_name IN LISTS provenances)
        # Look up the variable prefix for this provenance
        set(prefix "${MONOLITHIC_PREFIX_MAP_${prov_name}}")
        if(NOT prefix)
            message(
                FATAL_ERROR
                "compute_flow_add_monolithic_archive: no MONOLITHIC_PREFIX_MAP_${prov_name} set for provenance '${prov_name}'"
            )
        endif()

        set(prov_archives "${COMPUTE_FLOW_${prefix}_MONOLITHIC_ARCHIVES}")
        set(prov_glob_dir "${COMPUTE_FLOW_${prefix}_MONOLITHIC_GLOB_DIR}")

        # Write archive list (may be empty if dep is disabled)
        string(APPEND config_content "set(MONOLITHIC_ARCHIVES_${prov_name} \"${prov_archives}\")\n")

        # Write glob dir if non-empty
        if(NOT "${prov_glob_dir}" STREQUAL "")
            string(APPEND config_content "set(MONOLITHIC_GLOB_DIR_${prov_name} \"${prov_glob_dir}\")\n")
        endif()

        # Write explicit exclude list if non-empty
        set(prov_exclude "${COMPUTE_FLOW_${prefix}_MONOLITHIC_EXCLUDE}")
        if(NOT "${prov_exclude}" STREQUAL "")
            string(APPEND config_content "set(MONOLITHIC_EXCLUDE_${prov_name} \"${prov_exclude}\")\n")
        endif()

        # Add archive paths as LINK_DEPENDS so CMake rebuilds when they change
        if(prov_archives)
            set_property(TARGET "${target}" APPEND PROPERTY LINK_DEPENDS ${prov_archives})
        endif()
    endforeach()

    # Write global config
    list(JOIN provenances ";" prov_names_str)
    if(APPLE)
        set(_monolithic_is_macos ON)
        if(NOT LIBTOOL_EXECUTABLE)
            find_program(LIBTOOL_EXECUTABLE NAMES libtool REQUIRED)
        endif()
        set(_monolithic_merge_tool "${LIBTOOL_EXECUTABLE}")
    else()
        set(_monolithic_is_macos OFF)
        set(_monolithic_merge_tool "${CMAKE_AR}")
    endif()
    string(
        PREPEND config_content
        "set(MONOLITHIC_PRIORITY_ORDER \"${priority_order}\")\n"
        "set(MONOLITHIC_OUTPUT \"${out_path}\")\n"
        "set(MONOLITHIC_LOG \"${monolithic_log}\")\n"
        "set(MONOLITHIC_FILELIST \"${monolithic_filelist}\")\n"
        "set(MONOLITHIC_MRI \"${monolithic_mri}\")\n"
        "set(MONOLITHIC_MERGE_TOOL \"${_monolithic_merge_tool}\")\n"
        "set(MONOLITHIC_IS_MACOS ${_monolithic_is_macos})\n"
        "set(MONOLITHIC_OBJECTS_FILELIST \"${objects_filelist}\")\n"
        "set(MONOLITHIC_PROVENANCES \"${prov_names_str}\")\n"
    )

    # Use file(GENERATE) to resolve the out_path genex ($<TARGET_FILE:...>)
    file(GENERATE OUTPUT "${config_file}" CONTENT "${config_content}")

    # POST_BUILD: run the build-time script (collect, dedup, merge, log — all in one)
    add_custom_command(
        TARGET "${target}"
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" "-DMONOLITHIC_CONFIG=${config_file}" -P "${monolithic_archive_script}"
        VERBATIM
    )
endfunction()
