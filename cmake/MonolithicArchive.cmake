#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# =========================================================================================
# MonolithicArchive.cmake — Build-time monolithic archive assembly
# =========================================================================================
#
# Single build-time script that assembles a monolithic static archive from per-dependency
# archive declarations.
#
# This script is invoked as `cmake -P MonolithicArchive.cmake` from a POST_BUILD step.
# All inputs arrive via a generated config file (native CMake set() calls) that is
# include()'d at the top.
#
# == Phases ==
#
# 1. COLLECT
#    For each provenance (dependency name), read its declared archives from the config.
#    If a MONOLITHIC_GLOB_DIR_<provenance> is set, glob that directory for additional
#    .a files and merge them into that provenance's list.
#    If a MONOLITHIC_EXCLUDE_<provenance> is set (a list of basenames), any matching
#    archive is skipped during collection and logged as excluded.
#
# 2. DEDUPLICATE
#    Among prioritized provenances (those listed in MONOLITHIC_PRIORITY_ORDER), dedup
#    by basename — keep the copy from the higher-priority (lower-index) provenance and
#    exclude lower-priority duplicates. Archives from unprioritized provenances pass
#    through untouched (they are assumed to have non-conflicting symbols).
#
# 3. MERGE
#    Write the final archive list to a filelist. On macOS®, the caller uses
#    `libtool -static -filelist`. On Linux®, generate an MRI script for `ar -M`.
#
# 4. LOG
#    Write a manifest to MONOLITHIC_LOG showing:
#    - All included archives with provenance and full path
#    - All excluded archives with reason (explicit exclude or priority dedup)
#
# == Config file format ==
#
# The config file is a native CMake script with set() calls:
#
#   set(MONOLITHIC_PRIORITY_ORDER "tensorflow-lite" "executorch")
#   set(MONOLITHIC_OUTPUT "/path/to/libcompute_flow.a")
#   set(MONOLITHIC_LOG "/path/to/monolithic_archive_manifest.log")
#   set(MONOLITHIC_FILELIST "/path/to/monolithic.filelist")
#   set(MONOLITHIC_MRI "/path/to/monolithic.mri")  # Linux® only
#   set(MONOLITHIC_MERGE_TOOL "/usr/bin/libtool")   # libtool (macOS®) or ar (Linux®)
#   set(MONOLITHIC_IS_MACOS ON)                      # ON for macOS®, OFF for Linux®
#   set(MONOLITHIC_PROVENANCES "core;protobuf;tensorflow-lite;...")
#   set(MONOLITHIC_ARCHIVES_core "/path/obj1.o;/path/obj2.o")
#   set(MONOLITHIC_ARCHIVES_onnxruntime "/path/lib1.a;/path/lib2.a")
#   set(MONOLITHIC_GLOB_DIR_onnxruntime "/path/install/onnxruntime/lib")
#   set(MONOLITHIC_EXCLUDE_onnxruntime "libkleidiai.a")  # basenames to skip
#   ...
#
# Required variable (passed via -D):
#   MONOLITHIC_CONFIG - path to the generated config file
# =========================================================================================

if(NOT DEFINED MONOLITHIC_CONFIG)
    message(FATAL_ERROR "MonolithicArchive: MONOLITHIC_CONFIG not set")
endif()
if(NOT EXISTS "${MONOLITHIC_CONFIG}")
    message(FATAL_ERROR "MonolithicArchive: config file not found: ${MONOLITHIC_CONFIG}")
endif()

include("${MONOLITHIC_CONFIG}")

# =========================================================================================
# Phase 1: COLLECT
# =========================================================================================

set(all_archives "")
set(all_provenances "") # parallel list: provenance for each archive
set(excluded_archives "") # archives skipped by explicit exclude lists
set(excluded_archive_provenances "") # parallel list: provenance for each excluded archive

foreach(prov IN LISTS MONOLITHIC_PROVENANCES)
    set(prov_archives "${MONOLITHIC_ARCHIVES_${prov}}")

    # Add declared archives (skipping explicitly excluded basenames)
    foreach(archive IN LISTS prov_archives)
        get_filename_component(_basename "${archive}" NAME)
        if(DEFINED MONOLITHIC_EXCLUDE_${prov})
            list(FIND MONOLITHIC_EXCLUDE_${prov} "${_basename}" _excl_match)
            if(NOT _excl_match EQUAL -1)
                list(APPEND excluded_archives "${archive}")
                list(APPEND excluded_archive_provenances "${prov}")
                continue()
            endif()
        endif()
        list(APPEND all_archives "${archive}")
        list(APPEND all_provenances "${prov}")
    endforeach()

    # Glob additional archives if GLOB_DIR is set (also respects exclude list)
    if(DEFINED MONOLITHIC_GLOB_DIR_${prov} AND NOT "${MONOLITHIC_GLOB_DIR_${prov}}" STREQUAL "")
        set(glob_dir "${MONOLITHIC_GLOB_DIR_${prov}}")
        if(EXISTS "${glob_dir}")
            file(GLOB_RECURSE globbed_archives "${glob_dir}/*.a")
            foreach(globbed IN LISTS globbed_archives)
                get_filename_component(_basename "${globbed}" NAME)
                if(DEFINED MONOLITHIC_EXCLUDE_${prov})
                    list(FIND MONOLITHIC_EXCLUDE_${prov} "${_basename}" _excl_match)
                    if(NOT _excl_match EQUAL -1)
                        list(APPEND excluded_archives "${globbed}")
                        list(APPEND excluded_archive_provenances "${prov}")
                        continue()
                    endif()
                endif()
                # Skip if already present (declared or from a previous provenance)
                list(FIND all_archives "${globbed}" already_in_all)
                if(already_in_all EQUAL -1)
                    list(APPEND all_archives "${globbed}")
                    list(APPEND all_provenances "${prov}")
                endif()
            endforeach()
        endif()
    endif()
endforeach()

# Load object files from the objects filelist (generated by file(GENERATE) with genex)
set(object_files "")
if(DEFINED MONOLITHIC_OBJECTS_FILELIST AND NOT "${MONOLITHIC_OBJECTS_FILELIST}" STREQUAL "")
    if(EXISTS "${MONOLITHIC_OBJECTS_FILELIST}")
        file(STRINGS "${MONOLITHIC_OBJECTS_FILELIST}" object_files)
        list(FILTER object_files EXCLUDE REGEX "^$")
    endif()
endif()

list(LENGTH all_archives total_collected)
list(LENGTH object_files total_objects)
message(
    STATUS
    "MonolithicArchive: collected ${total_collected} archives + ${total_objects} objects from ${MONOLITHIC_PROVENANCES}"
)

# =========================================================================================
# Phase 2: DEDUPLICATE
# =========================================================================================
#
# Among prioritized provenances only: for each basename, keep the copy from the
# highest-priority (lowest-rank) provenance. Archives from unprioritized provenances
# pass through — they ship independent builds with non-conflicting symbols.
# Cross-provenance conflicts between prioritized and unprioritized deps are handled
# by explicit exclude lists during Phase 1 collection.

set(included_indices "")
set(excluded_indices "")
set(excluded_reasons "")

if(total_collected GREATER 0)
    math(EXPR _last_index "${total_collected} - 1")

    # -- Priority dedup: prioritized provenances only --
    foreach(idx RANGE 0 ${_last_index})
        list(GET all_archives ${idx} archive)
        list(GET all_provenances ${idx} prov)
        list(FIND MONOLITHIC_PRIORITY_ORDER "${prov}" prov_rank)

        # Unprioritized provenances: always include
        if(prov_rank EQUAL -1)
            list(APPEND included_indices ${idx})
            continue()
        endif()

        get_filename_component(basename "${archive}" NAME)

        if(NOT DEFINED _BEST_IDX_${basename})
            # First prioritized archive with this basename
            list(APPEND included_indices ${idx})
            set(_BEST_IDX_${basename} ${idx})
            set(_BEST_RANK_${basename} ${prov_rank})
        else()
            # Conflict: both are prioritized — keep the higher-priority (lower rank)
            set(prev_idx "${_BEST_IDX_${basename}}")
            set(prev_rank "${_BEST_RANK_${basename}}")

            if(prov_rank LESS prev_rank)
                # Current wins: replace previous
                list(REMOVE_ITEM included_indices ${prev_idx})
                list(APPEND excluded_indices ${prev_idx})
                list(APPEND excluded_reasons "superseded by: ${prov}")
                list(APPEND included_indices ${idx})
                set(_BEST_IDX_${basename} ${idx})
                set(_BEST_RANK_${basename} ${prov_rank})
            else()
                # Previous wins: exclude current
                list(GET all_provenances ${prev_idx} winner_prov)
                list(APPEND excluded_indices ${idx})
                list(APPEND excluded_reasons "superseded by: ${winner_prov}")
            endif()
        endif()
    endforeach()
endif()

list(LENGTH included_indices num_included)
list(LENGTH excluded_indices num_excluded)
message(STATUS "MonolithicArchive: ${num_included} included, ${num_excluded} excluded (duplicates)")

# =========================================================================================
# Phase 3: MERGE
# =========================================================================================

# Build the final archive list from included indices, prepended with object files
set(final_archives ${object_files})
foreach(idx IN LISTS included_indices)
    list(GET all_archives ${idx} archive)
    list(APPEND final_archives "${archive}")
endforeach()

file(REMOVE "${MONOLITHIC_OUTPUT}")

if(MONOLITHIC_IS_MACOS)
    # macOS®: write filelist then invoke libtool
    list(JOIN final_archives "\n" filelist_content)
    string(APPEND filelist_content "\n")
    file(WRITE "${MONOLITHIC_FILELIST}" "${filelist_content}")

    execute_process(
        COMMAND "${MONOLITHIC_MERGE_TOOL}" -static -o "${MONOLITHIC_OUTPUT}" -filelist "${MONOLITHIC_FILELIST}"
        RESULT_VARIABLE merge_result
        ERROR_VARIABLE merge_stderr
    )
else()
    # Linux®: generate MRI script then invoke ar -M
    set(mri_content "CREATE ${MONOLITHIC_OUTPUT}\n")
    foreach(entry IN LISTS final_archives)
        if(entry MATCHES "\\.(o|obj)$")
            string(APPEND mri_content "ADDMOD ${entry}\n")
        else()
            string(APPEND mri_content "ADDLIB ${entry}\n")
        endif()
    endforeach()
    string(APPEND mri_content "SAVE\nEND\n")
    file(WRITE "${MONOLITHIC_MRI}" "${mri_content}")

    execute_process(
        COMMAND /bin/sh -c "\"${MONOLITHIC_MERGE_TOOL}\" -M < \"${MONOLITHIC_MRI}\""
        RESULT_VARIABLE merge_result
        ERROR_VARIABLE merge_stderr
    )
endif()

if(NOT merge_result EQUAL 0)
    message(FATAL_ERROR "MonolithicArchive: merge failed (exit ${merge_result}): ${merge_stderr}")
endif()

message(STATUS "MonolithicArchive: merged ${num_included} archives into ${MONOLITHIC_OUTPUT}")

# =========================================================================================
# Phase 4: LOG
# =========================================================================================

if(DEFINED MONOLITHIC_LOG AND NOT "${MONOLITHIC_LOG}" STREQUAL "")
    set(log_content "=== Monolithic Archive Manifest ===\n")
    string(APPEND log_content "Generated: ${MONOLITHIC_OUTPUT}\n\n")

    string(APPEND log_content "--- Included (${num_included} archives) ---\n")
    foreach(idx IN LISTS included_indices)
        list(GET all_archives ${idx} archive)
        list(GET all_provenances ${idx} prov)
        get_filename_component(basename "${archive}" NAME)
        string(APPEND log_content "${basename} [${prov}] ${archive}\n")
    endforeach()

    if(num_excluded GREATER 0)
        string(APPEND log_content "\n--- Excluded (${num_excluded} priority duplicates) ---\n")
        set(_excl_idx 0)
        foreach(idx IN LISTS excluded_indices)
            list(GET all_archives ${idx} archive)
            list(GET all_provenances ${idx} prov)
            list(GET excluded_reasons ${_excl_idx} reason)
            get_filename_component(basename "${archive}" NAME)
            string(APPEND log_content "${basename} [${prov}] ${archive} (${reason})\n")
            math(EXPR _excl_idx "${_excl_idx} + 1")
        endforeach()
    endif()

    list(LENGTH excluded_archives num_explicitly_excluded)
    if(num_explicitly_excluded GREATER 0)
        string(APPEND log_content "\n--- Excluded (${num_explicitly_excluded} explicit) ---\n")
        set(_excl_idx 0)
        foreach(archive IN LISTS excluded_archives)
            list(GET excluded_archive_provenances ${_excl_idx} prov)
            get_filename_component(basename "${archive}" NAME)
            string(APPEND log_content "${basename} [${prov}] ${archive} (explicitly excluded)\n")
            math(EXPR _excl_idx "${_excl_idx} + 1")
        endforeach()
    endif()

    file(WRITE "${MONOLITHIC_LOG}" "${log_content}")
    message(STATUS "MonolithicArchive: manifest written to ${MONOLITHIC_LOG}")
endif()
