#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Shell completion for Compute Flow Brick Manager commands and brick CLI selectors.
# Supports bash and zsh when sourced from the project virtual environment.

_compute_flow_brick_cli_type_from_word() {
    local word="${1##*/}"
    word="${word%.py}"
    word="${word##*.}"
    [[ "$word" == *_cli ]] || return 1
    printf '%s\n' "${word%_cli}"
}

_compute_flow_brick_cli_script_dir() {
    if [ -n "${BASH_SOURCE[0]:-}" ]; then
        dirname "${BASH_SOURCE[0]}"
    elif [ -n "${(%):-%x}" ]; then
        dirname "${(%):-%x}"
    else
        pwd
    fi
}

_compute_flow_brick_cli_repo_dir() {
    local script_dir
    script_dir="$(_compute_flow_brick_cli_script_dir)"

    if [ -d "$script_dir/../../compute_flow/cli_examples" ]; then
        cd "$script_dir/../.." >/dev/null 2>&1 && pwd
        return
    fi

    if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/../compute_flow/cli_examples" ]; then
        cd "$VIRTUAL_ENV/.." >/dev/null 2>&1 && pwd
        return
    fi
}

_compute_flow_completion_commands() {
    if [ -n "${ZSH_VERSION:-}" ]; then
        setopt local_options null_glob
    fi

    printf '%s\n' python python3

    local repo_dir
    repo_dir="$(_compute_flow_brick_cli_repo_dir)"
    if [ -n "$repo_dir" ]; then
        local cli_source cli_name
        for cli_source in "$repo_dir"/compute_flow/cli_examples/*_cli.cpp "$repo_dir"/compute_flow/python/cli_examples/*_cli.py; do
            [ -f "$cli_source" ] || continue
            cli_name="${cli_source##*/}"
            printf '%s\n' "${cli_name%.*}"
            case "$cli_name" in
                *.py) printf '%s\n' "$cli_name" ;;
            esac
        done
    fi

    local path_dir path_entry command_name
    local old_ifs="$IFS"
    IFS=:
    for path_dir in $PATH; do
        [ -d "$path_dir" ] || continue
        for path_entry in "$path_dir"/*_cli "$path_dir"/*_cli.py "$path_dir"/python "$path_dir"/python[0-9] "$path_dir"/python[0-9].[0-9] "$path_dir"/python[0-9].[0-9][0-9]; do
            [ -x "$path_entry" ] || continue
            command_name="${path_entry##*/}"
            printf '%s\n' "$command_name"
        done
    done
    IFS="$old_ifs"
}

_compute_flow_models_dir() {
    if [ -n "${COMPUTE_FLOW_HOME:-}" ]; then
        printf '%s/models\n' "$COMPUTE_FLOW_HOME"
    else
        printf '%s/.compute-flow/models\n' "$HOME"
    fi
}

_compute_flow_manifest_selector_candidates_for_type() {
    local type="$1"
    local remainder="$2"
    local candidate_prefix="$3"
    local models_dir
    models_dir="$(_compute_flow_models_dir)"

    [ -d "$models_dir/$type" ] || return 0

    if [[ "$remainder" != */* ]]; then
        local brick_dir brick_name
        for brick_dir in "$models_dir/$type"/*; do
            [ -d "$brick_dir" ] || continue
            brick_name="${brick_dir##*/}"
            case "$brick_name" in
                "$remainder"*) printf '%s%s/\n' "$candidate_prefix" "$brick_name" ;;
            esac
        done
        return 0
    fi

    local brick_name="${remainder%%/*}"
    local after_brick="${remainder#*/}"
    [ -d "$models_dir/$type/$brick_name" ] || return 0

    if [[ "$after_brick" != */* ]]; then
        local model_dir model_name
        for model_dir in "$models_dir/$type/$brick_name"/*; do
            [ -d "$model_dir" ] || continue
            model_name="${model_dir##*/}"
            case "$model_name" in
                "$after_brick"*) printf '%s%s/%s/\n' "$candidate_prefix" "$brick_name" "$model_name" ;;
            esac
        done
        return 0
    fi

    local model_name="${after_brick%%/*}"
    local manifest_prefix="${after_brick#*/}"
    [[ "$manifest_prefix" == */* ]] && return 0
    [ -d "$models_dir/$type/$brick_name/$model_name" ] || return 0

    local manifest_file manifest_name
    for manifest_file in "$models_dir/$type/$brick_name/$model_name"/*manifest*.json; do
        [ -f "$manifest_file" ] || continue
        manifest_name="${manifest_file##*/}"
        case "$manifest_name" in
            "$manifest_prefix"*) printf '%s%s/%s/%s\n' "$candidate_prefix" "$brick_name" "$model_name" "$manifest_name" ;;
        esac
    done
}

_compute_flow_manifest_selector_candidates() {
    local current="$1"
    local default_type="$2"
    local selector_prefix="@manifest/"
    local models_dir
    models_dir="$(_compute_flow_models_dir)"

    [ -d "$models_dir/$default_type" ] || return 0

    case "$current" in
        /* | ./* | ../* | "~"*)
            return 0
            ;;
    esac

    if [[ "$current" == "$selector_prefix"* ]]; then
        _compute_flow_manifest_selector_candidates_for_type \
            "$default_type" "${current#"$selector_prefix"}" "$selector_prefix"
        return 0
    fi

    case "$selector_prefix" in
        "$current"*) printf '%s\n' "$selector_prefix" ;;
    esac
}

_compute_flow_token_allows_files() {
    local current="$1"
    case "$current" in
        /* | ./* | ../* | "~"*)
            return 0
            ;;
        @* | *:* | */*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

_compute_flow_sample_input_option() {
    case "$1" in
        -a | -c | -i) return 0 ;;
        *) return 1 ;;
    esac
}

_compute_flow_sample_root() {
    local manifest_value="$1"
    local brick_type="$2"
    local manifest_prefix="@manifest/"
    local package_root

    if [[ "$manifest_value" == "$manifest_prefix"* ]]; then
        local selector="${manifest_value#"$manifest_prefix"}"
        local brick_name="${selector%%/*}"
        local after_brick="${selector#*/}"
        local model_name="${after_brick%%/*}"
        local manifest_name="${after_brick#*/}"
        [[ "$selector" == */*/* ]] || return 0
        [[ "$manifest_name" == */* ]] && return 0

        package_root="$(_compute_flow_models_dir)/$brick_type/$brick_name/$model_name"
        [ -f "$package_root/$manifest_name" ] || return 0
    else
        [ -f "$manifest_value" ] || return 0
        package_root="$(dirname "$manifest_value")"
    fi

    [ -d "$package_root/samples" ] || return 0
    printf '%s\n' "$package_root/samples"
}

_compute_flow_sample_selector_candidates() {
    if [ -n "${ZSH_VERSION:-}" ]; then
        setopt local_options null_glob
    fi

    local current="$1"
    local brick_type="$2"
    local manifest_value="$3"
    local selector_prefix="@sample/"
    local sample_root
    sample_root="$(_compute_flow_sample_root "$manifest_value" "$brick_type")"
    [ -n "$sample_root" ] || return 0

    if [[ "$current" != "$selector_prefix"* ]]; then
        case "$selector_prefix" in
            "$current"*) printf '%s\n' "$selector_prefix" ;;
        esac
        return 0
    fi

    local relative_path="${current#"$selector_prefix"}"
    case "$relative_path" in
        .. | ../* | */.. | */../*) return 0 ;;
    esac

    local relative_dir=""
    local name_prefix="$relative_path"
    if [[ "$relative_path" == */* ]]; then
        relative_dir="${relative_path%/*}/"
        name_prefix="${relative_path##*/}"
    fi

    local current_dir="$sample_root/${relative_dir%/}"
    [ -d "$current_dir" ] || return 0

    local sample_path sample_name candidate
    for sample_path in "$current_dir"/*; do
        [ -e "$sample_path" ] || continue
        sample_name="${sample_path##*/}"
        case "$sample_name" in
            "$name_prefix"*)
                candidate="$selector_prefix$relative_dir$sample_name"
                [ -d "$sample_path" ] && candidate="$candidate/"
                printf '%s\n' "$candidate"
                ;;
        esac
    done
}

_compute_flow_bash_completing_manifest_arg() {
    local index word
    for ((index = COMP_CWORD - 1; index > 0; index--)); do
        word="${COMP_WORDS[index]}"
        if [ "$word" = "-m" ]; then
            return 0
        fi
        if [[ "$word" == -* ]]; then
            return 1
        fi
    done
    return 1
}

_compute_flow_bash_completing_sample_arg() {
    local index word
    for ((index = COMP_CWORD - 1; index > 0; index--)); do
        word="${COMP_WORDS[index]}"
        if _compute_flow_sample_input_option "$word"; then
            return 0
        fi
        if [[ "$word" == -* ]]; then
            return 1
        fi
    done
    return 1
}

_compute_flow_bash_manifest_value() {
    local index value=""
    for ((index = 0; index + 1 < ${#COMP_WORDS[@]}; index++)); do
        if [ "${COMP_WORDS[index]}" = "-m" ]; then
            value="${COMP_WORDS[index + 1]}"
        fi
    done
    printf '%s\n' "$value"
}

_compute_flow_bash_brick_cli_type() {
    local index brick_type
    for ((index = 0; index < COMP_CWORD; index++)); do
        brick_type="$(_compute_flow_brick_cli_type_from_word "${COMP_WORDS[index]}")" || continue
        printf '%s\n' "$brick_type"
        return 0
    done
    return 1
}

_compute_flow_bash_current_token() {
    local line
    line="${COMP_LINE:0:COMP_POINT}"
    printf '%s\n' "${line##*[[:space:]]}"
}

_compute_flow_bash_complete_files() {
    local current="$1"
    compopt -o filenames 2>/dev/null || true
    COMPREPLY=($(compgen -f -- "$current"))

    if ! command -v compopt >/dev/null 2>&1; then
        local index candidate
        for ((index = 0; index < ${#COMPREPLY[@]}; index++)); do
            candidate="${COMPREPLY[index]}"
            if [ -d "$candidate" ]; then
                [[ "$candidate" == */ ]] || COMPREPLY[index]="$candidate/"
            else
                COMPREPLY[index]="$candidate "
            fi
        done
    fi
}

_compute_flow_bash_keep_completing_directories() {
    local candidate index
    if ! command -v compopt >/dev/null 2>&1; then
        for ((index = 0; index < ${#COMPREPLY[@]}; index++)); do
            candidate="${COMPREPLY[index]}"
            [[ "$candidate" == */ ]] || COMPREPLY[index]="$candidate "
        done
        return
    fi

    for candidate in "${COMPREPLY[@]}"; do
        if [[ "$candidate" == */ ]]; then
            compopt -o nospace 2>/dev/null || true
            return
        fi
    done
}

_compute_flow_bash_complete_manifest_option() {
    local brick_type="$1"
    COMPREPLY=()

    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] && COMPREPLY+=("-m $candidate")
    done <<EOF
$(_compute_flow_manifest_selector_candidates "" "$brick_type")
EOF

    if [ ${#COMPREPLY[@]} -eq 0 ]; then
        COMPREPLY=("-m ")
    else
        _compute_flow_bash_keep_completing_directories
    fi
}

_compute_flow_bash_complete_sample_option() {
    local option="$1"
    local brick_type="$2"
    local manifest_value
    manifest_value="$(_compute_flow_bash_manifest_value)"
    COMPREPLY=()

    local candidate
    while IFS= read -r candidate; do
        [ -n "$candidate" ] && COMPREPLY+=("$option $candidate")
    done <<EOF
$(_compute_flow_sample_selector_candidates "" "$brick_type" "$manifest_value")
EOF

    if [ ${#COMPREPLY[@]} -eq 0 ]; then
        COMPREPLY=("$option ")
    else
        _compute_flow_bash_keep_completing_directories
    fi
}

_compute_flow_brick_cli_complete_bash() {
    local current
    current="$(_compute_flow_bash_current_token)"
    local brick_type
    brick_type="$(_compute_flow_bash_brick_cli_type)"
    if [ -z "$brick_type" ]; then
        _compute_flow_bash_complete_files "$current"
        return 0
    fi

    if [ "$current" = "-m" ]; then
        _compute_flow_bash_complete_manifest_option "$brick_type"
        return 0
    fi

    if _compute_flow_sample_input_option "$current"; then
        _compute_flow_bash_complete_sample_option "$current" "$brick_type"
        return 0
    fi

    local IFS=$'\n'
    local candidates
    if _compute_flow_bash_completing_manifest_arg; then
        candidates="$(_compute_flow_manifest_selector_candidates "$current" "$brick_type")"
        if [ -n "$candidates" ]; then
            COMPREPLY=($(compgen -W "$candidates" -- "$current"))
            _compute_flow_bash_keep_completing_directories
        elif _compute_flow_token_allows_files "$current"; then
            _compute_flow_bash_complete_files "$current"
        else
            COMPREPLY=()
        fi
        return 0
    fi

    if _compute_flow_bash_completing_sample_arg; then
        local manifest_value
        manifest_value="$(_compute_flow_bash_manifest_value)"
        candidates="$(_compute_flow_sample_selector_candidates "$current" "$brick_type" "$manifest_value")"
        if [ -n "$candidates" ]; then
            COMPREPLY=($(compgen -W "$candidates" -- "$current"))
            _compute_flow_bash_keep_completing_directories
        elif _compute_flow_token_allows_files "$current"; then
            _compute_flow_bash_complete_files "$current"
        else
            COMPREPLY=()
        fi
        return 0
    fi

    _compute_flow_bash_complete_files "$current"
}

_compute_flow_zsh_completing_manifest_arg() {
    local index
    for ((index = 2; index < CURRENT; index++)); do
        if [ "${words[index]}" = "-m" ] && [ $((index + 1)) -eq "$CURRENT" ]; then
            return 0
        fi
    done
    return 1
}

_compute_flow_zsh_completing_sample_arg() {
    local index
    for ((index = 2; index < CURRENT; index++)); do
        if _compute_flow_sample_input_option "${words[index]}" && [ $((index + 1)) -eq "$CURRENT" ]; then
            return 0
        fi
    done
    return 1
}

_compute_flow_zsh_manifest_value() {
    local index value=""
    for ((index = 1; index < ${#words[@]}; index++)); do
        if [ "${words[index]}" = "-m" ]; then
            value="${words[index + 1]}"
        fi
    done
    printf '%s\n' "$value"
}

_compute_flow_zsh_brick_cli_type() {
    local index brick_type
    for ((index = 1; index < CURRENT; index++)); do
        brick_type="$(_compute_flow_brick_cli_type_from_word "${words[index]}")" || continue
        printf '%s\n' "$brick_type"
        return 0
    done
    return 1
}

_compute_flow_brick_cli_complete_zsh() {
    _compskip=all

    local current="${words[CURRENT]}"
    local brick_type
    brick_type="$(_compute_flow_zsh_brick_cli_type)"
    if [ -z "$brick_type" ]; then
        _files
        return
    fi

    if [ "$current" = "-m" ]; then
        local -a candidates
        local candidate
        while IFS= read -r candidate; do
            [ -n "$candidate" ] && candidates+=("-m $candidate")
        done <<EOF
$(_compute_flow_manifest_selector_candidates "" "$brick_type")
EOF
        if [ ${#candidates[@]} -gt 0 ]; then
            compadd -Q -S '' -- "${candidates[@]}"
        else
            compadd -S ' ' -- "-m"
        fi
        return
    fi

    if _compute_flow_sample_input_option "$current"; then
        local manifest_value
        manifest_value="$(_compute_flow_zsh_manifest_value)"
        local -a sample_options
        local sample_option
        while IFS= read -r sample_option; do
            [ -n "$sample_option" ] && sample_options+=("$current $sample_option")
        done <<EOF
$(_compute_flow_sample_selector_candidates "" "$brick_type" "$manifest_value")
EOF
        if [ ${#sample_options[@]} -gt 0 ]; then
            compadd -Q -S '' -- "${sample_options[@]}"
        else
            compadd -S ' ' -- "$current"
        fi
        return
    fi

    if _compute_flow_zsh_completing_manifest_arg; then
        local -a candidates
        local candidate
        while IFS= read -r candidate; do
            [ -n "$candidate" ] && candidates+=("$candidate")
        done <<EOF
$(_compute_flow_manifest_selector_candidates "$current" "$brick_type")
EOF
        if [ ${#candidates[@]} -gt 0 ]; then
            compadd -S '' -- "${candidates[@]}"
        elif _compute_flow_token_allows_files "$current"; then
            _files
        fi
        return
    fi

    if _compute_flow_zsh_completing_sample_arg; then
        local manifest_value
        manifest_value="$(_compute_flow_zsh_manifest_value)"
        local -a candidates
        local candidate
        while IFS= read -r candidate; do
            [ -n "$candidate" ] && candidates+=("$candidate")
        done <<EOF
$(_compute_flow_sample_selector_candidates "$current" "$brick_type" "$manifest_value")
EOF
        if [ ${#candidates[@]} -gt 0 ]; then
            compadd -S '' -- "${candidates[@]}"
        elif _compute_flow_token_allows_files "$current"; then
            _files
        fi
        return
    fi

    _files
}

_compute_flow_manager_complete_bash() {
    local current="${COMP_WORDS[COMP_CWORD]}"
    if { [ "${COMP_WORDS[1]:-}" = "update" ] ||
        [ "${COMP_WORDS[1]:-}" = "registry" ]; } &&
        [ "${COMP_WORDS[COMP_CWORD-1]:-}" = "--registry" ]; then
        _compute_flow_bash_complete_files "$current"
        return
    fi

    local -a candidate_descriptions candidate_values
    local candidate common_prefix description index
    local has_descriptions=0
    local max_candidate_width=0
    local output
    output="$(
        command compute-flow __complete --descriptions \
            "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null
    )"
    while IFS=$'\t' read -r candidate description; do
        [ -n "$candidate" ] || continue
        candidate_values+=("$candidate")
        candidate_descriptions+=("$description")
        (( ${#candidate} > max_candidate_width )) && max_candidate_width=${#candidate}
        [ -n "$description" ] && has_descriptions=1
    done <<< "$output"

    COMPREPLY=("${candidate_values[@]}")
    if (( has_descriptions && ${#COMPREPLY[@]} > 1 )) &&
        [ -z "$(trap -p WINCH)" ]; then
        common_prefix="${COMPREPLY[0]}"
        for candidate in "${COMPREPLY[@]:1}"; do
            while [[ "$candidate" != "$common_prefix"* ]]; do
                common_prefix="${common_prefix%?}"
            done
        done

        if [ "$common_prefix" = "$current" ]; then
            printf '\r\n' >&2
            for ((index = 0; index < ${#candidate_values[@]}; index++)); do
                printf '\e[1m%-*s\e[22m  %s\r\n' \
                    "$max_candidate_width" \
                    "${candidate_values[index]}" \
                    "${candidate_descriptions[index]}" >&2
            done
            COMPREPLY=("$current")
            builtin kill -WINCH "$$"
            return
        fi
    fi

    for ((index = 0; index < ${#COMPREPLY[@]}; index++)); do
        case "${COMPREPLY[index]}" in
            *:) ;;
            *) COMPREPLY[index]="${COMPREPLY[index]} " ;;
        esac
    done
}

_compute_flow_manager_complete_zsh() {
    local -a completion_words candidates descriptions
    local candidate description display padded_candidate
    local has_descriptions=0
    local max_candidate_width=0
    local output
    local index

    if [[ ( "${words[2]:-}" == "update" || "${words[2]:-}" == "registry" ) &&
        "${words[CURRENT-1]:-}" == "--registry" ]]; then
        _files
        return
    fi

    for ((index = 2; index <= CURRENT; index++)); do
        completion_words+=("${words[index]}")
    done
    output="$(
        command compute-flow __complete --descriptions \
            "${completion_words[@]}" 2>/dev/null
    )"
    [ -n "$output" ] || return
    while IFS=$'\t' read -r candidate description; do
        [ -n "$candidate" ] || continue
        candidates+=("$candidate")
        descriptions+=("$description")
        (( ${#candidate} > max_candidate_width )) && max_candidate_width=${#candidate}
        [ -n "$description" ] && has_descriptions=1
    done <<< "$output"

    if (( has_descriptions )); then
        local -a display_strings
        for ((index = 1; index <= ${#candidates[@]}; index++)); do
            printf -v padded_candidate "%-${max_candidate_width}s" "${candidates[index]}"
            display=$'\e[1m'"${padded_candidate}"$'\e[22m'"  ${descriptions[index]}"
            display_strings+=("$display")
        done

        compadd -S '' -V compute-flow-models -l -d display_strings -- "${candidates[@]}"
        return
    fi

    compadd -S '' -- "${candidates[@]}"
}

if [ -n "${BASH_VERSION:-}" ] && command -v complete >/dev/null 2>&1; then
    COMP_WORDBREAKS="${COMP_WORDBREAKS//:}"
    _compute_flow_bash_cli_options=()
    # Bash 3.2 has no compopt, so suppress spaces in the completion registration.
    if ! command -v compopt >/dev/null 2>&1; then
        _compute_flow_bash_cli_options=(-o nospace)
    fi

    complete "${_compute_flow_bash_cli_options[@]}" -D \
        -F _compute_flow_brick_cli_complete_bash 2>/dev/null || true
    complete -o nospace -F _compute_flow_manager_complete_bash compute-flow 2>/dev/null || true

    while IFS= read -r _compute_flow_command; do
        [ -n "$_compute_flow_command" ] || continue
        complete "${_compute_flow_bash_cli_options[@]}" \
            -F _compute_flow_brick_cli_complete_bash "$_compute_flow_command" 2>/dev/null || true
        complete "${_compute_flow_bash_cli_options[@]}" \
            -F _compute_flow_brick_cli_complete_bash "./$_compute_flow_command" 2>/dev/null || true
    done <<EOF
$(_compute_flow_completion_commands | sort -u)
EOF

    unset _compute_flow_bash_cli_options
fi

if [ -n "${ZSH_VERSION:-}" ]; then
    autoload -Uz compinit
    if ! command -v compdef >/dev/null 2>&1; then
        compinit -i
    fi

    if command -v compdef >/dev/null 2>&1; then
        while IFS= read -r _compute_flow_command; do
            [ -n "$_compute_flow_command" ] || continue
            compdef -d -p "*/$_compute_flow_command" 2>/dev/null || true
            compdef _compute_flow_brick_cli_complete_zsh "$_compute_flow_command"
            compdef _compute_flow_brick_cli_complete_zsh "./$_compute_flow_command"
        done <<EOF
$(_compute_flow_completion_commands | sort -u)
EOF
        compdef -p _compute_flow_brick_cli_complete_zsh "*_cli" "*/*_cli" "*_cli.py" "*/*_cli.py"
        compdef _compute_flow_manager_complete_zsh compute-flow
    fi
fi

unset _compute_flow_command
