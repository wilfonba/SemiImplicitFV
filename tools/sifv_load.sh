#!/usr/bin/env bash
# Print module commands for a compute environment.
# Sourced by sifv.sh — do not execute directly.

COMPUTE_CONF="$TOOLS_DIR/compute.conf"

load_usage() {
    cat <<EOF
Usage: ./sifv.sh load -c <compute>

Load modules for a compute environment defined in tools/compute.conf.

  source ./sifv.sh load -c phoenix

Options:
  -c <name>    Compute environment name (required)
  -l, --list   List available compute environments
  -h, --help   Show this help

Available environments:
$(list_compute_envs)
EOF
    exit "${1:-0}"
}

# List all environment names from compute.conf
list_compute_envs() {
    if [[ ! -f "$COMPUTE_CONF" ]]; then
        echo "  (none — $COMPUTE_CONF not found)"
        return
    fi
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            echo "  ${BASH_REMATCH[1]}"
        fi
    done < "$COMPUTE_CONF"
}

# Print the module commands for a given environment name.
# Returns 1 if the environment is not found.
_get_compute_commands() {
    local name="$1"
    local in_section=false

    if [[ ! -f "$COMPUTE_CONF" ]]; then
        echo "Error: $COMPUTE_CONF not found" >&2
        return 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and blank lines outside sections
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$name" ]]; then
                in_section=true
            else
                $in_section && return 0
            fi
            continue
        fi
        if $in_section; then
            # Skip blank lines, comments, and key=value metadata lines
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            [[ "$line" =~ ^[[:space:]]*[a-zA-Z_]+[[:space:]]*= ]] && continue
            echo "$line"
        fi
    done < "$COMPUTE_CONF"

    if ! $in_section; then
        echo "Error: compute environment '$name' not found in $COMPUTE_CONF" >&2
        echo "Available environments:" >&2
        list_compute_envs >&2
        return 1
    fi
}

# Read a key=value field from a compute environment section.
# Usage: _get_compute_value <env_name> <key>
# Prints the value to stdout. Returns 1 if not found.
_get_compute_value() {
    local name="$1" key="$2"
    local in_section=false

    [[ ! -f "$COMPUTE_CONF" ]] && return 1

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$name" ]]; then
                in_section=true
            else
                $in_section && return 1
            fi
            continue
        fi
        if $in_section && [[ "$line" =~ ^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    done < "$COMPUTE_CONF"
    return 1
}

# Parse -c <name> from args and print the compute commands.
# Used by both cmd_load (script mode) and the sourced path in sifv.sh.
_get_compute_commands_from_args() {
    local COMPUTE_NAME=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c)
                COMPUTE_NAME="$2"
                shift 2
                ;;
            -l|--list)
                echo "Available compute environments:" >&2
                list_compute_envs >&2
                return 0
                ;;
            -h|--help)
                load_usage 0
                return 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                return 1
                ;;
        esac
    done

    if [[ -z "$COMPUTE_NAME" ]]; then
        echo "Error: -c <compute> is required" >&2
        return 1
    fi

    _get_compute_commands "$COMPUTE_NAME"
}

cmd_load() {
    _get_compute_commands_from_args "$@"
}
