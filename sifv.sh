#!/usr/bin/env bash

# When sourced, only the "load" command is supported.  It evaluates the
# module commands in the caller's shell so that `source ./sifv.sh load -c
# phoenix` just works.  Everything else requires running as a script.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    _sifv_tools_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/tools" && pwd)"
    TOOLS_DIR="$_sifv_tools_dir"
    source "$_sifv_tools_dir/sifv_load.sh"

    if [[ "${1:-}" != "load" ]]; then
        echo "Error: only 'load' is supported when sourcing sifv.sh." >&2
        echo "  source ./sifv.sh load -c <compute>" >&2
        unset _sifv_tools_dir
        return 1
    fi
    shift
    _sifv_cmds="$(_get_compute_commands_from_args "$@")"
    _sifv_rc=$?
    if [[ $_sifv_rc -eq 0 && -n "$_sifv_cmds" ]]; then
        eval "$_sifv_cmds"
        _sifv_rc=$?
    fi
    unset _sifv_tools_dir _sifv_cmds TOOLS_DIR
    return "$_sifv_rc"
fi

set -euo pipefail

# Source shared variables and helpers
TOOLS_DIR="$(cd "$(dirname "$0")/tools" && pwd)"
source "$TOOLS_DIR/sifv_common.sh"
source "$TOOLS_DIR/sifv_list.sh"
source "$TOOLS_DIR/sifv_batch.sh"
source "$TOOLS_DIR/sifv_build.sh"
source "$TOOLS_DIR/sifv_load.sh"
source "$TOOLS_DIR/sifv_run.sh"
source "$TOOLS_DIR/sifv_test.sh"

# ---- Top-level usage ----

usage() {
    cat <<EOF
Usage: ./sifv.sh <command> [options]

Commands:
  build [options]            Build the sifv driver
  run   <case> [options]     Build and run a simulation case
  test  [options]            Build and run the test suite
  load  -c <compute>         Load modules for a compute environment
  list                       List available cases

Run "./sifv.sh run --help" or "./sifv.sh test --help" for command-specific options.
EOF
    exit "${1:-0}"
}

# ---- Main dispatch ----

if [[ $# -eq 0 ]]; then
    usage 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    build)
        cmd_build "$@"
        ;;
    run)
        cmd_run "$@"
        ;;
    test)
        cmd_test "$@"
        ;;
    load)
        cmd_load "$@"
        ;;
    list|-l|--list)
        list_cases
        ;;
    -h|--help|help)
        usage 0
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        echo ""
        usage 1
        ;;
esac
