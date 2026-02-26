#!/usr/bin/env bash
set -euo pipefail

# Source shared variables and helpers
TOOLS_DIR="$(cd "$(dirname "$0")/tools" && pwd)"
source "$TOOLS_DIR/sifv_common.sh"
source "$TOOLS_DIR/sifv_list.sh"
source "$TOOLS_DIR/sifv_run.sh"
source "$TOOLS_DIR/sifv_test.sh"

# ---- Top-level usage ----

usage() {
    cat <<EOF
Usage: ./sifv.sh <command> [options]

Commands:
  run   <case> [options]    Build and run a simulation case
  test  [options]           Build and run the test suite
  list                      List available cases

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
    run)
        cmd_run "$@"
        ;;
    test)
        cmd_test "$@"
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
