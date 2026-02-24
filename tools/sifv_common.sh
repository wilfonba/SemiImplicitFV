#!/usr/bin/env bash
# Common variables and helpers shared by sifv subcommands.
# Sourced by sifv.sh — do not execute directly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CASES_DIR="$ROOT_DIR/cases"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Default parallel job count
default_jobs() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

# Clean build directory
clean_build() {
    if [[ -d "$BUILD_DIR" ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
}

# Check if cmake needs to reconfigure.
# Usage: needs_configure KEY1=VAL1 KEY2=VAL2 ...
# Returns 0 (true) if reconfiguration is needed.
needs_configure() {
    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        return 0
    fi
    for kv in "$@"; do
        if ! grep -q "${kv}$" "$BUILD_DIR/CMakeCache.txt"; then
            return 0
        fi
    done
    return 1
}
