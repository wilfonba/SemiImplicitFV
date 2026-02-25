#!/usr/bin/env bash
# Common variables and helpers shared by sifv subcommands.
# Sourced by sifv.sh — do not execute directly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CASES_DIR="$ROOT_DIR/cases"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TOGGLES_FILE="$ROOT_DIR/.sifv_toggles"

# ---- Persistent toggles ----

# Load stored toggles (sets TOGGLE_PETSC).
# Defaults to false if the file doesn't exist or a key is missing.
load_toggles() {
    TOGGLE_PETSC=false
    if [[ -f "$TOGGLES_FILE" ]]; then
        local key val
        while IFS='=' read -r key val; do
            key="${key// /}"
            val="${val// /}"
            case "$key" in
                petsc) [[ "$val" == "true" ]] && TOGGLE_PETSC=true ;;
            esac
        done < "$TOGGLES_FILE"
    fi
}

# Save current toggles to disk.
save_toggles() {
    cat > "$TOGGLES_FILE" <<EOF
petsc=$TOGGLE_PETSC
EOF
}

# Print active toggles for user awareness.
print_toggles() {
    local parts=()
    $TOGGLE_PETSC && parts+=("petsc")
    if [[ ${#parts[@]} -gt 0 ]]; then
        echo "[toggles: ${parts[*]}]"
    fi
}

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
