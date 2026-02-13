#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
EXAMPLES_DIR="$ROOT_DIR/examples"
BUILD_TYPE="${BUILD_TYPE:-Release}"

usage() {
    cat <<EOF
Usage: ./run_case.sh [options] <case_name> [-- program args...]

Build and run a case from examples/<case_name>/.

Options:
  -l, --list        List available cases
  -c, --clean       Clean rebuild (remove build directory first)
  -d, --debug       Build in Debug mode
  -n <N>            Number of MPI ranks (default: 1)
  --build-only      Only build, do not run
  --hypre           Enable Hypre pressure solver (PCG+PFMG)
  -j <N>            Parallel build jobs (default: number of cores)
  -h, --help        Show this help

Examples:
  ./run_case.sh 1D_sod_shocktube
  ./run_case.sh --debug 1D_advection
  ./run_case.sh -n 4 1D_sod_shocktube
  ./run_case.sh -j4 1D_sod_shocktube -- --some-arg
  ./run_case.sh --hypre 2D_rising_bubble -- --semi-implicit --hypre
EOF
    exit "${1:-0}"
}

list_cases() {
    echo "Available cases:"
    for dir in "$EXAMPLES_DIR"/*/; do
        name="$(basename "$dir")"
        # Check that the directory has at least one .cpp file
        if compgen -G "$dir"/*.cpp > /dev/null 2>&1; then
            echo "  $name"
        fi
    done
}

# --- Parse arguments ---
CLEAN=false
BUILD_ONLY=false
ENABLE_HYPRE=false
MPI_RANKS=1
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
CASE_NAME=""
PROGRAM_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--list)
            list_cases
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -n)
            MPI_RANKS="$2"
            shift 2
            ;;
        -n*)
            MPI_RANKS="${1#-n}"
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --hypre)
            ENABLE_HYPRE=true
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        --)
            shift
            PROGRAM_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
        *)
            if [[ -z "$CASE_NAME" ]]; then
                CASE_NAME="$1"
            else
                PROGRAM_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$CASE_NAME" ]]; then
    echo "Error: no case name specified" >&2
    echo ""
    usage 1
fi

CASE_DIR="$EXAMPLES_DIR/$CASE_NAME"
if [[ ! -d "$CASE_DIR" ]]; then
    echo "Error: case '$CASE_NAME' not found at $CASE_DIR" >&2
    echo ""
    list_cases
    exit 1
fi

# --- Clean if requested ---
if $CLEAN && [[ -d "$BUILD_DIR" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# --- Resolve cmake option values ---
if $ENABLE_HYPRE; then
    HYPRE_OPT="ON"
else
    HYPRE_OPT="OFF"
fi

# --- Configure if needed or if build type / options changed ---
NEED_CONFIGURE=false
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    NEED_CONFIGURE=true
elif ! grep -q "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}$" "$BUILD_DIR/CMakeCache.txt"; then
    NEED_CONFIGURE=true
elif ! grep -q "ENABLE_HYPRE:BOOL=${HYPRE_OPT}$" "$BUILD_DIR/CMakeCache.txt"; then
    NEED_CONFIGURE=true
fi

if $NEED_CONFIGURE; then
    echo "Configuring (${BUILD_TYPE}, HYPRE=${HYPRE_OPT})..."
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DENABLE_HYPRE="$HYPRE_OPT"
fi

# --- Build just this target ---
echo "Building $CASE_NAME..."
cmake --build "$BUILD_DIR" --target "$CASE_NAME" -j "$JOBS"

# --- Run from the case directory so output files land there ---
if ! "$BUILD_ONLY"; then
    echo ""
    echo "=== Running $CASE_NAME with $MPI_RANKS MPI rank(s) ==="
    echo ""
    cd "$CASE_DIR"
    mpirun -np "$MPI_RANKS" "$BUILD_DIR/$CASE_NAME" "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
fi
