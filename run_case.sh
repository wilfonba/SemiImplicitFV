#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
EXAMPLES_DIR="$ROOT_DIR/examples"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CMAKE_ARGS=()

usage() {
    cat <<EOF
Usage: ./run_case.sh [options] <case_name> [-- program args...]

Build and run a case from examples/<case_name>/.

Options:
  -l, --list        List available cases
  -c, --clean       Clean rebuild (remove build directory first)
  -d, --debug       Build in Debug mode
  --mpi             Enable MPI (adds -DENABLE_MPI=ON and runs with mpirun)
  -n <N>            Number of MPI ranks (default: 2, implies --mpi)
  --build-only      Only build, do not run
  -j <N>            Parallel build jobs (default: number of cores)
  -h, --help        Show this help

Examples:
  ./run_case.sh 1D_sod_shocktube
  ./run_case.sh --debug 1D_advection
  ./run_case.sh --mpi -n 4 1D_sod_shocktube
  ./run_case.sh -j4 1D_sod_shocktube -- --some-arg
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
USE_MPI=false
MPI_RANKS=2
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
        --mpi)
            USE_MPI=true
            shift
            ;;
        -n)
            MPI_RANKS="$2"
            USE_MPI=true
            shift 2
            ;;
        -n*)
            MPI_RANKS="${1#-n}"
            USE_MPI=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
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
                echo "Error: multiple case names given" >&2
                usage 1
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

# --- MPI setup ---
if $USE_MPI; then
    CMAKE_ARGS+=("-DENABLE_MPI=ON")
else
    CMAKE_ARGS+=("-DENABLE_MPI=OFF")
fi

# --- Clean if requested ---
if $CLEAN && [[ -d "$BUILD_DIR" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# --- Configure (or reconfigure if MPI setting changed) ---
NEED_CONFIGURE=false
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    NEED_CONFIGURE=true
elif $USE_MPI && ! grep -q "ENABLE_MPI:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    NEED_CONFIGURE=true
elif ! $USE_MPI && grep -q "ENABLE_MPI:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    NEED_CONFIGURE=true
fi

if $NEED_CONFIGURE; then
    echo "Configuring (${BUILD_TYPE}, MPI=$(if $USE_MPI; then echo ON; else echo OFF; fi))..."
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        "${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"}"
fi

# --- Build just this target ---
echo "Building $CASE_NAME..."
cmake --build "$BUILD_DIR" --target "$CASE_NAME" -j "$JOBS"

# --- Run from the case directory so output files land there ---
if ! "$BUILD_ONLY"; then
    echo ""
    if $USE_MPI; then
        echo "=== Running $CASE_NAME with $MPI_RANKS MPI ranks ==="
    else
        echo "=== Running $CASE_NAME ==="
    fi
    echo ""
    cd "$CASE_DIR"
    if $USE_MPI; then
        mpirun -np "$MPI_RANKS" "$BUILD_DIR/$CASE_NAME" "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
    else
        "$BUILD_DIR/$CASE_NAME" "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
    fi
fi
