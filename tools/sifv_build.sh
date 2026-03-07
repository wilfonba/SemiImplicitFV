#!/usr/bin/env bash
# Build the sifv driver.
# Sourced by sifv.sh — do not execute directly.

build_usage() {
    cat <<EOF
Usage: ./sifv.sh build [options]

Build the sifv driver binary.

Options:
  -c, --clean       Clean rebuild (remove build directory first)
  -d, --debug       Build in Debug mode
  --petsc           Enable PETSc pressure solver (saved across runs)
  --no-petsc        Disable PETSc pressure solver (saved across runs)
  --nsys            Enable NVTX profiling ranges
  -j <N>            Parallel build jobs (default: number of cores)
  -h, --help        Show this help
EOF
    exit "${1:-0}"
}

cmd_build() {
    load_toggles

    local CLEAN=false
    local ENABLE_PETSC=$TOGGLE_PETSC
    local ENABLE_NSYS=false
    local JOBS
    JOBS="$(default_jobs)"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--clean)
                CLEAN=true
                shift
                ;;
            -d|--debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            --petsc)
                ENABLE_PETSC=true
                shift
                ;;
            --no-petsc)
                ENABLE_PETSC=false
                shift
                ;;
            --nsys)
                ENABLE_NSYS=true
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
                build_usage 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                build_usage 1
                ;;
        esac
    done

    # Save toggles if they changed
    if [[ "$ENABLE_PETSC" != "$TOGGLE_PETSC" ]]; then
        TOGGLE_PETSC=$ENABLE_PETSC
        save_toggles
    fi

    print_toggles

    # Clean if requested
    $CLEAN && clean_build

    # Resolve cmake option values
    local PETSC_OPT="OFF"
    local NVTX_OPT="OFF"
    $ENABLE_PETSC && PETSC_OPT="ON"
    $ENABLE_NSYS && NVTX_OPT="ON"

    # Configure if needed
    if needs_configure \
        "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}" \
        "ENABLE_PETSC:BOOL=${PETSC_OPT}" \
        "ENABLE_NVTX:BOOL=${NVTX_OPT}"; then

        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT"
    fi

    echo "Building sifv driver..."
    cmake --build "$BUILD_DIR" --target sifv -j "$JOBS"
    echo "Build complete: $BUILD_DIR/sifv"
}
