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
  --gpu             Enable OpenMP target offload to NVIDIA GPUs (saved)
  --no-gpu          Disable GPU offload (saved)
  --gpu-cc <cc>     Target compute capability (e.g. cc80, cc90). Optional.
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
    local ENABLE_GPU=${TOGGLE_GPU:-false}
    local GPU_CC=${TOGGLE_GPU_CC:-""}
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
            --gpu)
                ENABLE_GPU=true
                shift
                ;;
            --no-gpu)
                ENABLE_GPU=false
                shift
                ;;
            --gpu-cc)
                GPU_CC="$2"
                shift 2
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
    local toggles_changed=false
    if [[ "$ENABLE_PETSC" != "$TOGGLE_PETSC" ]]; then
        TOGGLE_PETSC=$ENABLE_PETSC
        toggles_changed=true
    fi
    if [[ "$ENABLE_GPU" != "${TOGGLE_GPU:-false}" ]]; then
        TOGGLE_GPU=$ENABLE_GPU
        toggles_changed=true
    fi
    if [[ "$GPU_CC" != "${TOGGLE_GPU_CC:-}" ]]; then
        TOGGLE_GPU_CC=$GPU_CC
        toggles_changed=true
    fi
    $toggles_changed && save_toggles

    print_toggles

    # Clean if requested
    $CLEAN && clean_build

    # Resolve cmake option values
    local PETSC_OPT="OFF"
    local GPU_OPT="OFF"
    local NVTX_OPT="OFF"
    $ENABLE_PETSC && PETSC_OPT="ON"
    $ENABLE_GPU && GPU_OPT="ON"
    $ENABLE_NSYS && NVTX_OPT="ON"

    # When GPU offload is on, force the NVHPC compiler — PETSc is not ported
    # to it here, so turn that off automatically.
    local CMAKE_COMPILER_ARGS=()
    if $ENABLE_GPU; then
        if command -v nvc++ >/dev/null 2>&1; then
            CMAKE_COMPILER_ARGS+=("-DCMAKE_CXX_COMPILER=nvc++")
        else
            echo "Error: --gpu requested but nvc++ not on PATH. Did you 'source ./sifv.sh load -c phoenix-gpu'?" >&2
            return 1
        fi
        if [[ "$PETSC_OPT" == "ON" ]]; then
            echo "--gpu: disabling PETSc for this build (not supported with nvc++ here)"
            PETSC_OPT="OFF"
        fi
    fi

    # Configure if needed
    if needs_configure \
        "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}" \
        "ENABLE_PETSC:BOOL=${PETSC_OPT}" \
        "ENABLE_NVTX:BOOL=${NVTX_OPT}" \
        "ENABLE_GPU_OFFLOAD:BOOL=${GPU_OPT}" \
        "GPU_OFFLOAD_CC:STRING=${GPU_CC}"; then

        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT}, GPU=${GPU_OPT}${GPU_CC:+/${GPU_CC}})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            "${CMAKE_COMPILER_ARGS[@]}" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT" \
            -DENABLE_GPU_OFFLOAD="$GPU_OPT" \
            -DGPU_OFFLOAD_CC="$GPU_CC"
    fi

    echo "Building sifv driver..."
    cmake --build "$BUILD_DIR" --target sifv -j "$JOBS"
    echo "Build complete: $BUILD_DIR/sifv"
}
