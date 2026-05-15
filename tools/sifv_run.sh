#!/usr/bin/env bash
# Build and run a simulation case.
# Sourced by sifv.sh — do not execute directly.

run_usage() {
    cat <<EOF
Usage: ./sifv.sh run [options] <case> [-- program args...]

Run a simulation case. <case> can be:
  - A case name:       looks up cases/<name>/<name>.jsonc (or .json / .cpp)
  - A path to a file:  any .jsonc or .json file (absolute or relative)

Each case can have:
  - A JSON input file:  cases/<name>/<name>.jsonc  (run via sifv generic driver)
  - A compiled source:  cases/<name>/*.cpp          (built and run directly)

If both exist, the JSON file is used by default. Use --case-optimization to
generate and compile an optimized C++ source from the JSON instead.

Each case runs in its own directory under cases/<case_name>/, where all output
(VTK files, logs) is written.

Options:
  -c, --clean             Clean rebuild (remove build directory first)
  -d, --debug             Build in Debug mode
  -N <nodes>              Number of nodes (default: 1)
  -n <ppn>                MPI ranks per node (default: 1)
  --build-only            Only build, do not run
  --compiled              Force using the compiled C++ source instead of JSON
  --case-optimization     Generate and compile a custom C++ main() from the
                          JSON input for maximum performance (codegen path)
  --petsc                 Enable PETSc pressure solver (saved across runs)
  --no-petsc              Disable PETSc pressure solver (saved across runs)
  --gpu                   Enable OpenMP target offload to NVIDIA GPUs (saved)
  --no-gpu                Disable GPU offload (saved)
  --gpu-cc <cc>           Target compute capability (e.g. cc80, cc90). Optional.
  --nsys                  Profile with Nsight Systems and print a post-run
                          summary (CUDA kernels with %-of-total, host<->device
                          memory transfers, and nested NVTX ranges).  Traces
                          cuda,nvtx,openmp,mpi,osrt with CUDA memory usage on.
  --srun                  Use srun instead of mpirun (for Slurm-managed systems)
  -j <N>                  Parallel build jobs (default: number of cores)
  -o, --output-dir <dir>  Override the run directory (default: cases/<case_name>)
  -h, --help              Show this help

Batch options (submit a job to a scheduler instead of running interactively):
  --batch                 Submit as a batch job instead of running interactively
  -a <account>            Scheduler account/project (required for --batch)
  -w <walltime>           Wall time limit, e.g. 01:00:00 (required for --batch)
  -p <partition>          Partition / queue name (optional)
  -q <qos>               QOS or queue (optional)
  -# <name>              Job name (default: case name); also used for .out/.err files
  --template <name>       Job template name from tools/templates/ (default: auto-detect)

Examples:
  ./sifv.sh run 1D_sod_shocktube                      # JSON case via sifv driver
  ./sifv.sh run /path/to/my_input.jsonc                # Run an arbitrary JSON file
  ./sifv.sh run ../other_dir/input.jsonc -o out/       # Arbitrary file, custom output dir
  ./sifv.sh run --case-optimization 1D_sod_shocktube   # JSON case via codegen
  ./sifv.sh run --compiled 1D_sod_shocktube            # compiled C++ source
  ./sifv.sh run -N 2 -n 4 2D_rising_bubble --srun     # 2 nodes, 4 ranks/node
  ./sifv.sh run -n 4 2D_rising_bubble                  # 1 node, 4 ranks
  ./sifv.sh run -d 1D_sod_shocktube
  ./sifv.sh run --petsc 2D_rising_bubble
  ./sifv.sh run --nsys 2D_rising_bubble

  # Batch submission (Slurm)
  ./sifv.sh run --batch -N 2 -n 24 -a MY_ACCOUNT -w 02:00:00 2D_rising_bubble
  ./sifv.sh run --batch -N 1 -n 4 -a MY_ACCOUNT -w 00:30:00 -p debug 1D_sod_shocktube
EOF
    exit "${1:-0}"
}

cmd_run() {
    load_toggles

    local CLEAN=false
    local BUILD_ONLY=false
    local ENABLE_PETSC=$TOGGLE_PETSC
    local ENABLE_GPU=${TOGGLE_GPU:-false}
    local GPU_CC=${TOGGLE_GPU_CC:-""}
    local ENABLE_NSYS=false
    local USE_SRUN=false
    local CASE_OPTIMIZATION=false
    local FORCE_COMPILED=false
    local NODES=1
    local PPN=1
    local JOBS
    JOBS="$(default_jobs)"
    local CASE_NAME=""
    local OUTPUT_DIR=""
    local PROGRAM_ARGS=()

    # Batch-specific
    local BATCH=false
    local BATCH_ACCOUNT=""
    local BATCH_WALLTIME=""
    local BATCH_PARTITION=""
    local BATCH_QOS=""
    local BATCH_JOB_NAME=""
    local BATCH_TEMPLATE=""

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
            -N)
                NODES="$2"
                shift 2
                ;;
            -N*)
                NODES="${1#-N}"
                shift
                ;;
            -n)
                PPN="$2"
                shift 2
                ;;
            -n*)
                PPN="${1#-n}"
                shift
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --case-optimization)
                CASE_OPTIMIZATION=true
                shift
                ;;
            --compiled)
                FORCE_COMPILED=true
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
            --srun)
                USE_SRUN=true
                shift
                ;;
            --batch)
                BATCH=true
                shift
                ;;
            -a)
                BATCH_ACCOUNT="$2"
                shift 2
                ;;
            -w)
                BATCH_WALLTIME="$2"
                shift 2
                ;;
            -p)
                BATCH_PARTITION="$2"
                shift 2
                ;;
            -q)
                BATCH_QOS="$2"
                shift 2
                ;;
            '-#')
                BATCH_JOB_NAME="$2"
                shift 2
                ;;
            --template)
                BATCH_TEMPLATE="$2"
                shift 2
                ;;
            -j)
                JOBS="$2"
                shift 2
                ;;
            -j*)
                JOBS="${1#-j}"
                shift
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                run_usage 0
                ;;
            --)
                shift
                PROGRAM_ARGS=("$@")
                break
                ;;
            -*)
                echo "Unknown option: $1" >&2
                run_usage 1
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

    # Compute total MPI ranks
    local MPI_RANKS=$((NODES * PPN))

    # Validate batch options
    if $BATCH; then
        if [[ -z "$BATCH_ACCOUNT" ]]; then
            echo "Error: --batch requires -a <account>" >&2
            exit 1
        fi
        if [[ -z "$BATCH_WALLTIME" ]]; then
            echo "Error: --batch requires -w <walltime>" >&2
            exit 1
        fi
    fi

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

    if [[ -z "$CASE_NAME" ]]; then
        echo "Error: no case name specified" >&2
        echo ""
        run_usage 1
    fi

    # --- Determine case type ---
    local JSON_FILE=""
    local COMPILED_DIR=""
    local BARE_NAME=""

    # Check if CASE_NAME is a path to an existing file
    if [[ -f "$CASE_NAME" && ("$CASE_NAME" == *.jsonc || "$CASE_NAME" == *.json) ]]; then
        # Direct file path provided
        JSON_FILE="$(cd "$(dirname "$CASE_NAME")" && pwd)/$(basename "$CASE_NAME")"
        BARE_NAME="$(basename "$CASE_NAME")"
        BARE_NAME="${BARE_NAME%.jsonc}"
        BARE_NAME="${BARE_NAME%.json}"
        # Default output directory to the file's parent directory
        if [[ -z "$OUTPUT_DIR" ]]; then
            OUTPUT_DIR="$(cd "$(dirname "$CASE_NAME")" && pwd)"
        fi
    else
        # Treat as a case name — look up in cases/
        BARE_NAME="${CASE_NAME%.jsonc}"
        BARE_NAME="${BARE_NAME%.json}"

        for ext in jsonc json; do
            if [[ -f "$CASES_DIR/$BARE_NAME/$BARE_NAME.$ext" ]]; then
                JSON_FILE="$CASES_DIR/$BARE_NAME/$BARE_NAME.$ext"
                break
            fi
        done

        if [[ -d "$CASES_DIR/$BARE_NAME" ]]; then
            if compgen -G "$CASES_DIR/$BARE_NAME"/*.cpp > /dev/null 2>&1; then
                COMPILED_DIR="$CASES_DIR/$BARE_NAME"
            fi
        fi

        if [[ -z "$JSON_FILE" && -z "$COMPILED_DIR" ]]; then
            echo "Error: case '$CASE_NAME' not found." >&2
            echo "  Looked for: cases/$BARE_NAME/$BARE_NAME.jsonc, cases/$BARE_NAME/$BARE_NAME.json, cases/$BARE_NAME/*.cpp" >&2
            echo ""
            list_cases
            exit 1
        fi
    fi

    local USE_JSON
    if $FORCE_COMPILED; then
        if [[ -z "$COMPILED_DIR" ]]; then
            echo "Error: --compiled requested but no compiled source found at cases/$BARE_NAME/" >&2
            exit 1
        fi
        USE_JSON=false
    elif $CASE_OPTIMIZATION; then
        if [[ -z "$JSON_FILE" ]]; then
            echo "Error: --case-optimization requires a JSON case file (cases/$BARE_NAME/$BARE_NAME.jsonc)" >&2
            exit 1
        fi
        USE_JSON=true
    elif [[ -n "$JSON_FILE" ]]; then
        USE_JSON=true
    else
        USE_JSON=false
    fi

    # --- Set up run directory ---
    # (OUTPUT_DIR may already be set by direct file path or -o flag)
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$CASES_DIR/$BARE_NAME"
    fi
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

    # --- Clean if requested ---
    $CLEAN && clean_build

    # --- Resolve cmake option values ---
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

    # --- Configure if needed ---
    if needs_configure \
        "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}" \
        "ENABLE_PETSC:BOOL=${PETSC_OPT}" \
        "ENABLE_NVTX:BOOL=${NVTX_OPT}" \
        "ENABLE_GPU_OFFLOAD:BOOL=${GPU_OPT}" \
        "GPU_OFFLOAD_CC:STRING=${GPU_CC}"; then

        # Also force reconfigure for compiled-source cases
        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT}, GPU=${GPU_OPT}${GPU_CC:+/${GPU_CC}})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            "${CMAKE_COMPILER_ARGS[@]}" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT" \
            -DENABLE_GPU_OFFLOAD="$GPU_OPT" \
            -DGPU_OFFLOAD_CC="$GPU_CC"
    elif ! $USE_JSON && [[ -n "$COMPILED_DIR" ]]; then
        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT}, GPU=${GPU_OPT}${GPU_CC:+/${GPU_CC}})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            "${CMAKE_COMPILER_ARGS[@]}" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT" \
            -DENABLE_GPU_OFFLOAD="$GPU_OPT" \
            -DGPU_OFFLOAD_CC="$GPU_CC"
    fi

    # --- Build (always, for both interactive and batch) ---
    local EXECUTABLE=""
    local EXEC_ARGS=""

    if $USE_JSON; then
        if $CASE_OPTIMIZATION; then
            _build_codegen "$BARE_NAME" "$JSON_FILE" "$JOBS" "$ENABLE_PETSC"
            EXECUTABLE="$BUILD_DIR/codegen_${BARE_NAME}"
            EXEC_ARGS="${PROGRAM_ARGS[*]+"${PROGRAM_ARGS[*]}"}"
        else
            echo "Building sifv driver..."
            cmake --build "$BUILD_DIR" --target sifv -j "$JOBS"
            EXECUTABLE="$BUILD_DIR/sifv"
            EXEC_ARGS="$JSON_FILE${PROGRAM_ARGS[*]+ ${PROGRAM_ARGS[*]}}"
        fi
    else
        local TARGET_NAME="$BARE_NAME"
        echo "Building $TARGET_NAME..."
        cmake --build "$BUILD_DIR" --target "$TARGET_NAME" -j "$JOBS"
        EXECUTABLE="$BUILD_DIR/$TARGET_NAME"
        EXEC_ARGS="${PROGRAM_ARGS[*]+"${PROGRAM_ARGS[*]}"}"
    fi

    if [[ "$BUILD_ONLY" == true ]]; then return 0; fi

    # --- Batch or interactive? ---
    if $BATCH; then
        local JOB_NAME="${BATCH_JOB_NAME:-$BARE_NAME}"
        _submit_batch "$JOB_NAME" "$NODES" "$PPN" "$MPI_RANKS" \
            "$BATCH_ACCOUNT" "$BATCH_WALLTIME" "$BATCH_PARTITION" "$BATCH_QOS" \
            "$BATCH_TEMPLATE" "$OUTPUT_DIR" "$EXECUTABLE $EXEC_ARGS"
    else
        _run_interactive "$BARE_NAME" "$OUTPUT_DIR" "$MPI_RANKS" "$NODES" "$PPN" \
            "$USE_SRUN" "$ENABLE_NSYS" "$EXECUTABLE" "$EXEC_ARGS"
    fi
}

# --- Internal helpers ---

_run_interactive() {
    local BARE_NAME="$1" OUTPUT_DIR="$2" MPI_RANKS="$3" NODES="$4" PPN="$5"
    local USE_SRUN="$6" ENABLE_NSYS="$7" EXECUTABLE="$8" EXEC_ARGS="$9"

    # Nsight Systems wrapper
    local NSYS_PREFIX=()
    local NSYS_REPORT=""
    if [[ "$ENABLE_NSYS" == true ]]; then
        if ! command -v nsys &>/dev/null; then
            echo "Error: nsys not found in PATH. Install NVIDIA Nsight Systems." >&2
            exit 1
        fi
        NSYS_REPORT="${OUTPUT_DIR}/${BARE_NAME}.nsys-rep"
        # Trace CUDA kernels + OpenMP target offload + NVTX + MPI so the
        # report includes per-kernel timing, host-device transfers, and
        # the nested NVTX ranges in the solver.
        NSYS_PREFIX=(nsys profile
            --trace=cuda,nvtx,openmp,mpi,osrt
            --cuda-memory-usage=true
            --gpu-metrics-devices=none
            --output="$NSYS_REPORT"
            --force-overwrite=true)
    fi

    # MPI launcher
    local MPI_LAUNCHER
    if [[ "$USE_SRUN" == true ]]; then
        MPI_LAUNCHER=(srun --unbuffered -N "$NODES" --ntasks-per-node="$PPN")
    else
        MPI_LAUNCHER=(mpirun -np "$MPI_RANKS")
    fi

    echo ""
    echo "=== Running $BARE_NAME with $MPI_RANKS MPI rank(s) ($NODES node(s) x $PPN ppn) ==="
    echo "=== Output directory: $OUTPUT_DIR ==="
    echo ""
    cd "$OUTPUT_DIR"
    # shellcheck disable=SC2086
    "${NSYS_PREFIX[@]+"${NSYS_PREFIX[@]}"}" \
        "${MPI_LAUNCHER[@]}" "$EXECUTABLE" $EXEC_ARGS
    local RUN_RC=$?

    if [[ "$ENABLE_NSYS" == true && -f "$NSYS_REPORT" ]]; then
        echo ""
        echo "=== Nsight Systems summary: $NSYS_REPORT ==="
        # cuda_gpu_kern_sum: per-kernel time with % of total kernel time.
        # cuda_gpu_mem_time_sum: host<->device memory transfer breakdown.
        # nvtx_sum: nested NVTX range timings (shows Reconstruction / etc.).
        nsys stats \
            --force-export=true \
            --report cuda_gpu_kern_sum \
            --report cuda_gpu_mem_time_sum \
            --report nvtx_sum \
            --format table \
            "$NSYS_REPORT" || true
        echo ""
        echo "Open in GUI:    nsys-ui $NSYS_REPORT"
        echo "More reports:   nsys stats --help-reports"
    fi

    return $RUN_RC
}

_build_codegen() {
    local BARE_NAME="$1" JSON_FILE="$2" JOBS="$3" ENABLE_PETSC="$4"

    local CODEGEN_DIR="$BUILD_DIR/codegen"
    mkdir -p "$CODEGEN_DIR"
    local GEN_SRC="$CODEGEN_DIR/${BARE_NAME}.cpp"
    local GEN_TARGET="codegen_${BARE_NAME}"

    echo "Generating optimized source from $JSON_FILE..."
    python3 "$ROOT_DIR/tools/codegen.py" "$JSON_FILE" -o "$GEN_SRC"

    echo "Building library and $GEN_TARGET (optimized)..."
    cmake --build "$BUILD_DIR" --target SemiImplicitFV -j "$JOBS"

    cd "$BUILD_DIR"
    local COMPILE_CMD
    COMPILE_CMD="$(grep -m1 'CXX_COMPILER' CMakeCache.txt | cut -d= -f2)"
    COMPILE_CMD="${COMPILE_CMD:-c++}"
    local INCLUDE_FLAGS="-I$ROOT_DIR/include"

    # Probe MPI wrappers in the order they actually exist on most systems.
    # NVHPC's bundled OpenMPI ships mpicxx/mpic++/mpicc but not mpiCC, so the
    # old `mpiCC --showme:` chain silently fell through to `-lmpi` which
    # nvc++'s own ld can't resolve.
    local MPI_CFLAGS=""
    local MPI_LDFLAGS=""
    for _wrap in mpicxx mpic++ mpiCC; do
        if command -v "$_wrap" >/dev/null 2>&1; then
            MPI_CFLAGS="$($_wrap --showme:compile 2>/dev/null || true)"
            MPI_LDFLAGS="$($_wrap --showme:link 2>/dev/null || true)"
            if [[ -n "$MPI_LDFLAGS" ]]; then break; fi
        fi
    done
    if [[ -z "$MPI_LDFLAGS" ]]; then
        MPI_CFLAGS="$(pkg-config --cflags ompi 2>/dev/null || true)"
        MPI_LDFLAGS="$(pkg-config --libs   ompi 2>/dev/null || echo "-lmpi")"
    fi

    # Pick up the OpenMP target-offload runtime when CMake is configured for
    # GPU offload — the codegen binary calls into the SemiImplicitFV library
    # which contains target regions and needs the matching -mp link.
    local GPU_FLAGS=""
    if [[ "$(grep -m1 ENABLE_GPU_OFFLOAD CMakeCache.txt | cut -d= -f2)" == "ON" ]]; then
        GPU_FLAGS="-mp=gpu"
        local _gpu_cc
        _gpu_cc="$(grep -m1 GPU_OFFLOAD_CC CMakeCache.txt | cut -d= -f2)"
        if [[ -n "$_gpu_cc" ]]; then
            GPU_FLAGS="$GPU_FLAGS -gpu=$_gpu_cc"
        fi
    fi

    local PETSC_CFLAGS=""
    local PETSC_LDFLAGS=""
    if [[ "$ENABLE_PETSC" == true ]]; then
        local PETSC_INSTALL="$BUILD_DIR/petsc-install"
        PETSC_CFLAGS="-DSIFV_HAS_PETSC -I$PETSC_INSTALL/include"
        PETSC_LDFLAGS="-L$PETSC_INSTALL/lib -lpetsc -lf2clapack -lf2cblas -lm -ldl"
    fi

    # shellcheck disable=SC2086
    "$COMPILE_CMD" -std=c++17 -O3 -march=native \
        $GPU_FLAGS \
        $INCLUDE_FLAGS \
        $PETSC_CFLAGS \
        $MPI_CFLAGS \
        "$GEN_SRC" \
        -L"$BUILD_DIR" -lSemiImplicitFV \
        $PETSC_LDFLAGS \
        $MPI_LDFLAGS \
        -o "$BUILD_DIR/$GEN_TARGET" \
        2>&1
    cd "$ROOT_DIR"
}
