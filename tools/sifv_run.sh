#!/usr/bin/env bash
# Build and run a simulation case.
# Sourced by sifv.sh — do not execute directly.

run_usage() {
    cat <<EOF
Usage: ./sifv.sh run [options] <case_name> [-- program args...]

Run a simulation case from cases/.

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
  -n <N>                  Number of MPI ranks (default: 1)
  --build-only            Only build, do not run
  --compiled              Force using the compiled C++ source instead of JSON
  --case-optimization     Generate and compile a custom C++ main() from the
                          JSON input for maximum performance (codegen path)
  --petsc                 Enable PETSc pressure solver (KSP+GAMG)
  --nsys                  Profile with Nsight Systems (enables NVTX ranges)
  --srun                  Use srun instead of mpirun (for Slurm-managed systems)
  -j <N>                  Parallel build jobs (default: number of cores)
  -o, --output-dir <dir>  Override the run directory (default: cases/<case_name>)
  -h, --help              Show this help

Examples:
  ./sifv.sh run 1D_sod_shocktube                      # JSON case via sifv driver
  ./sifv.sh run --case-optimization 1D_sod_shocktube   # JSON case via codegen
  ./sifv.sh run --compiled 1D_sod_shocktube            # compiled C++ source
  ./sifv.sh run -n 4 2D_rising_bubble
  ./sifv.sh run -d 1D_sod_shocktube
  ./sifv.sh run --petsc 2D_rising_bubble
  ./sifv.sh run --nsys 2D_rising_bubble
EOF
    exit "${1:-0}"
}

cmd_run() {
    local CLEAN=false
    local BUILD_ONLY=false
    local ENABLE_PETSC=false
    local ENABLE_NSYS=false
    local USE_SRUN=false
    local CASE_OPTIMIZATION=false
    local FORCE_COMPILED=false
    local MPI_RANKS=1
    local JOBS
    JOBS="$(default_jobs)"
    local CASE_NAME=""
    local OUTPUT_DIR=""
    local PROGRAM_ARGS=()

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
            --nsys)
                ENABLE_NSYS=true
                shift
                ;;
            --srun)
                USE_SRUN=true
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

    if [[ -z "$CASE_NAME" ]]; then
        echo "Error: no case name specified" >&2
        echo ""
        run_usage 1
    fi

    # --- Determine case type ---
    local JSON_FILE=""
    local COMPILED_DIR=""

    local BARE_NAME="${CASE_NAME%.jsonc}"
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
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$CASES_DIR/$BARE_NAME"
    fi
    mkdir -p "$OUTPUT_DIR"

    # --- Clean if requested ---
    $CLEAN && clean_build

    # --- Resolve cmake option values ---
    local PETSC_OPT="OFF"
    local NVTX_OPT="OFF"
    $ENABLE_PETSC && PETSC_OPT="ON"
    $ENABLE_NSYS && NVTX_OPT="ON"

    # --- Configure if needed ---
    if needs_configure \
        "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}" \
        "ENABLE_PETSC:BOOL=${PETSC_OPT}" \
        "ENABLE_NVTX:BOOL=${NVTX_OPT}"; then

        # Also force reconfigure for compiled-source cases
        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT"
    elif ! $USE_JSON && [[ -n "$COMPILED_DIR" ]]; then
        echo "Configuring (${BUILD_TYPE}, PETSC=${PETSC_OPT}, NVTX=${NVTX_OPT})..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DENABLE_PETSC="$PETSC_OPT" \
            -DENABLE_NVTX="$NVTX_OPT"
    fi

    # --- Nsight Systems wrapper ---
    local NSYS_PREFIX=()
    if $ENABLE_NSYS; then
        if ! command -v nsys &>/dev/null; then
            echo "Error: nsys not found in PATH. Install NVIDIA Nsight Systems." >&2
            exit 1
        fi
        NSYS_PREFIX=(nsys profile --trace=mpi,nvtx --output="${OUTPUT_DIR}/${BARE_NAME}.nsys-rep" --force-overwrite=true)
    fi

    # --- MPI launcher ---
    local MPI_LAUNCHER
    if $USE_SRUN; then
        MPI_LAUNCHER=(srun -n "$MPI_RANKS")
    else
        MPI_LAUNCHER=(mpirun -np "$MPI_RANKS")
    fi

    # --- Build and run ---
    if $USE_JSON; then
        if $CASE_OPTIMIZATION; then
            _run_codegen "$BARE_NAME" "$JSON_FILE" "$OUTPUT_DIR" "$JOBS" "$BUILD_ONLY" \
                "$ENABLE_PETSC" "$MPI_RANKS" \
                NSYS_PREFIX MPI_LAUNCHER PROGRAM_ARGS
        else
            _run_json_driver "$BARE_NAME" "$JSON_FILE" "$OUTPUT_DIR" "$JOBS" "$BUILD_ONLY" \
                "$MPI_RANKS" \
                NSYS_PREFIX MPI_LAUNCHER PROGRAM_ARGS
        fi
    else
        _run_compiled "$BARE_NAME" "$OUTPUT_DIR" "$JOBS" "$BUILD_ONLY" "$MPI_RANKS" \
            NSYS_PREFIX MPI_LAUNCHER PROGRAM_ARGS
    fi
}

# --- Internal helpers ---

_run_json_driver() {
    local BARE_NAME="$1" JSON_FILE="$2" OUTPUT_DIR="$3" JOBS="$4" BUILD_ONLY="$5"
    local MPI_RANKS="$6"
    # Array arguments passed by name
    local -n _NSYS_PREFIX="$7" _MPI_LAUNCHER="$8" _PROGRAM_ARGS="$9"

    echo "Building sifv driver..."
    cmake --build "$BUILD_DIR" --target sifv -j "$JOBS"

    if [[ "$BUILD_ONLY" == true ]]; then return 0; fi

    echo ""
    echo "=== Running $BARE_NAME via sifv driver with $MPI_RANKS MPI rank(s) ==="
    echo "=== Output directory: $OUTPUT_DIR ==="
    echo ""
    cd "$OUTPUT_DIR"
    "${_NSYS_PREFIX[@]+"${_NSYS_PREFIX[@]}"}" \
        "${_MPI_LAUNCHER[@]}" "$BUILD_DIR/sifv" "$JSON_FILE" \
        "${_PROGRAM_ARGS[@]+"${_PROGRAM_ARGS[@]}"}"
}

_run_codegen() {
    local BARE_NAME="$1" JSON_FILE="$2" OUTPUT_DIR="$3" JOBS="$4" BUILD_ONLY="$5"
    local ENABLE_PETSC="$6" MPI_RANKS="$7"
    local -n _NSYS_PREFIX="$8" _MPI_LAUNCHER="$9" _PROGRAM_ARGS="${10}"

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
    local MPI_CFLAGS
    MPI_CFLAGS="$(pkg-config --cflags ompi 2>/dev/null || mpiCC --showme:compile 2>/dev/null || true)"
    local MPI_LDFLAGS
    MPI_LDFLAGS="$(pkg-config --libs ompi 2>/dev/null || mpiCC --showme:link 2>/dev/null || echo "-lmpi")"

    local PETSC_CFLAGS=""
    local PETSC_LDFLAGS=""
    if [[ "$ENABLE_PETSC" == true ]]; then
        local PETSC_INSTALL="$BUILD_DIR/petsc-install"
        PETSC_CFLAGS="-DSIFV_HAS_PETSC -I$PETSC_INSTALL/include"
        PETSC_LDFLAGS="-L$PETSC_INSTALL/lib -lpetsc -lf2clapack -lf2cblas -lm -ldl"
    fi

    "$COMPILE_CMD" -std=c++17 -O3 -march=native \
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

    if [[ "$BUILD_ONLY" == true ]]; then return 0; fi

    echo ""
    echo "=== Running $BARE_NAME (codegen optimized) with $MPI_RANKS MPI rank(s) ==="
    echo "=== Output directory: $OUTPUT_DIR ==="
    echo ""
    cd "$OUTPUT_DIR"
    "${_NSYS_PREFIX[@]+"${_NSYS_PREFIX[@]}"}" \
        "${_MPI_LAUNCHER[@]}" "$BUILD_DIR/$GEN_TARGET" \
        "${_PROGRAM_ARGS[@]+"${_PROGRAM_ARGS[@]}"}"
}

_run_compiled() {
    local BARE_NAME="$1" OUTPUT_DIR="$2" JOBS="$3" BUILD_ONLY="$4" MPI_RANKS="$5"
    local -n _NSYS_PREFIX="$6" _MPI_LAUNCHER="$7" _PROGRAM_ARGS="$8"

    local TARGET_NAME="$BARE_NAME"

    echo "Building $TARGET_NAME..."
    cmake --build "$BUILD_DIR" --target "$TARGET_NAME" -j "$JOBS"

    if [[ "$BUILD_ONLY" == true ]]; then return 0; fi

    echo ""
    echo "=== Running $TARGET_NAME with $MPI_RANKS MPI rank(s) ==="
    echo "=== Output directory: $OUTPUT_DIR ==="
    echo ""
    cd "$OUTPUT_DIR"
    "${_NSYS_PREFIX[@]+"${_NSYS_PREFIX[@]}"}" \
        "${_MPI_LAUNCHER[@]}" "$BUILD_DIR/$TARGET_NAME" \
        "${_PROGRAM_ARGS[@]+"${_PROGRAM_ARGS[@]}"}"
}
