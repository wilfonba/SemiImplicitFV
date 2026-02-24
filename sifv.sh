#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CASES_DIR="$ROOT_DIR/cases"
BUILD_TYPE="${BUILD_TYPE:-Release}"

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

# ---- Case listing ----

list_cases() {
    echo "Cases in cases/:"
    echo ""

    declare -A cases
    for dir in "$CASES_DIR"/*/; do
        [[ -d "$dir" ]] || continue
        name="$(basename "$dir")"
        for ext in jsonc json; do
            if [[ -f "$dir/$name.$ext" ]]; then
                cases[$name]="${cases[$name]:-}json "
                break
            fi
        done
        if compgen -G "$dir"/*.cpp > /dev/null 2>&1; then
            cases[$name]="${cases[$name]:-}compiled "
        fi
    done

    for name in $(echo "${!cases[@]}" | tr ' ' '\n' | sort); do
        types="${cases[$name]}"
        tags=""
        [[ "$types" == *json* ]] && tags="${tags}json"
        [[ "$types" == *compiled* ]] && { [[ -n "$tags" ]] && tags="${tags}, "; tags="${tags}compiled"; }
        printf "  %-30s [%s]\n" "$name" "$tags"
    done
}

# ---- Run command ----

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
    JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
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
    if $CLEAN && [[ -d "$BUILD_DIR" ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi

    # --- Resolve cmake option values ---
    local PETSC_OPT="OFF"
    local NVTX_OPT="OFF"
    $ENABLE_PETSC && PETSC_OPT="ON"
    $ENABLE_NSYS && NVTX_OPT="ON"

    # --- Configure if needed ---
    local NEED_CONFIGURE=false
    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        NEED_CONFIGURE=true
    elif ! grep -q "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}$" "$BUILD_DIR/CMakeCache.txt"; then
        NEED_CONFIGURE=true
    elif ! grep -q "ENABLE_PETSC:BOOL=${PETSC_OPT}$" "$BUILD_DIR/CMakeCache.txt"; then
        NEED_CONFIGURE=true
    elif ! grep -q "ENABLE_NVTX:BOOL=${NVTX_OPT}$" "$BUILD_DIR/CMakeCache.txt"; then
        NEED_CONFIGURE=true
    elif ! $USE_JSON && [[ -n "$COMPILED_DIR" ]]; then
        NEED_CONFIGURE=true
    fi

    if $NEED_CONFIGURE; then
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
            # ---- Code generation path ----
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
            if $ENABLE_PETSC; then
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

            if ! $BUILD_ONLY; then
                echo ""
                echo "=== Running $BARE_NAME (codegen optimized) with $MPI_RANKS MPI rank(s) ==="
                echo "=== Output directory: $OUTPUT_DIR ==="
                echo ""
                cd "$OUTPUT_DIR"
                "${NSYS_PREFIX[@]+"${NSYS_PREFIX[@]}"}" \
                    "${MPI_LAUNCHER[@]}" "$BUILD_DIR/$GEN_TARGET" \
                    "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
            fi
        else
            # ---- Generic driver path ----
            echo "Building sifv driver..."
            cmake --build "$BUILD_DIR" --target sifv -j "$JOBS"

            if ! $BUILD_ONLY; then
                echo ""
                echo "=== Running $BARE_NAME via sifv driver with $MPI_RANKS MPI rank(s) ==="
                echo "=== Output directory: $OUTPUT_DIR ==="
                echo ""
                cd "$OUTPUT_DIR"
                "${NSYS_PREFIX[@]+"${NSYS_PREFIX[@]}"}" \
                    "${MPI_LAUNCHER[@]}" "$BUILD_DIR/sifv" "$JSON_FILE" \
                    "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
            fi
        fi
    else
        # ---- Compiled source path ----
        local TARGET_NAME="$BARE_NAME"

        echo "Building $TARGET_NAME..."
        cmake --build "$BUILD_DIR" --target "$TARGET_NAME" -j "$JOBS"

        if ! $BUILD_ONLY; then
            echo ""
            echo "=== Running $TARGET_NAME with $MPI_RANKS MPI rank(s) ==="
            echo "=== Output directory: $OUTPUT_DIR ==="
            echo ""
            cd "$OUTPUT_DIR"
            "${NSYS_PREFIX[@]+"${NSYS_PREFIX[@]}"}" \
                "${MPI_LAUNCHER[@]}" "$BUILD_DIR/$TARGET_NAME" \
                "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
        fi
    fi
}

# ---- Test command ----

test_usage() {
    cat <<EOF
Usage: ./sifv.sh test [options] [tier...]

Build and run the test suite.

Tiers:
  unit            Unit tests (EOS, Riemann solvers, reconstruction, etc.)
  integration     Integration tests (BCs, state conversion, solvers)
  regression      Regression tests (50-step runs vs reference data, np=1 and np=4)

If no tier is specified, all tests are run.

Options:
  -c, --clean             Clean rebuild (remove build directory first)
  -d, --debug             Build in Debug mode
  -j <N>                  Parallel build and test jobs (default: number of cores)
  -l, --list              List all test names
  -o <pattern>            Run only tests matching <pattern> (regex, passed to ctest -R)
  --generate              Regenerate regression reference data; use with -o to target
                          a specific test (e.g. -o TaylorGreenVortex3D --generate)
  --build-only            Only build test executables, do not run
  -h, --help              Show this help

Examples:
  ./sifv.sh test                                      # Run all tests
  ./sifv.sh test -j 8                                 # Build and test with 8 parallel jobs
  ./sifv.sh test unit                                 # Unit tests only
  ./sifv.sh test integration regression               # Integration and regression tests
  ./sifv.sh test -l                                   # List all test names
  ./sifv.sh test -o EOS                               # Run tests matching "EOS"
  ./sifv.sh test -o TaylorGreenVortex3D               # Run a specific test
  ./sifv.sh test --generate                           # Regenerate all regression references
  ./sifv.sh test -o TaylorGreenVortex3D --generate    # Regenerate one reference
  ./sifv.sh test -d unit                              # Debug build, unit tests only
EOF
    exit "${1:-0}"
}

cmd_test() {
    local CLEAN=false
    local BUILD_ONLY=false
    local GENERATE_REFS=false
    local LIST_TESTS=false
    local TEST_PATTERN=""
    local JOBS
    JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    local TIERS=()

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
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --generate-references|--generate)
                GENERATE_REFS=true
                shift
                ;;
            -l|--list)
                LIST_TESTS=true
                shift
                ;;
            -o)
                TEST_PATTERN="$2"
                shift 2
                ;;
            -o*)
                TEST_PATTERN="${1#-o}"
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
                test_usage 0
                ;;
            -*)
                echo "Unknown option: $1" >&2
                test_usage 1
                ;;
            *)
                TIERS+=("$1")
                shift
                ;;
        esac
    done

    # -o and tier selection are mutually exclusive
    if [[ -n "$TEST_PATTERN" && ${#TIERS[@]} -gt 0 ]]; then
        echo "Error: -o <pattern> and tier names cannot be used together" >&2
        exit 1
    fi

    # Default: all tiers (only when not using -o or -l)
    if [[ ${#TIERS[@]} -eq 0 && -z "$TEST_PATTERN" && "$LIST_TESTS" == false && "$GENERATE_REFS" == false ]]; then
        TIERS=(unit integration regression)
    fi

    # Validate tier names
    for tier in "${TIERS[@]}"; do
        case "$tier" in
            unit|integration|regression) ;;
            *)
                echo "Error: unknown test tier '$tier'" >&2
                echo "Valid tiers: unit, integration, regression" >&2
                exit 1
                ;;
        esac
    done

    # --- Clean if requested ---
    if $CLEAN && [[ -d "$BUILD_DIR" ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi

    # --- Configure with tests enabled ---
    local NEED_CONFIGURE=false
    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        NEED_CONFIGURE=true
    elif ! grep -q "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}$" "$BUILD_DIR/CMakeCache.txt"; then
        NEED_CONFIGURE=true
    elif ! grep -q "BUILD_TESTS:BOOL=ON$" "$BUILD_DIR/CMakeCache.txt"; then
        NEED_CONFIGURE=true
    fi

    if $NEED_CONFIGURE; then
        echo "Configuring (${BUILD_TYPE}, tests enabled)..."
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DBUILD_TESTS=ON \
            -DBUILD_DRIVER=ON
    fi

    # --- Build ---
    echo "Building tests..."
    cmake --build "$BUILD_DIR" -j "$JOBS"

    if $BUILD_ONLY; then
        echo "Build complete (--build-only)."
        return 0
    fi

    # --- List tests if requested ---
    if $LIST_TESTS; then
        echo ""
        echo "Available tests:"
        echo ""
        cd "$BUILD_DIR"
        ctest --show-only -R . 2>&1 | sed -n 's/.*Test #[0-9]*: //p' | while read -r name; do
            printf "  %s\n" "$name"
        done
        return 0
    fi

    # --- Generate references if requested ---
    if $GENERATE_REFS; then
        echo ""
        cd "$BUILD_DIR"
        if [[ -n "$TEST_PATTERN" ]]; then
            echo "=== Generating regression reference data for: $TEST_PATTERN ==="
            echo ""
            OMPI_MCA_rmaps_base_oversubscribe=1 GENERATE_REFERENCES=1 \
                "${MPIEXEC:-mpirun}" -np 1 ./tests/test_regression \
                "--gtest_filter=Regression.$TEST_PATTERN"
        else
            echo "=== Generating all regression reference data ==="
            echo ""
            OMPI_MCA_rmaps_base_oversubscribe=1 GENERATE_REFERENCES=1 \
                "${MPIEXEC:-mpirun}" -np 1 ./tests/test_regression
        fi
        echo ""
        echo "Reference files written to tests/regression/references/"
        echo "Copying to build directory..."
        cp "$ROOT_DIR/tests/regression/references/"*.dat "$BUILD_DIR/tests/regression/references/"
        echo "Done."
        return 0
    fi

    # --- Run specific test by pattern ---
    if [[ -n "$TEST_PATTERN" ]]; then
        echo ""
        echo "=== Running tests matching: $TEST_PATTERN ==="
        echo ""
        cd "$BUILD_DIR"
        ctest -j "$JOBS" -R "$TEST_PATTERN" --output-on-failure --timeout 600
        return $?
    fi

    # --- Run requested tiers, tracking results ---
    # Associative array: test name -> PASS/FAIL
    declare -A RESULTS
    local FAILED=0

    cd "$BUILD_DIR"
    for tier in "${TIERS[@]}"; do
        case "$tier" in
            unit)
                echo ""
                echo "=== Running unit tests ==="
                echo ""
                if ctest -j "$JOBS" -L unit --output-on-failure --timeout 120; then
                    RESULTS[unit]="PASS"
                else
                    RESULTS[unit]="FAIL"
                    FAILED=1
                fi
                ;;
            integration)
                echo ""
                echo "=== Running integration tests ==="
                echo ""
                if ctest -j "$JOBS" -L integration --output-on-failure --timeout 120; then
                    RESULTS[integration]="PASS"
                else
                    RESULTS[integration]="FAIL"
                    FAILED=1
                fi
                ;;
            regression)
                echo ""
                echo "=== Running regression tests (np=1) ==="
                echo ""
                if ctest -j "$JOBS" -L regression_np1 --output-on-failure --timeout 600; then
                    RESULTS["regression (np=1)"]="PASS"
                else
                    RESULTS["regression (np=1)"]="FAIL"
                    FAILED=1
                fi

                echo ""
                echo "=== Running regression tests (np=4) ==="
                echo ""
                if ctest -j "$JOBS" -L regression_np4 --output-on-failure --timeout 600; then
                    RESULTS["regression (np=4)"]="PASS"
                else
                    RESULTS["regression (np=4)"]="FAIL"
                    FAILED=1
                fi
                ;;
        esac
    done

    # --- Print summary ---
    echo ""
    echo "=== Test Results ==="
    echo ""

    # Print in a consistent order matching the tier list
    local ORDER=()
    for tier in "${TIERS[@]}"; do
        case "$tier" in
            unit)        ORDER+=(unit) ;;
            integration) ORDER+=(integration) ;;
            regression)  ORDER+=("regression (np=1)" "regression (np=4)") ;;
        esac
    done

    for name in "${ORDER[@]}"; do
        printf "  %-25s %s\n" "$name" "${RESULTS[$name]}"
    done

    echo ""
    if [[ $FAILED -ne 0 ]]; then
        echo "Some tests FAILED."
        exit 1
    fi
    echo "All tests PASSED."
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
