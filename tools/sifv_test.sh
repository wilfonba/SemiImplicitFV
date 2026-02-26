#!/usr/bin/env bash
# Build and run the test suite.
# Sourced by sifv.sh — do not execute directly.

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
    JOBS="$(default_jobs)"
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
    $CLEAN && clean_build

    # --- Configure with tests enabled ---
    if needs_configure \
        "CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}" \
        "BUILD_TESTS:BOOL=ON"; then
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
        _test_generate_refs "$TEST_PATTERN"
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

    # --- Run requested tiers ---
    _test_run_tiers "$JOBS" "${TIERS[@]}"
}

# --- Internal helpers ---

_test_generate_refs() {
    local TEST_PATTERN="$1"
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
}

_test_run_tiers() {
    local JOBS="$1"
    shift
    local TIERS=("$@")

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
