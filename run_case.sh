#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
CASES_DIR="$ROOT_DIR/cases"
BUILD_TYPE="${BUILD_TYPE:-Release}"

usage() {
    cat <<EOF
Usage: ./run_case.sh [options] <case_name> [-- program args...]

Run a simulation case from cases/.

Each case can have:
  - A JSON input file:  cases/<name>/<name>.jsonc  (run via sifv generic driver)
  - A compiled source:  cases/<name>/*.cpp          (built and run directly)

If both exist, the JSON file is used by default. Use --case-optimization to
generate and compile an optimized C++ source from the JSON instead.

Each case runs in its own directory under cases/<case_name>/, where all output
(VTK files, logs) is written.

Options:
  -l, --list              List available cases
  -c, --clean             Clean rebuild (remove build directory first)
  -d, --debug             Build in Debug mode
  -n <N>                  Number of MPI ranks (default: 1)
  --build-only            Only build, do not run
  --compiled              Force using the compiled C++ source instead of JSON
  --case-optimization     Generate and compile a custom C++ main() from the
                          JSON input for maximum performance (codegen path)
  --hypre                 Enable Hypre pressure solver (PCG+PFMG)
  -j <N>                  Parallel build jobs (default: number of cores)
  -o, --output-dir <dir>  Override the run directory (default: cases/<case_name>)
  -h, --help              Show this help

Examples:
  ./run_case.sh 1D_sod_shocktube                      # JSON case via sifv driver
  ./run_case.sh --case-optimization 1D_sod_shocktube   # JSON case via codegen
  ./run_case.sh --compiled 1D_sod_shocktube            # compiled C++ source
  ./run_case.sh -n 4 2D_rising_bubble
  ./run_case.sh -d 1D_sod_shocktube
  ./run_case.sh --hypre 2D_rising_bubble -- --semi-implicit
EOF
    exit "${1:-0}"
}

list_cases() {
    echo "Cases in cases/:"
    echo ""

    # Collect all case names (from subdirectories containing JSON and/or C++ files)
    declare -A cases
    for dir in "$CASES_DIR"/*/; do
        [[ -d "$dir" ]] || continue
        name="$(basename "$dir")"
        # Check for JSON file inside subdirectory
        for ext in jsonc json; do
            if [[ -f "$dir/$name.$ext" ]]; then
                cases[$name]="${cases[$name]:-}json "
                break
            fi
        done
        # Check for compiled C++ source
        if compgen -G "$dir"/*.cpp > /dev/null 2>&1; then
            cases[$name]="${cases[$name]:-}compiled "
        fi
    done

    # Print sorted
    for name in $(echo "${!cases[@]}" | tr ' ' '\n' | sort); do
        types="${cases[$name]}"
        tags=""
        [[ "$types" == *json* ]] && tags="${tags}json"
        [[ "$types" == *compiled* ]] && { [[ -n "$tags" ]] && tags="${tags}, "; tags="${tags}compiled"; }
        printf "  %-30s [%s]\n" "$name" "$tags"
    done
}

# --- Parse arguments ---
CLEAN=false
BUILD_ONLY=false
ENABLE_HYPRE=false
CASE_OPTIMIZATION=false
FORCE_COMPILED=false
MPI_RANKS=1
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CASE_NAME=""
OUTPUT_DIR=""
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
        --case-optimization)
            CASE_OPTIMIZATION=true
            shift
            ;;
        --compiled)
            FORCE_COMPILED=true
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
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
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

# --- Determine case type ---
JSON_FILE=""
COMPILED_DIR=""

# Strip extension if user provided one
BARE_NAME="${CASE_NAME%.jsonc}"
BARE_NAME="${BARE_NAME%.json}"

# Check for JSON case inside subdirectory: cases/<name>/<name>.jsonc
for ext in jsonc json; do
    if [[ -f "$CASES_DIR/$BARE_NAME/$BARE_NAME.$ext" ]]; then
        JSON_FILE="$CASES_DIR/$BARE_NAME/$BARE_NAME.$ext"
        break
    fi
done

# Check for compiled source in cases/<name>/
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

# Decide mode: JSON (driver or codegen) vs compiled
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

# --- Build and run ---
if $USE_JSON; then
    if $CASE_OPTIMIZATION; then
        # ---- Code generation path ----
        CODEGEN_DIR="$BUILD_DIR/codegen"
        mkdir -p "$CODEGEN_DIR"
        GEN_SRC="$CODEGEN_DIR/${BARE_NAME}.cpp"
        GEN_TARGET="codegen_${BARE_NAME}"

        echo "Generating optimized source from $JSON_FILE..."
        python3 "$ROOT_DIR/tools/codegen.py" "$JSON_FILE" -o "$GEN_SRC"

        # Build the core library first, then compile the generated source
        echo "Building library and $GEN_TARGET (optimized)..."
        cmake --build "$BUILD_DIR" --target SemiImplicitFV -j "$JOBS"

        # Compile the generated file directly against the library
        cd "$BUILD_DIR"
        COMPILE_CMD="$(grep -m1 'CXX_COMPILER' CMakeCache.txt | cut -d= -f2)"
        COMPILE_CMD="${COMPILE_CMD:-c++}"
        INCLUDE_FLAGS="-I$ROOT_DIR/include"
        MPI_CFLAGS="$(pkg-config --cflags ompi 2>/dev/null || mpiCC --showme:compile 2>/dev/null || true)"
        MPI_LDFLAGS="$(pkg-config --libs ompi 2>/dev/null || mpiCC --showme:link 2>/dev/null || echo "-lmpi")"

        # Hypre flags for codegen builds (static lib not transitively linked)
        HYPRE_CFLAGS=""
        HYPRE_LDFLAGS=""
        if $ENABLE_HYPRE; then
            HYPRE_CFLAGS="-DSIFV_HAS_HYPRE"
            HYPRE_BUILD="$BUILD_DIR/_deps/hypre-build"
            HYPRE_LDFLAGS="-L$HYPRE_BUILD -L$HYPRE_BUILD/lib -lHYPRE"
        fi

        "$COMPILE_CMD" -std=c++17 -O3 -march=native \
            $INCLUDE_FLAGS \
            $HYPRE_CFLAGS \
            $MPI_CFLAGS \
            "$GEN_SRC" \
            -L"$BUILD_DIR" -lSemiImplicitFV \
            $HYPRE_LDFLAGS \
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
            mpirun -np "$MPI_RANKS" "$BUILD_DIR/$GEN_TARGET" \
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
            mpirun -np "$MPI_RANKS" "$BUILD_DIR/sifv" "$JSON_FILE" \
                "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
        fi
    fi
else
    # ---- Compiled source path ----
    # Find the CMake target name (basename of the subdirectory)
    TARGET_NAME="$BARE_NAME"

    echo "Building $TARGET_NAME..."
    cmake --build "$BUILD_DIR" --target "$TARGET_NAME" -j "$JOBS"

    if ! $BUILD_ONLY; then
        echo ""
        echo "=== Running $TARGET_NAME with $MPI_RANKS MPI rank(s) ==="
        echo "=== Output directory: $OUTPUT_DIR ==="
        echo ""
        cd "$OUTPUT_DIR"
        mpirun -np "$MPI_RANKS" "$BUILD_DIR/$TARGET_NAME" \
            "${PROGRAM_ARGS[@]+"${PROGRAM_ARGS[@]}"}"
    fi
fi
