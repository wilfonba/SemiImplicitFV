# Contributing to SemiImplicitFV

## Adding a New Case

The preferred way to define a simulation is with a **JSON input file**. No C++ coding is required for standard cases.

### JSON Case (Preferred)

1. Create `cases/<name>/<name>.jsonc`
2. Define the simulation using the JSON schema (see below)
3. Test: `./sifv.sh run <name>`
4. VTK output appears in `cases/<name>/VTK/`

Example (`cases/my_case/my_case.jsonc`):

```jsonc
{
    "config": {
        "dim": 1,
        "nGhost": 3,
        "RKOrder": 3,
        "reconOrder": "WENO5",
        "explicitParams": { "cfl": 0.6 }
    },
    "eos": { "type": "IdealGas", "gamma": 1.4, "R": 287.0 },
    "riemannSolver": "HLLC",
    "mesh": { "nx": 200, "xMin": 0.0, "xMax": 1.0 },
    "boundaryConditions": { "xLow": "Outflow", "xHigh": "Outflow" },
    "timeLoop": { "endTime": 0.2, "outputInterval": 0.01 },
    "output": { "baseName": "my_case" },
    "initialConditions": {
        "default": { "rho": 1.0, "u": 0.0, "p": 1.0 },
        "patches": [
            {
                "geometry": { "type": "plane", "point": [0.5, 0, 0], "normal": [1, 0, 0] },
                "state": { "rho": 0.125, "p": 0.1 }
            }
        ]
    }
}
```

### JSON Schema Sections

| Section | Required | Description |
|---|---|---|
| `config` | Yes | Solver parameters: `dim`, `nGhost`, `RKOrder`, `reconOrder`, `semiImplicit`, `useIGR`, CFL/time-step params, multi-phase, viscosity, body forces, surface tension, IGR |
| `eos` | No | Equation of state: `"IdealGas"` (default) or `"StiffenedGas"` with `gamma`, `R`, `pInf` |
| `riemannSolver` | No | `"LF"`, `"Rusanov"`, or `"HLLC"` (default) |
| `pressureSolver` | No | `"GaussSeidel"` (default), `"Jacobi"`, or `"PETSc"` (requires `--petsc` build flag) |
| `mesh` | Yes | `nx`/`ny`/`nz` and `xMin`/`xMax`/`yMin`/`yMax`/`zMin`/`zMax` |
| `boundaryConditions` | No | Per-face: `"Outflow"`, `"Periodic"`, `"Symmetry"`, `"SlipWall"`, `"NoSlipWall"` |
| `timeLoop` | Yes | `endTime`, `outputInterval`, `printInterval`, `checkNaN` |
| `output` | No | `baseName`, `directory`, and `format` (`"VTKText"` or `"VTKRaw"`) for VTK output |
| `initialConditions` | Yes | `default` state + `patches` array with geometry and state overrides |
| `smoothing` | No | `iterations` for post-initialization field smoothing |
| `restart` | No | `checkpoint` (bool) and `file` (path) for checkpoint/restart |

### Initial Condition Geometries

- **`box`**: `min` [x,y,z], `max` [x,y,z]
- **`sphere`**: `center` [x,y,z], `radius`
- **`plane`**: `point` [x,y,z], `normal` [x,y,z] — positive side gets the patch state
- **`analytic`**: expressions evaluated per cell (with optional `region` sub-geometry)

### Compiled C++ Case (When Needed)

Use compiled cases only when you need logic not expressible in JSON — custom diagnostics, convergence studies, drag/lift computation, etc.

1. Create `cases/<name>/<name>.cpp`
2. Build and run: `./sifv.sh run --compiled <name>`

Compiled cases use the C-style API: call `config_defaults()` to initialize a `SimulationConfig`, set fields, then use `runtime_init()` / `run_time_loop()` / `runtime_free()`.

### Code Generation

The `--case-optimization` flag generates optimized C++ from JSON and compiles it:

```bash
./sifv.sh run --case-optimization <name>
```

This hardcodes all parameters at compile time for maximum performance.

## Testing

The project has a three-tier test suite: unit, integration, and regression. All tests use [GoogleTest](https://github.com/google/googletest) and are registered with CTest. Tests run through `mpirun` (even single-rank tests) because the solver requires MPI initialization.

### Building and Running Tests

```bash
./sifv.sh test                          # Build and run all tests
./sifv.sh test unit                     # Unit tests only
./sifv.sh test integration              # Integration tests only
./sifv.sh test regression               # Regression tests (np=1 and np=4)
./sifv.sh test unit integration         # Multiple tiers
./sifv.sh test -j 8                     # Parallel build and test execution
./sifv.sh test -d unit                  # Debug build (AddressSanitizer)
./sifv.sh test -c                       # Clean build directory first
./sifv.sh test --build-only             # Build without running
./sifv.sh test -o <pattern>             # Run tests matching regex pattern
./sifv.sh test -l                       # List all test names
./sifv.sh test --generate               # Regenerate all regression references
./sifv.sh test -o TaylorGreenVortex3D --generate  # Regenerate one reference
```

`sifv.sh test` automatically configures CMake with `-DBUILD_TESTS=ON`, builds, and runs the requested tiers via CTest. The `-j` flag controls both CMake build parallelism (`cmake --build -j`) and CTest parallel test execution (`ctest -j`). Each GoogleTest `TEST()` case is registered as an individual CTest entry, giving granular pass/fail reporting.

### Test Organization

```
tests/
├── CMakeLists.txt              Build system (GoogleTest fetch, executables, CTest)
├── test_main.cpp               Shared main() with MPI_Init/Finalize
├── unit/                       Unit tests — single functions in isolation
│   ├── test_eos.cpp
│   ├── test_riemann_solver.cpp
│   ├── test_reconstruction.cpp
│   ├── test_mixture_eos.cpp
│   └── test_igr.cpp
├── integration/                Integration tests — multiple modules together
│   ├── test_boundary_conditions.cpp
│   ├── test_state_conversion.cpp
│   ├── test_explicit_rhs.cpp
│   └── test_pressure_solver.cpp
└── regression/                 Regression tests — full simulations vs reference data
    ├── test_regression.cpp
    ├── cases/                  Short-run JSONC input files (50 steps each)
    └── references/             Committed reference .dat files
```

All three tiers compile into separate executables (`test_unit`, `test_integration`, `test_regression`) that share a common `test_main.cpp` handling MPI init/finalize.

### Adding Unit Tests

Unit tests verify individual functions in isolation. Add tests to an existing file in `tests/unit/` or create a new file.

**When to write a unit test:** You added or modified a function that computes a value from its inputs (EOS calculations, flux computations, stencil reconstructions, etc.).

Example — testing a new EOS function:

```cpp
// In tests/unit/test_eos.cpp
TEST(EOS, MyNewFunction) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);
    double result = eos_my_new_function(&eos, /* args */);
    EXPECT_NEAR(result, expected_value, 1e-14);
}
```

If you create a new test file, add it to the `test_unit` executable in `tests/CMakeLists.txt`:

```cmake
add_executable(test_unit
    unit/test_eos.cpp
    unit/test_riemann_solver.cpp
    # ...
    unit/test_my_new_module.cpp    # <-- add here
)
```

### Adding Integration Tests

Integration tests verify that multiple modules work together correctly: boundary conditions filling ghost cells, state conversion round-trips, solver steps on small meshes.

**When to write an integration test:** You added a new boundary condition type, changed how modules interact (e.g. solver + halo exchange), or added a new solver variant.

Integration tests typically create a small mesh (8-32 cells), set up a known state, perform an operation, and check the result. Tests that call the solver need to use the `Runtime` API for proper MPI/halo setup:

```cpp
// In tests/integration/test_my_feature.cpp
TEST(MyFeature, BasicBehavior) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.nGhost = 2;

    Runtime rt;
    memset(&rt, 0, sizeof(rt));
    MPI_Comm_rank(MPI_COMM_WORLD, &rt.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt.size);

    int periods[3] = {0, 0, 0};
    RectilinearMesh mesh;
    runtime_create_uniform_mesh_1d(&rt, &mesh, &config, 16, 0.0, 1.0, periods);
    runtime_set_bc(&rt, &mesh, XLOW, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, XHIGH, BC_OUTFLOW);

    // ... set up state, run operation, check results ...

    mesh_free(&mesh);
    if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
    if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
}
```

### Adding Regression Tests

Regression tests run full simulations for 50 time steps and compare every interior cell's conservative fields pointwise against committed reference data. They catch unintended changes in solver behavior across the entire pipeline.

**When to write a regression test:** You added a new case type, a new solver combination, or want to lock down the behavior of an existing configuration.

#### Step 1: Create a short-run JSONC case

Add a file in `tests/regression/cases/` derived from an existing case in `cases/`. Key requirements:

- Use `constDt` to guarantee exactly 50 steps: `"constDt": endTime / 50`
- Set a large `outputInterval` to suppress VTK output: `"outputInterval": 1e10`
- Set a large `printInterval` to suppress step printing: `"printInterval": 100000`
- Use small meshes for speed (1D: 100, 2D: 50x50 or smaller)

Example (`tests/regression/cases/my_case_50.jsonc`):

```jsonc
{
    "config": {
        "dim": 1, "nGhost": 3, "RKOrder": 3, "reconOrder": "WENO5",
        "explicitParams": { "constDt": 0.004 }
    },
    "eos": { "type": "IdealGas", "gamma": 1.4, "R": 287.0 },
    "riemannSolver": "HLLC",
    "mesh": { "nx": 100, "xMin": 0.0, "xMax": 1.0 },
    "boundaryConditions": { "xLow": "Outflow", "xHigh": "Outflow" },
    "timeLoop": { "endTime": 0.2, "outputInterval": 1e10, "printInterval": 100000 },
    "initialConditions": {
        "default": { "rho": 1.0, "u": 0.0, "p": 1.0 },
        "patches": [
            { "geometry": { "type": "plane", "point": [0.5,0,0], "normal": [1,0,0] },
              "state": { "rho": 0.125, "p": 0.1 } }
        ]
    }
}
```

#### Step 2: Register the test

Add a `REGRESSION_TEST` macro call in `tests/regression/test_regression.cpp`:

```cpp
REGRESSION_TEST(MyCaseTest, "my_case_50.jsonc", "my_case_50.dat", 50)
```

Arguments: test name, JSONC filename, reference filename, number of steps.

Then add the test name to the `REGRESSION_CASES` list in `tests/CMakeLists.txt` so it gets registered for both np=1 and np=4 CTest execution:

```cmake
set(REGRESSION_CASES
    SodShocktube1D
    AdvectionSI1D
    GasGasShocktube1D
    IsentropicVortex2D
    ChannelFlow2D
    TaylorGreenVortex3D
    TaylorGreenVortexSI3D
    MyCaseTest              # <-- add here
)
```

#### Step 3: Generate reference data

```bash
./sifv.sh test -o MyCaseTest --generate
```

This writes the reference file to `tests/regression/references/my_case_50.dat`. Then verify it passes:

```bash
./sifv.sh test -o MyCaseTest
```

#### Step 4: Commit the reference file

Commit both the JSONC case and the `.dat` reference file. The reference file contains one line per interior cell with full `%.17g` precision for all conservative fields.

### Regenerating All References

When solver behavior changes intentionally (algorithm improvement, bug fix), regenerate all reference data:

```bash
./sifv.sh test --generate-references    # regenerate all reference .dat files
./sifv.sh test                          # verify all tiers pass
```

### Regression Tolerance

Regression tests default to a relative tolerance of **1e-8** to accommodate floating-point differences between single-rank and multi-rank MPI runs (reduction order, domain decomposition boundaries). Cases with iterative pressure solvers in 3D may need a looser tolerance — use `REGRESSION_TEST_TOL` instead of `REGRESSION_TEST` to specify a custom tolerance:

```cpp
REGRESSION_TEST_TOL(MyCaseTest, "my_case_50.jsonc", "my_case_50.dat", 50, 1e-6)
```

If a comparison fails, the test prints the first 20 mismatched cells with field name, computed value, reference value, and relative error.

### CI

GitHub Actions runs all test tiers on every push and pull request to `master`. The workflow is defined in `.github/workflows/tests.yml` and uses `sifv.sh test` to run each tier. CI uses Ubuntu with system OpenMPI and runs regression tests at both np=1 and np=4.

## Code Style

- C++17, no namespaces (except `ExpressionEvaluator` class which uses pimpl)
- Headers use `#ifndef` include guards (not `#pragma once`)
- Plain C structs with `_init()` / `_free()` lifecycle functions for heap-allocated resources
- Free functions with `module_action()` naming (e.g. `config_validate()`, `eos_pressure()`, `mesh_index()`)
- Enums are plain `enum` (not `enum class`) with `UPPER_CASE` values (e.g. `EOS_IDEAL_GAS`, `RS_HLLC`, `BC_OUTFLOW`, `WENO5`)
- Flat `double*` arrays for field data; multi-phase uses stride-based access (`alpha[phase * totalCells + cell]`)
- `#define MAX_PHASES 8` for fixed-size phase arrays in structs
- No virtual dispatch — use enum + switch
- No `std::shared_ptr` or `std::unique_ptr` in data structs (except `ExpressionEvaluator` pimpl)

## Building and Running

`sifv.sh` handles configuring, building, and running automatically:

```bash
./sifv.sh run <name>                         # Build and run
./sifv.sh run --debug <name>                 # Debug build (AddressSanitizer)
./sifv.sh run -n 4 <name>                    # Run with 4 MPI ranks
./sifv.sh run --case-optimization <name>     # Codegen optimized build
./sifv.sh run --petsc <name>                 # Enable PETSc (saved across runs)
./sifv.sh run --no-petsc <name>              # Disable PETSc (saved across runs)
./sifv.sh run --srun -n 4 <name>             # Use srun instead of mpirun (Slurm)
./sifv.sh run -o <dir> <name>               # Override output directory
./sifv.sh list                               # List all cases
```
