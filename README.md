# SemiImplicitFV

A finite volume solver for the compressible Euler equations on rectilinear meshes in 1D, 2D, and 3D. Supports explicit (SSP Runge-Kutta) and semi-implicit (pressure-split) time integration, high-order WENO and upwind reconstruction, multiple Riemann solvers, and Information Geometric Regularization (IGR). Output is in VTK format for visualization with ParaView.

## Features

- **Explicit and semi-implicit time integration** — SSP-RK1/2/3 for explicit; advective CFL with implicit pressure correction for semi-implicit (Kwatra et al.)
- **High-order spatial reconstruction** — WENO and upwind schemes at 1st, 3rd, and 5th order
- **Riemann solvers** — Lax-Friedrichs, Rusanov, and HLLC
- **Equations of state** — Ideal gas and stiffened gas
- **N-phase compressible flow** — Volume-fraction-based multi-phase model with per-phase stiffened gas EOS, Wood's mixture sound speed, and mixture Riemann solvers
- **Viscosity** — Newtonian viscous stress tensor with Stokes hypothesis; per-phase viscosity via arithmetic mixture rule for multi-phase flows
- **Body forces** — Time-dependent gravitational / body force acceleration per dimension
- **Surface tension** — Capillary stress tensor (Schmidmayer et al. 2017) with CSF interface force
- **Information Geometric Regularization (IGR)** — Entropic pressure via elliptic solve for improved stability
- **1D / 2D / 3D** on rectilinear (uniform) meshes with ghost cells
- **Boundary conditions** — Periodic, Reflective, Outflow, Slip Wall, No-Slip Wall
- **MPI parallelism** — Cartesian domain decomposition with non-blocking halo exchange
- **Checkpoint / restart** — Periodic binary checkpoints with automatic restart for HPC wall-time resilience
- **VTK output** — `.vtr` (serial), `.pvtr` (parallel), and `.pvd` (time series) for ParaView; ASCII or binary (appended raw) format
- **PETSc pressure solver** — Optional CG + GAMG algebraic multigrid for mesh-independent semi-implicit pressure convergence
- **NVTX profiling** — Nsight Systems integration with NVTX push/pop macros for performance analysis
- **GPU-ready architecture** — C-style structs, free functions, enum+switch dispatch, flat arrays — no virtual dispatch, no heap allocations in hot loops

## Convergence

The 1D advection test case (Gaussian density pulse on a periodic domain, WENO5 + RK3) demonstrates the expected convergence rates for both solvers across grid resolutions from 128 to 2048 cells:

![Convergence of explicit and semi-implicit solvers](cases/1D_advection_E/convergence_plot.png)

- **Explicit solver** (red, CFL = 0.8): 5th-order convergence in all norms, matching the WENO5 spatial accuracy
- **Semi-implicit solver** (blue, effective CFL ~ 28): 3rd-order convergence, limited by the RK3 time integrator but with substantially larger time steps

## Quick Start

```bash
# Run the Sod shock tube (builds automatically on first run)
./sifv.sh run 1D_sod_shocktube

# List all available cases
./sifv.sh list
```

Output VTK files are written to a `VTK/` directory inside the case folder (e.g. `cases/1D_sod_shocktube/VTK/`). Open the `.pvd` file in [ParaView](https://www.paraview.org/) to visualize the results.

## Building and Running

The `sifv.sh` script handles configuring, building, and running automatically. There is no need to invoke CMake or Make directly.

```bash
./sifv.sh run 1D_sod_shocktube              # Build and run (1 MPI rank)
./sifv.sh run -n 4 2D_riemann               # Run with 4 MPI ranks
./sifv.sh run --debug 1D_advection           # Debug build (enables AddressSanitizer)
./sifv.sh run --build-only 2D_riemann        # Build without running
./sifv.sh run --case-optimization 1D_sod     # Codegen: compile JSON into optimized C++
./sifv.sh run --petsc 3D_taylor_green_vortex # Enable PETSc pressure solver
./sifv.sh run --nsys 2D_riemann             # Profile with Nsight Systems (NVTX)
./sifv.sh list                         # List available cases
```

### Requirements

- CMake 3.14+
- C++17 compiler
- MPI implementation (e.g., Open MPI, MPICH)
- PETSc (optional, for `--petsc` pressure solver)
- NVIDIA Nsight Systems (optional, for `--nsys` profiling)

### Manual Build

If you prefer to build manually instead of using `sifv.sh`:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Defining Cases (JSON Input)

Cases are defined as JSON files in `cases/<name>/<name>.jsonc`. This is the primary way to set up simulations — no C++ coding required. The `sifv` generic driver reads the JSON and runs the simulation.

### Minimal Example

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
    "mesh": { "nx": 100, "xMin": 0.0, "xMax": 1.0 },
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

### JSON Schema Reference

**Top-level sections:**

| Section | Required | Description |
|---|---|---|
| `config` | Yes | Solver parameters (dim, RK order, CFL, etc.) |
| `eos` | No | Equation of state (`"IdealGas"` or `"StiffenedGas"`) |
| `riemannSolver` | No | `"LF"`, `"Rusanov"`, or `"HLLC"` (default) |
| `pressureSolver` | No | `"GaussSeidel"` (default), `"Jacobi"`, or `"PETSc"` (requires `--petsc` build flag) |
| `mesh` | Yes | Grid dimensions and extents |
| `boundaryConditions` | No | Per-face BC: `"Outflow"`, `"Periodic"`, `"Symmetry"`, `"SlipWall"`, `"NoSlipWall"` |
| `timeLoop` | Yes | End time, output interval, print interval |
| `output` | No | VTK base name, directory, and format (`"VTKText"` or `"VTKRaw"`) |
| `initialConditions` | Yes | Default state and geometry-based patches |
| `smoothing` | No | Post-initialization field smoothing iterations |
| `restart` | No | Checkpoint/restart settings |

**Initial condition patches** support `"box"`, `"sphere"`, `"plane"`, and `"analytic"` geometry types. Patch states inherit from the default state — only specify fields that differ.

### Code Generation (Case Optimization)

For maximum performance, the `--case-optimization` flag generates a standalone C++ `main()` from the JSON with all parameters hardcoded as compile-time constants:

```bash
./sifv.sh run --case-optimization 1D_sod_shocktube
```

This eliminates runtime parsing overhead and enables the compiler to optimize aggressively.

## Configuration Reference

All simulation parameters are set through `SimulationConfig` (defined in `include/SimulationConfig.hpp`). In JSON cases, these appear under the `"config"` section. In compiled C++ cases, they are set directly on the struct via `config_defaults()` followed by field assignment.

### Config Fields

| Field | Default | Description |
|---|---|---|
| `dim` | 1 | Spatial dimensions (1, 2, or 3) |
| `nGhost` | 3 | Ghost cells (must match reconstruction order) |
| `RKOrder` | 1 | Runge-Kutta order (1, 2, or 3) |
| `reconOrder` | `"WENO1"` | Reconstruction scheme |
| `semiImplicit` | false | Use semi-implicit solver |
| `useIGR` | false | Enable IGR |
| `wenoEps` | 1e-6 | WENO smoothness parameter |

### Explicit Solver Parameters (`explicitParams`)

| Field | Description |
|---|---|
| `cfl` | CFL number (acoustic) |
| `constDt` | Fixed time step (0 = adaptive) |
| `maxDt` | Maximum time step |
| `minDt` | Minimum time step |

### Semi-Implicit Solver Parameters (`semiImplicitParams`)

| Field | Description |
|---|---|
| `cfl` | CFL number (advective) |
| `constDt` | Fixed time step (0 = adaptive) |
| `maxDt` / `minDt` | Time step bounds |
| `maxAcousticCFL` | If > 0, limits dt so acoustic CFL stays below this value |
| `maxPressureIters` | Max pressure Poisson iterations |
| `pressureTol` | Pressure solve convergence tolerance |
| `singlePressureSolve` | Only solve pressure on the final RK stage (default: false) |

### Multi-Phase Configuration (`multiPhaseParams`)

Enable N-phase flow with per-phase stiffened gas EOS. Set `pInf = 0` for ideal gas phases.

```jsonc
"multiPhaseParams": {
    "nPhases": 2,
    "phases": [
        { "gamma": 4.4, "pInf": 6.0e8 },
        { "gamma": 1.4, "pInf": 0.0 }
    ],
    "alphaMin": 1e-8
}
```

### Viscosity (`viscousParams`)

```jsonc
"viscousParams": {
    "mu": 1.81e-5,
    "phaseMu": [1.0e-3, 1.81e-5]
}
```

Set `mu` for uniform viscosity or `phaseMu` for per-phase viscosity (multi-phase only). When `phaseMu` is set, the scalar `mu` is ignored.

### Body Forces (`bodyForceParams`)

Per-dimension acceleration of the form `a(t) = a + b * cos(c * t + d)`:

```jsonc
"bodyForceParams": {
    "a": [0.0, -9.81, 0.0]
}
```

### Surface Tension (`surfaceTensionParams`)

Capillary stress tensor (Schmidmayer et al. 2017) for multi-phase flows:

```jsonc
"surfaceTensionParams": {
    "sigma": 0.0728,
    "epsGradAlpha": 1e-8
}
```

### IGR Parameters (`igrParams`)

| Field | Description |
|---|---|
| `alphaCoeff` | Regularization coefficient |
| `IGRIters` | Gauss-Seidel iterations per step |
| `IGRWarmStartIters` | Warm-start iterations at t=0 |

### Reconstruction Orders

| Scheme | Order | Ghost Cells Required |
|---|---|---|
| `WENO1` / `UPWIND1` | 1st | 1 |
| `WENO3` / `UPWIND3` | 3rd | 2 |
| `WENO5` / `UPWIND5` | 5th | 3 |

WENO schemes include nonlinear shock-capturing weights. Upwind schemes use standard polynomial reconstruction.

## Checkpoint / Restart

Long-running simulations can write periodic binary checkpoints and restart from them. This is essential for jobs on HPC clusters with wall-time limits.

Add an optional `"restart"` section to the JSON input file:

```jsonc
"restart": {
    "checkpoint": true,                          // write checkpoints at outputInterval (default: false)
    "file": "Checkpoint/checkpoint.{rank}.bin"   // restart from this file (omit for a fresh run)
}
```

| Field | Default | Description |
|---|---|---|
| `checkpoint` | false | When `true`, writes checkpoint files at every `outputInterval` to a `Checkpoint/` directory. |
| `file` | (empty) | Path to a checkpoint file to restart from. `{rank}` is replaced with the zero-padded MPI rank (e.g. `0000`). When set, initial conditions and smoothing are skipped. |

Checkpoint files are written as `checkpoint.RRRR.bin` (one per MPI rank, zero-padded to 4 digits). Only the latest checkpoint is kept — each write overwrites the previous file. The checkpoint stores all conservative variables and multi-phase fields in a compact binary format; primitive variables are recomputed on restart.

### Example: restart a killed job

```bash
# 1. Run with checkpoints enabled
#    (add "restart": {"checkpoint": true} to the case JSONC)
./sifv.sh run 1D_sod_shocktube

# 2. Job is killed at the wall-time limit...

# 3. Add the restart file path and re-run
#    (add "file": "Checkpoint/checkpoint.{rank}.bin" to the "restart" section)
./sifv.sh run 1D_sod_shocktube
```

The simulation resumes from the saved time and step count.

## Writing Compiled Cases (C++)

For cases requiring custom post-processing, diagnostics, or logic not expressible in JSON, you can write a standalone C++ source file. Place it in `cases/<name>/<name>.cpp` (generated `.cpp` files in `cases/` are git-ignored).

Use `--compiled` to build and run a compiled case:

```bash
./sifv.sh run --compiled 2D_flow_over_circle
```

Compiled cases use the C-style API directly — call `config_defaults()` to initialize a `SimulationConfig`, set fields, then use `runtime_init()` / `run_time_loop()` / `runtime_free()`.

## MPI Execution

Run with multiple MPI ranks using the `-n` flag:

```bash
./sifv.sh run -n 4 2D_riemann
```

The `Runtime` struct and associated free functions handle domain decomposition, halo exchange, and parallel VTK output automatically. Each rank writes its own `.vtr` piece file, and rank 0 writes the `.pvtr` and `.pvd` metadata files.

## Testing

The project includes a three-tier test suite built with [GoogleTest](https://github.com/google/googletest) and run via CTest. All tests are MPI-aware and executed through `mpirun`. Each GoogleTest `TEST()` case is registered as an individual CTest entry, giving granular pass/fail reporting (57 tests total across all tiers).

### Running Tests

`sifv.sh test` handles configuring, building, and running automatically:

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
```

The `-j` flag controls both CMake build parallelism and CTest parallel test execution.

### Test Tiers

| Tier | Label | Description | Approx. Time |
|---|---|---|---|
| **Unit** | `unit` | Individual functions: EOS, Riemann solvers, reconstruction stencils, mixture EOS, IGR | ~1s |
| **Integration** | `integration` | Multi-module interactions: boundary conditions, state conversion round-trips, explicit time stepping, pressure solvers | ~1s |
| **Regression (np=1)** | `regression_np1` | Run 5 cases for 50 steps on 1 rank, compare pointwise against reference data | ~2s |
| **Regression (np=4)** | `regression_np4` | Same 5 cases on 4 MPI ranks, verify results match within tolerance (1e-8) | ~2s |

### Regression Test Cases

| Case | Dim | Mesh | Solver |
|---|---|---|---|
| `1D_sod_shocktube_50` | 1D | 100 | Explicit (HLLC, WENO5, RK3) |
| `1D_advection_SI_50` | 1D | 100 | Semi-implicit (HLLC, WENO5, RK3) |
| `1D_liquid_gas_shocktube_50` | 1D | 100 | Explicit multi-phase (HLLC, WENO5, RK3) |
| `2D_isentropic_vortex_50` | 2D | 50x50 | Semi-implicit (HLLC, UPWIND5, RK3) |
| `2D_channel_flow_50` | 2D | 20x20 | Explicit (HLLC, WENO5, RK3) |
| `3D_taylor_green_vortex_50` | 3D | 40x40x40 | Explicit (HLLC, WENO5, RK3) + viscosity |
| `3D_taylor_green_vortex_SI_50` | 3D | 40x40x40 | Semi-implicit (HLLC, WENO5, RK3) + viscosity |

### Regenerating Reference Data

Reference files are committed to `tests/regression/references/`. To regenerate after changing solver behavior:

```bash
./sifv.sh test --generate-references
```

Then re-run the full suite to verify:

```bash
./sifv.sh test regression
```

## Project Structure

```
SemiImplicitFV/
├── cases/                 Case definitions (JSON input files)
│   ├── 1D_advection_E/
│   ├── 1D_advection_SI/
│   ├── 1D_sod_shocktube/
│   ├── 1D_gas_gas_shocktube/
│   ├── 1D_liquid_gas_shocktube/
│   ├── 1D_hydrostatic_water/
│   ├── 2D_channel_flow/
│   ├── 2D_isentropic_vortex/
│   ├── 2D_laplace_pressure_jump/
│   ├── 2D_quasi1D_sod/
│   ├── 2D_riemann/
│   ├── 2D_rising_bubble/
│   └── 3D_taylor_green_vortex/
├── driver/                Generic JSON driver (sifv)
├── include/               Header files
├── src/                   Library source files
├── tests/                 Test suite (GoogleTest + CTest)
│   ├── unit/              Unit tests (EOS, Riemann, reconstruction, etc.)
│   ├── integration/       Integration tests (BCs, solvers, state conversion)
│   └── regression/        Regression tests (50-step runs with reference data)
├── tools/                 Code generation and utilities
│   └── codegen.py         JSON → optimized C++ source generator
├── .github/workflows/     GitHub Actions CI
├── CMakeLists.txt
└── sifv.sh                Build, run, and test helper script
```

## Visualization

Output files are VTK XML RectilinearGrid format, viewable in [ParaView](https://www.paraview.org/):

1. Open the `.pvd` file in ParaView to load the full time series
2. Apply a color map to fields like `Density`, `Pressure`, or `Velocity`
3. Use the animation controls to step through time

Fields written per cell: density, velocity (u, v, w), momentum, pressure, temperature, total energy, and entropic pressure (sigma). Multi-phase simulations additionally write per-phase volume fractions (`Alpha_0`, `Alpha_1`, ...) and partial densities (`AlphaRho_0`, `AlphaRho_1`, ...).

## License

See [LICENSE](LICENSE) for details.
