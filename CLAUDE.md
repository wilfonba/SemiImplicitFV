# SemiImplicitFV

Finite volume solver for compressible Euler equations on rectilinear meshes (1D/2D/3D) with explicit and semi-implicit time integration and Information Geometric Regularization (IGR).

## Build and Run

Use `sifv.sh` to configure, build, and run cases or tests automatically:

```bash
./sifv.sh run <case>                         # Build and run a JSON case
./sifv.sh run --debug <case>                 # Debug build (AddressSanitizer)
./sifv.sh run -n 4 <case>                    # Run with 4 MPI ranks
./sifv.sh run --case-optimization <case>     # Codegen optimized build
./sifv.sh run --compiled <case>              # Build compiled C++ case
./sifv.sh run --petsc <case>                 # Enable PETSc pressure solver
./sifv.sh run --nsys <case>                  # Profile with Nsight Systems (NVTX)
./sifv.sh list                               # List available cases
./sifv.sh test                               # Run all tests
./sifv.sh test unit                          # Run unit tests only
./sifv.sh test -j 8                          # Parallel build and test execution
```

Manual build (if needed):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Project Structure

- `include/` — all headers
- `src/` — library source files
- `driver/` — generic JSON driver (`sifv`), the main entry point for running cases
- `cases/` — case definitions as JSON/JSONC input files (the primary way to define simulations)
- `tests/` — three-tier test suite (unit, integration, regression) using GoogleTest + CTest
- `tools/` — code generation (`codegen.py`: JSON → optimized C++) and utilities
- `.github/workflows/` — GitHub Actions CI pipeline

## Case System

**JSON input files** in `cases/<name>/<name>.jsonc` are the standard way to define simulations. The `sifv` driver reads a JSON file and runs the simulation without any C++ coding. VTK output is written into the case's own directory (e.g. `cases/1D_sod_shocktube/VTK/`).

Run cases via `sifv.sh run`:
- `./sifv.sh run <case>` — build `sifv` and run the JSON case
- `./sifv.sh run --case-optimization <case>` — generate optimized C++ from JSON via `tools/codegen.py`, compile, and run
- `./sifv.sh run --compiled <case>` — build and run a compiled `.cpp` source directly (for cases with custom post-processing)

When adding new cases, prefer JSON input files. Use compiled C++ only when the case requires logic not expressible in JSON (custom diagnostics, drag/lift computation, convergence studies, etc.).

### JSON Schema

Top-level sections: `config`, `eos`, `riemannSolver`, `pressureSolver`, `mesh`, `boundaryConditions`, `timeLoop`, `output`, `initialConditions`, `smoothing`, `restart`. All sections except `config`, `mesh`, `timeLoop`, and `initialConditions` are optional.

The `pressureSolver` key selects the pressure solver for semi-implicit runs: `"GaussSeidel"` (default), `"Jacobi"`, or `"PETSc"` (CG + GAMG algebraic multigrid via PETSc; requires `--petsc` build flag).

The `output` section supports a `"format"` field: `"VTKText"` (default, ASCII) or `"VTKRaw"` (appended raw binary, compact and fast).

Initial condition patches support `"box"`, `"sphere"`, `"plane"`, and `"analytic"` geometry types. Patch states inherit from the default state.

### Adding a New JSON Case

1. Create `cases/<name>/<name>.jsonc`
2. Define config, mesh, BCs, ICs, and any optional sections
3. Run: `./sifv.sh run <name>`
4. VTK output appears in `cases/<name>/VTK/`

## Checkpoint / Restart

Simulations can write periodic binary checkpoints and restart from them, which is essential for long-running jobs on HPC clusters with wall-time limits.

### Enabling checkpoints

Add an optional `"restart"` section to the JSON input file:

```jsonc
"restart": {
    "checkpoint": true,                          // write checkpoints at outputInterval (default: false)
    "file": "Checkpoint/checkpoint.{rank}.bin"   // restart file path (omit for a fresh run)
}
```

- **`checkpoint`** — When `true`, writes checkpoint files at every `outputInterval` to a `Checkpoint/` directory (created automatically) in the run directory. Files are named `checkpoint.RRRR.bin` (one per MPI rank, zero-padded to 4 digits). Only the latest checkpoint is kept (overwritten each time).
- **`file`** — If present, the simulation loads state from this checkpoint instead of applying initial conditions and smoothing. The placeholder `{rank}` is replaced with the zero-padded MPI rank (e.g. `0000`). After loading, `config.time` and `config.step` are restored from the file and the time loop resumes from where it left off.

### Checkpoint file format

Simple binary format (`Checkpoint.hpp` / `Checkpoint.cpp`):

```
[Header]
  magic (uint64), version (int32), dim (int32), nPhases (int32),
  nx (int32), ny (int32), nz (int32), nGhost (int32),
  step (int32), time (double)
[Data — totalCells doubles each, including ghost cells]
  rho, rhoU, rhoV (if dim>=2), rhoW (if dim>=3), rhoE
  alpha[0..nPhases-1]     (if multi-phase)
  alphaRho[0..nPhases-1]  (if multi-phase)
```

Only conservative variables are saved. Primitives are recomputed on restart via `state_cons_to_prim()`.

### Example workflow

```bash
# 1. Run with checkpoints enabled (add "restart": {"checkpoint": true} to the JSONC)
./sifv.sh run 1D_sod_shocktube

# 2. Job gets killed at wall-time limit...

# 3. Restart: add "file": "Checkpoint/checkpoint.{rank}.bin" to the "restart" section
./sifv.sh run 1D_sod_shocktube
```

## Testing

Three-tier test suite using GoogleTest + CTest. All tests run through `mpirun` since the solver requires MPI initialization. Each GoogleTest `TEST()` case is registered as an individual CTest entry for granular pass/fail reporting.

```bash
./sifv.sh test                          # Build and run all tests
./sifv.sh test unit                     # Unit tests only
./sifv.sh test integration              # Integration tests only
./sifv.sh test regression               # Regression tests (np=1 and np=4)
./sifv.sh test -j 8                     # Parallel build (-j) and test execution (-j)
./sifv.sh test -d unit                  # Debug build
./sifv.sh test -c                       # Clean build directory first
./sifv.sh test --build-only             # Build without running
./sifv.sh test --generate-references    # Regenerate regression reference data
```

- **Unit tests** (`tests/unit/`): Individual functions — EOS, Riemann solvers, reconstruction, mixture EOS, IGR
- **Integration tests** (`tests/integration/`): Multi-module — BCs, state conversion, explicit stepping, pressure solvers
- **Regression tests** (`tests/regression/`): Full 50-step simulations compared pointwise against committed reference data at both np=1 and np=4. Tolerance: 1e-8. Cases are listed in `REGRESSION_CASES` in `tests/CMakeLists.txt`.

CI runs all tiers on every push/PR to master via `.github/workflows/tests.yml`.

## Architecture

The codebase uses C-style architecture: plain C structs, free functions, and enum+switch dispatch. There are no classes (except `ExpressionEvaluator`, which uses the pimpl pattern to isolate the exprtk dependency), no virtual dispatch, no namespaces, and no `shared_ptr` polymorphism. This design is chosen for GPU acceleration readiness.

### Naming conventions

- Structs: `PascalCase` (e.g. `SimulationConfig`, `SolutionState`, `RectilinearMesh`)
- Free functions: `module_action()` (e.g. `config_validate()`, `eos_pressure()`, `mesh_index()`, `state_cons_to_prim()`)
- Enums: plain `enum` (not `enum class`), `UPPER_CASE` values (e.g. `EOS_IDEAL_GAS`, `RS_HLLC`, `BC_OUTFLOW`, `WENO5`)
- Defines: `UPPER_CASE` (e.g. `MAX_PHASES`, `XLOW`, `ZHIGH`)
- Lifecycle: `_init()` / `_free()` for structs with heap allocations

### Module layout

**State** (`State.hpp`): Core data types — `ConservativeState`, `PrimitiveState`, `RiemannFlux`. Defines `MAX_PHASES` (8). All use plain `double` arrays (e.g. `double u[3]`, `double alpha[MAX_PHASES]`), no `std::array`.

**SimulationConfig** (`SimulationConfig.hpp/cpp`): All simulation parameters in a single struct. Sub-structs for `ExplicitParams`, `SemiImplicitParams`, `IGRParams`, `MultiPhaseParams`, etc. Fixed-size arrays (`PhaseEOS phases[MAX_PHASES]`). Free functions: `config_defaults()`, `config_validate()`, `config_is_multi_phase()`, `config_has_viscosity()`, `config_required_ghost_cells()`.

**EOS** (`EquationOfState.hpp/cpp`): Unified module replacing the old `IdealGasEOS`/`StiffenedGasEOS` class hierarchy. `enum EOSType { EOS_IDEAL_GAS, EOS_STIFFENED_GAS }`. `struct EOSData` with type tag. Free functions: `eos_pressure()`, `eos_temperature()`, `eos_sound_speed()`, `eos_internal_energy()`, `eos_total_energy()`, `eos_to_primitive()`, `eos_to_conservative()`.

**Riemann solvers** (`RiemannSolver.hpp/cpp`): Unified module replacing `LFSolver`/`RusanovSolver`/`HLLCSolver`. `enum RiemannSolverType { RS_LF, RS_RUSANOV, RS_HLLC }`. `struct FluxConfig` for solver parameters. Free functions `computeLFFlux()`, `computeRusanovFlux()`, `computeHLLCFlux()` dispatched via `computeFluxDirect()`.

**Reconstruction** (`Reconstruction.hpp/cpp`): `struct ReconstructorData` with pre-allocated `PrimitiveState*` face arrays. Free functions: `reconstructor_init()/_free()`, `reconstruct()`. `enum ReconstructionOrder { WENO1, WENO3, WENO5, UPWIND1, UPWIND3, UPWIND5 }`.

**SolutionState** (`SolutionState.hpp/cpp`): Flat `double*` arrays for all fields. Multi-phase uses flat arrays with stride: `alpha[phase * totalCells + cell]`. Free functions: `solution_state_init()/_free()`, `state_get_conservative()`, `state_set_conservative()`, `state_get_primitive()`, `state_set_primitive()`, `state_cons_to_prim()`, `state_prim_to_cons()`.

**Mesh** (`RectilinearMesh.hpp/cpp`): `struct RectilinearMesh` with `double*` node arrays. `enum BoundaryCondition { BC_SYMMETRY, BC_OUTFLOW, BC_PERIODIC, ... }`. Inline accessors: `mesh_index()`, `mesh_dx()`, `mesh_total_cells()`, `mesh_cellCentroidX()`. Non-inline: `mesh_init()`, `mesh_init_uniform()`, `mesh_free()`, `mesh_apply_bcs()`, `mesh_fill_scalar_ghosts()`.

**Pressure solvers** (`PressureSolver.hpp/cpp`, `PETScPressureSolver.hpp/cpp`): Unified module replacing `GaussSeidelPressureSolver`/`JacobiPressureSolver`/`PETScPressureSolver`. `enum PressureSolverType { PS_GAUSS_SEIDEL, PS_JACOBI, PS_PETSC }`. `struct PressureSolverData`. Free functions: `pressure_solver_init()/_free()`, `pressure_solve()/_mpi()`.

**IGR** (`IGR.hpp/cpp`): `typedef double GradientTensor[3][3]`. Free functions: `igr_compute_alpha()`, `igr_compute_rhs()`, `igr_solve_entropic_pressure()/_mpi()`, `igr_compute_velocity_gradient()`.

**MPI** (`MPIContext.hpp/cpp`): `struct MPIContext` with `MPI_Comm cartComm`. Free functions: `mpi_context_create()/_free()`, `mpi_is_physical_boundary()`.

**Halo exchange** (`HaloExchange.hpp/cpp`): `struct HaloExchange`. Free functions: `halo_init()/_free()`, `halo_exchange_state()/_scalar()`, `halo_exchange_state_direction()/_scalar_direction()`.

**Multi-phase** (`MixtureEOS.hpp/cpp`): Free functions (no namespace): `mixture_pressure()`, `mixture_sound_speed()`, `mixture_total_energy()`, `effective_gamma_and_pi_inf()`. All take raw pointers (`const double*`, `const PhaseEOS*`).

**Solvers** (`ExplicitSolver.hpp/cpp`, `SemiImplicitSolver.hpp/cpp`): `struct ExplicitSolverWork` / `struct SemiImplicitSolverWork` with pre-allocated scratch arrays. Free functions: `explicit_solver_init()/_free()`, `explicit_step()`, `semi_implicit_solver_init()/_free()`, `semi_implicit_step()`.

**Time stepping** (`RKTimeStepping.hpp/cpp`): `struct TimeLoopParams`. `run_time_loop()` takes a C function pointer + `void* ctx` for per-step callbacks.

**Runtime** (`Runtime.hpp/cpp`): `struct Runtime` aggregating mesh, state, halo, VTK session, and solver work structs. Free functions: `runtime_init()/_free()`, `runtime_create_uniform_mesh()`, `runtime_set_bc()`.

**VTK output** (`VTKWriter.hpp/cpp`, `VTKSession.hpp/cpp`): `enum VTKFormat { VTK_TEXT, VTK_RAW }`. Free functions for writing `.vtr`/`.pvtr`/`.pvd` files. `struct VTKSession` with `vtk_session_init()/_write()/_finalize()`.

**Viscous flux** (`ViscousFlux.hpp/cpp`): Free function `add_viscous_fluxes()`.

**Surface tension** (`SurfaceTension.hpp/cpp`): Free function `add_surface_tension_fluxes()`.

**Checkpoint** (`Checkpoint.hpp/cpp`): Free functions for binary checkpoint I/O.

**Profiling** (`NvtxRange.hpp`): `NVTX_PUSH()` / `NVTX_POP()` macros for Nsight Systems.

**Input parsing** (`InputParser.hpp/cpp`): Plain C structs with `char[]` fields. Free functions: `parse_input_file()`, `input_data_free()`. Internally uses nlohmann/json but the public API is C-compatible.

**Expression evaluator** (`ExpressionEvaluator.hpp/cpp`): The one exception — kept as a C++ class with pimpl pattern to isolate the exprtk header dependency. Used only at initialization for analytic IC expressions.

**Code generation** (`tools/codegen.py`): Reads a JSON case file and emits a standalone C++ `main()` using the C-style API.

## Key Configuration

All simulation parameters live in `SimulationConfig` (see `include/SimulationConfig.hpp`):
- `dim` (1-3), `nGhost`, `RKOrder` (1-3), `reconOrder`, `useIGR`, `semiImplicit`
- `ExplicitParams`: cfl, constDt, maxDt, minDt
- `SemiImplicitParams`: cfl, constDt, maxDt, minDt, maxAcousticCFL, maxPressureIters, pressureTol, singlePressureSolve
- `IGRParams`: alphaCoeff, IGRIters, IGRWarmStartIters
- `MultiPhaseParams`: nPhases (0=single-phase), phases (`PhaseEOS phases[MAX_PHASES]`), alphaMin
- `RestartParams` (in `InputData`): file, checkpoint

## GPU Readiness

The codebase uses C-style architecture specifically designed for GPU offloading:

- **No virtual dispatch** — All dispatch uses enum + switch (EOS, Riemann solvers, pressure solvers, IC geometry)
- **No heap allocations in hot loops** — All scratch arrays pre-allocated at solver init time
- **No lambda captures** — ViscousFlux uses a static helper; time loop uses C function pointers + `void* ctx`
- **Flat arrays** — Multi-phase data stored as flat `double*` with stride (`alpha[phase * totalCells + cell]`), not `vector<vector<double>>`
- **Raw-pointer APIs** — All MixtureEOS functions take `const double*`/`const PhaseEOS*`, callable from device code
- **`gammaEff`/`piInfEff` always set on face states** — Reconstructor populates these for both single-phase and multi-phase, so Riemann solvers never need EOS calls

Remaining items for future GPU porting:
- AoS to SoA conversion for face reconstruction data (`PrimitiveState*` arrays)
- OpenACC/CUDA pragmas on compute loops
- Gauss-Seidel to Jacobi iteration for IGR (GS is inherently serial)

## Code Style

- C++17, no namespaces (except `ExpressionEvaluator` class)
- Headers use `#ifndef` include guards (not `#pragma once`)
- Plain C structs with `_init()` / `_free()` lifecycle functions
- Free functions with `module_action()` naming
- `SolutionState` holds all field data as flat `double*` arrays
- New cases should be JSON files in `cases/`; compiled C++ cases are for specialized post-processing only
