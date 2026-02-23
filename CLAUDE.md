# SemiImplicitFV

Finite volume solver for compressible Euler equations on rectilinear meshes (1D/2D/3D) with explicit and semi-implicit time integration and Information Geometric Regularization (IGR).

## Build and Run

Use `run_case.sh` to configure, build, and run cases automatically:

```bash
./run_case.sh <case>                         # Build and run a JSON case
./run_case.sh --debug <case>                 # Debug build (AddressSanitizer)
./run_case.sh -n 4 <case>                    # Run with 4 MPI ranks
./run_case.sh --case-optimization <case>     # Codegen optimized build
./run_case.sh --compiled <case>              # Build compiled C++ case
./run_case.sh --petsc <case>                 # Enable PETSc pressure solver
./run_case.sh --nsys <case>                  # Profile with Nsight Systems (NVTX)
./run_case.sh --list                         # List available cases
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
- `tools/` — code generation (`codegen.py`: JSON → optimized C++) and utilities

## Case System

**JSON input files** in `cases/<name>/<name>.jsonc` are the standard way to define simulations. The `sifv` driver reads a JSON file and runs the simulation without any C++ coding. VTK output is written into the case's own directory (e.g. `cases/1D_sod_shocktube/VTK/`).

Run cases via `run_case.sh`:
- `./run_case.sh <case>` — build `sifv` and run the JSON case
- `./run_case.sh --case-optimization <case>` — generate optimized C++ from JSON via `tools/codegen.py`, compile, and run
- `./run_case.sh --compiled <case>` — build and run a compiled `.cpp` source directly (for cases with custom post-processing)

When adding new cases, prefer JSON input files. Use compiled C++ only when the case requires logic not expressible in JSON (custom diagnostics, drag/lift computation, convergence studies, etc.).

### JSON Schema

Top-level sections: `config`, `eos`, `riemannSolver`, `pressureSolver`, `mesh`, `boundaryConditions`, `timeLoop`, `output`, `initialConditions`, `smoothing`, `restart`. All sections except `config`, `mesh`, `timeLoop`, and `initialConditions` are optional.

The `pressureSolver` key selects the pressure solver for semi-implicit runs: `"GaussSeidel"` (default) or `"PETSc"` (CG + GAMG algebraic multigrid via PETSc; requires `--petsc` build flag).

The `output` section supports a `"format"` field: `"VTKText"` (default, ASCII) or `"VTKRaw"` (appended raw binary, compact and fast).

Initial condition patches support `"box"`, `"sphere"`, `"plane"`, and `"analytic"` geometry types. Patch states inherit from the default state.

### Adding a New JSON Case

1. Create `cases/<name>/<name>.jsonc`
2. Define config, mesh, BCs, ICs, and any optional sections
3. Run: `./run_case.sh <name>`
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

Only conservative variables are saved. Primitives are recomputed on restart via `convertConservativeToPrimitiveVariables()`.

### Example workflow

```bash
# 1. Run with checkpoints enabled (add "restart": {"checkpoint": true} to the JSONC)
./run_case.sh 1D_sod_shocktube

# 2. Job gets killed at wall-time limit...

# 3. Restart: add "file": "Checkpoint/checkpoint.{rank}.bin" to the "restart" section
./run_case.sh 1D_sod_shocktube
```

## Architecture

**Solvers**: `ExplicitSolver` (SSP-RK1/2/3 with acoustic CFL) and `SemiImplicitSolver` (advective CFL + implicit pressure). Both use shared RK utilities from `RKTimeStepping.hpp`.

**Riemann solvers**: `LFSolver` (Lax-Friedrichs), `RusanovSolver`, `HLLCSolver` — all inherit from `RiemannSolver`. Hot-path flux computation uses devirtualized free functions (`computeLFFlux`, `computeRusanovFlux`, `computeHLLCFlux`) dispatched via `RiemannSolverType` enum + switch in `computeFluxDirect()`. See `RiemannSolver.hpp` for the enum, `FluxConfig` struct, and inline dispatch function.

**Reconstruction**: WENO1/3/5 and UPWIND1/3/5 schemes in `Reconstruction.cpp`. Ghost cell count in `SimulationConfig::nGhost` must satisfy `requiredGhostCells()`. The `Reconstructor` always populates `gammaEff`/`piInfEff` on face states — for multi-phase from mixture EOS, for single-phase from the scalar EOS gamma/pInf passed at construction. This ensures Riemann solvers never need virtual EOS calls.

**EOS**: `IdealGasEOS` and `StiffenedGasEOS`, both inherit from `EquationOfState`. The base class provides virtual `gamma()` and `pInf()` accessors so solvers can extract scalar EOS parameters at construction time for use in devirtualized compute loops.

**Multi-phase**: `MixtureEOS` namespace (`MixtureEOS.hpp` / `MixtureEOS.cpp`) provides N-phase mixture routines — effective gamma/piInf from volume fractions, Wood's mixture sound speed, mixture pressure, and mixture total energy. All functions have raw-pointer overloads (`const double*`, `const PhaseEOS*`) for GPU readiness alongside `std::vector` convenience wrappers. Enabled by setting `config.multiPhaseParams.nPhases >= 2` with per-phase `{gamma, pInf}` in `PhaseEOS`. All N volume fractions (`alpha[k]` for k=0..nPhases-1) and N partial densities (`alphaRho[k]` for k=0..nPhases-1) are stored and advected in `SolutionState`. After each RK stage, alphas are clamped to `alphaMin` and normalized so `sum(alpha) = 1`. At faces, `gammaEff` and `piInfEff` are computed from reconstructed alphas via `MixtureEOS::effectiveGammaAndPiInf()`. Cell-center sound speed uses the full Wood's formula.

**IGR**: `IGRSolver` computes entropic pressure via Gauss-Seidel iteration on the elliptic equation. Controlled by `SimulationConfig::useIGR` and `IGRParams`.

**Pressure solvers**: `GaussSeidelPressureSolver` (default) for the semi-implicit pressure equation. `PETScPressureSolver` uses CG + GAMG algebraic multigrid via PETSc for mesh-independent convergence; enabled with `--petsc` build flag and `"pressureSolver": "PETSc"` in JSON.

**Mesh**: `RectilinearMesh` with ghost cells and boundary conditions (Periodic, Reflective, Outflow).

**Output**: `VTKWriter` produces `.vtr` and `.pvd` files in ASCII (`VTKText`) or appended raw binary (`VTKRaw`) format. Multi-phase fields (`Alpha_k`, `AlphaRho_k`) are written automatically when present.

**Profiling**: `NvtxRange.hpp` provides RAII-scoped NVTX ranges for Nsight Systems profiling. Enabled with `--nsys` build/run flag which sets `ENABLE_NVTX=ON` and wraps execution with `nsys profile`.

**Input parsing**: `InputParser` (`InputParser.hpp/cpp`) reads JSON/JSONC files into `InputData` structs. This includes `SimulationConfig`, mesh/EOS/BC parameters, and initial condition patches. The driver (`driver/main.cpp`) converts these data structs into runtime objects.

**Code generation**: `tools/codegen.py` reads a JSON case file and emits a standalone C++ `main()` with hardcoded parameters for maximum performance.

## Key Configuration

All simulation parameters live in `SimulationConfig` (see `include/SimulationConfig.hpp`):
- `dim` (1-3), `nGhost`, `RKOrder` (1-3), `reconOrder`, `useIGR`, `semiImplicit`
- `ExplicitParams`: cfl, constDt, maxDt, minDt
- `SemiImplicitParams`: cfl, constDt, maxDt, minDt, maxAcousticCFL, maxPressureIters, pressureTol, singlePressureSolve
- `IGRParams`: alphaCoeff, IGRIters, IGRWarmStartIters
- `MultiPhaseParams`: nPhases (0=single-phase), phases (vector of `PhaseEOS{gamma, pInf}`), alphaMin
- `RestartParams` (in `InputData`): file, checkpoint

## GPU Readiness (OpenACC)

The compute-path code has been refactored to eliminate patterns incompatible with GPU offloading:

- **No virtual dispatch in hot loops** — Riemann solver flux computation uses free functions + enum dispatch (`RiemannSolverType` / `computeFluxDirect()`). EOS calls in time step computation, pressure solve, and correction step are inlined using scalar gamma/pInf.
- **No per-cell heap allocations** — All scratch arrays (`scratchAlphas_`, `scratchAlphaRhos_`) are pre-allocated at solver construction time.
- **No lambda captures in compute paths** — ViscousFlux uses a static helper function instead of a lambda.
- **Raw-pointer MixtureEOS overloads** — `mixturePressure`, `mixtureSoundSpeed`, `mixtureTotalEnergy` all have `const double*`/`const PhaseEOS*` overloads callable from device code.
- **`gammaEff`/`piInfEff` always set on face states** — Reconstructor populates these for both single-phase and multi-phase, so Riemann solvers never fall back to virtual EOS calls.

Remaining items for future GPU porting:
- AoS → SoA conversion for face reconstruction data (`std::vector<PrimitiveState>`)
- Flat multi-phase arrays (`vector<vector<double>>` → single flat vector with stride)
- Gauss-Seidel → Jacobi iteration for IGR and pressure solvers (inherently serial)

## Code Style

- C++17, namespace `SemiImplicitFV`
- Headers use `#ifndef` include guards (not `#pragma once`)
- Solver classes take shared pointers to EOS and Riemann solver
- `SolutionState` holds all field data (rho, momentum, energy, sigma, primitives, and for multi-phase: alpha[k], alphaRho[k])
- New cases should be JSON files in `cases/`; compiled C++ cases are for specialized post-processing only
