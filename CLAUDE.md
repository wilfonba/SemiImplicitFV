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
- `examples/` — legacy compiled example programs (standalone C++ files)

## Case System

**JSON input files** in `cases/<name>/<name>.jsonc` are the standard way to define simulations. The `sifv` driver reads a JSON file and runs the simulation without any C++ coding. VTK output is written into the case's own directory (e.g. `cases/1D_sod_shocktube/VTK/`).

Run cases via `run_case.sh`:
- `./run_case.sh <case>` — build `sifv` and run the JSON case
- `./run_case.sh --case-optimization <case>` — generate optimized C++ from JSON via `tools/codegen.py`, compile, and run
- `./run_case.sh --compiled <case>` — build and run a compiled `.cpp` source directly (for cases with custom post-processing)

When adding new cases, prefer JSON input files. Use compiled C++ only when the case requires logic not expressible in JSON (custom diagnostics, drag/lift computation, convergence studies, etc.).

### JSON Schema

Top-level sections: `config`, `eos`, `riemannSolver`, `mesh`, `boundaryConditions`, `timeLoop`, `output`, `initialConditions`, `immersedBoundaries`, `smoothing`. All sections except `config`, `mesh`, `timeLoop`, and `initialConditions` are optional.

Initial condition patches support `"box"`, `"sphere"`, `"plane"`, and `"analytic"` geometry types. Patch states inherit from the default state.

### Immersed Boundaries

The `"immersedBoundaries"` section defines solid bodies via the ghost-cell IBM. Only supported with the explicit solver (`"semiImplicit": false`). Body types: `"circle"`, `"rectangle"`, `"cylinder"`, `"rectangularPrism"`. Each body has `center`, shape-specific parameters (`radius`, `halfWidths`, `axis`), and `wallType` (`"NoSlip"` or `"Slip"`).

Parsed into `std::vector<IBBodyDef>` in `InputData`. The driver (`driver/main.cpp`) converts `IBBodyDef` structs into concrete `IBBody` subclasses, calls `ibm.classifyCells(mesh)`, and attaches via `rt.attachIBM(ibm, solver)`. The codegen (`tools/codegen.py`) emits equivalent C++ directly.

### Adding a New JSON Case

1. Create `cases/<name>/<name>.jsonc`
2. Define config, mesh, BCs, ICs, and any optional sections
3. Run: `./run_case.sh <name>`
4. VTK output appears in `cases/<name>/VTK/`

## Architecture

**Solvers**: `ExplicitSolver` (SSP-RK1/2/3 with acoustic CFL) and `SemiImplicitSolver` (advective CFL + implicit pressure). Both use shared RK utilities from `RKTimeStepping.hpp`.

**Riemann solvers**: `LFSolver` (Lax-Friedrichs), `RusanovSolver`, `HLLCSolver` — all inherit from `RiemannSolver`. Hot-path flux computation uses devirtualized free functions (`computeLFFlux`, `computeRusanovFlux`, `computeHLLCFlux`) dispatched via `RiemannSolverType` enum + switch in `computeFluxDirect()`. See `RiemannSolver.hpp` for the enum, `FluxConfig` struct, and inline dispatch function.

**Reconstruction**: WENO1/3/5 and UPWIND1/3/5 schemes in `Reconstruction.cpp`. Ghost cell count in `SimulationConfig::nGhost` must satisfy `requiredGhostCells()`. The `Reconstructor` always populates `gammaEff`/`piInfEff` on face states — for multi-phase from mixture EOS, for single-phase from the scalar EOS gamma/pInf passed at construction. This ensures Riemann solvers never need virtual EOS calls.

**EOS**: `IdealGasEOS` and `StiffenedGasEOS`, both inherit from `EquationOfState`. The base class provides virtual `gamma()` and `pInf()` accessors so solvers can extract scalar EOS parameters at construction time for use in devirtualized compute loops.

**Multi-phase**: `MixtureEOS` namespace (`MixtureEOS.hpp` / `MixtureEOS.cpp`) provides N-phase mixture routines — effective gamma/piInf from volume fractions, Wood's mixture sound speed, mixture pressure, and mixture total energy. All functions have raw-pointer overloads (`const double*`, `const PhaseEOS*`) for GPU readiness alongside `std::vector` convenience wrappers. Enabled by setting `config.multiPhaseParams.nPhases >= 2` with per-phase `{gamma, pInf}` in `PhaseEOS`. All N volume fractions (`alpha[k]` for k=0..nPhases-1) and N partial densities (`alphaRho[k]` for k=0..nPhases-1) are stored and advected in `SolutionState`. After each RK stage, alphas are clamped to `alphaMin` and normalized so `sum(alpha) = 1`. At faces, `gammaEff` and `piInfEff` are computed from reconstructed alphas via `MixtureEOS::effectiveGammaAndPiInf()`. Cell-center sound speed uses the full Wood's formula.

**IBM**: `ImmersedBoundaryMethod` (`ImmersedBoundary.hpp/cpp`) classifies cells as Fluid/Ghost/Dead and applies ghost-cell interpolation to enforce wall BCs on embedded bodies. Body shapes: `IBCircle`, `IBRectangle`, `IBCylinder`, `IBRectangularPrism`. Only works with `ExplicitSolver` (attached via `Runtime::attachIBM()`). In the JSON driver, IBM bodies are defined via `IBBodyDef` structs parsed from the `"immersedBoundaries"` section.

**IGR**: `IGRSolver` computes entropic pressure via Gauss-Seidel iteration on the elliptic equation. Controlled by `SimulationConfig::useIGR` and `IGRParams`.

**Mesh**: `RectilinearMesh` with ghost cells and boundary conditions (Periodic, Reflective, Outflow).

**Output**: `VTKWriter` produces `.vtr` and `.pvd` files. Multi-phase fields (`Alpha_k`, `AlphaRho_k`) are written automatically when present.

**Input parsing**: `InputParser` (`InputParser.hpp/cpp`) reads JSON/JSONC files into `InputData` structs. This includes `SimulationConfig`, mesh/EOS/BC parameters, initial condition patches, and `IBBodyDef` structs. The driver (`driver/main.cpp`) converts these data structs into runtime objects.

**Code generation**: `tools/codegen.py` reads a JSON case file and emits a standalone C++ `main()` with hardcoded parameters for maximum performance. Supports all JSON features including immersed boundaries.

## Key Configuration

All simulation parameters live in `SimulationConfig` (see `include/SimulationConfig.hpp`):
- `dim` (1-3), `nGhost`, `RKOrder` (1-3), `reconOrder`, `useIGR`, `semiImplicit`
- `ExplicitParams`: cfl, constDt, maxDt, minDt
- `SemiImplicitParams`: cfl, maxDt, minDt, maxPressureIters, pressureTol
- `IGRParams`: alphaCoeff, IGRIters, IGRWarmStartIters
- `MultiPhaseParams`: nPhases (0=single-phase), phases (vector of `PhaseEOS{gamma, pInf}`), alphaMin

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
