# SemiImplicitFV

Finite volume solver for compressible Euler equations on rectilinear meshes (1D/2D/3D) with explicit and semi-implicit time integration and Information Geometric Regularization (IGR).

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

CMake options:
- `-DBUILD_EXAMPLES=ON` (default ON) — builds all examples in `examples/`
- `-DBUILD_TESTS=ON` (default OFF)
- `-DENABLE_OPENMP=ON` (default OFF)
- Debug builds enable AddressSanitizer automatically

## Project Structure

- `include/` — all headers
- `src/` — library source files
- `examples/` — standalone example programs (each subdirectory becomes an executable)

## Architecture

**Solvers**: `ExplicitSolver` (SSP-RK1/2/3 with acoustic CFL) and `SemiImplicitSolver` (advective CFL + implicit pressure). Both use shared RK utilities from `RKTimeStepping.hpp`.

**Riemann solvers**: `LFSolver` (Lax-Friedrichs), `RusanovSolver`, `HLLCSolver` — all inherit from `RiemannSolver`. Hot-path flux computation uses devirtualized free functions (`computeLFFlux`, `computeRusanovFlux`, `computeHLLCFlux`) dispatched via `RiemannSolverType` enum + switch in `computeFluxDirect()`. See `RiemannSolver.hpp` for the enum, `FluxConfig` struct, and inline dispatch function.

**Reconstruction**: WENO1/3/5 and UPWIND1/3/5 schemes in `Reconstruction.cpp`. Ghost cell count in `SimulationConfig::nGhost` must satisfy `requiredGhostCells()`. The `Reconstructor` always populates `gammaEff`/`piInfEff` on face states — for multi-phase from mixture EOS, for single-phase from the scalar EOS gamma/pInf passed at construction. This ensures Riemann solvers never need virtual EOS calls.

**EOS**: `IdealGasEOS` and `StiffenedGasEOS`, both inherit from `EquationOfState`. The base class provides virtual `gamma()` and `pInf()` accessors so solvers can extract scalar EOS parameters at construction time for use in devirtualized compute loops.

**Multi-phase**: `MixtureEOS` namespace (`MixtureEOS.hpp` / `MixtureEOS.cpp`) provides N-phase mixture routines — effective gamma/piInf from volume fractions, Wood's mixture sound speed, mixture pressure, and mixture total energy. All functions have raw-pointer overloads (`const double*`, `const PhaseEOS*`) for GPU readiness alongside `std::vector` convenience wrappers. Enabled by setting `config.multiPhaseParams.nPhases >= 2` with per-phase `{gamma, pInf}` in `PhaseEOS`. All N volume fractions (`alpha[k]` for k=0..nPhases-1) and N partial densities (`alphaRho[k]` for k=0..nPhases-1) are stored and advected in `SolutionState`. After each RK stage, alphas are clamped to `alphaMin` and normalized so `sum(alpha) = 1`. At faces, `gammaEff` and `piInfEff` are computed from reconstructed alphas via `MixtureEOS::effectiveGammaAndPiInf()`. Cell-center sound speed uses the full Wood's formula.

**IGR**: `IGRSolver` computes entropic pressure via Gauss-Seidel iteration on the elliptic equation. Controlled by `SimulationConfig::useIGR` and `IGRParams`.

**Mesh**: `RectilinearMesh` with ghost cells and boundary conditions (Periodic, Reflective, Outflow).

**Output**: `VTKWriter` produces `.vtr` and `.pvd` files. Multi-phase fields (`Alpha_k`, `AlphaRho_k`) are written automatically when present.

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
