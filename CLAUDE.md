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

**Riemann solvers**: `LFSolver` (Lax-Friedrichs), `RusanovSolver`, `HLLCSolver` — all inherit from `RiemannSolver`.

**Reconstruction**: WENO1/3/5 and UPWIND1/3/5 schemes in `Reconstruction.cpp`. Ghost cell count in `SimulationConfig::nGhost` must satisfy `requiredGhostCells()`.

**EOS**: `IdealGasEOS` and `StiffenedGasEOS`, both inherit from `EquationOfState`.

**Multi-phase**: `MixtureEOS` namespace (`MixtureEOS.hpp` / `MixtureEOS.cpp`) provides N-phase mixture routines — effective gamma/piInf from volume fractions, Wood's mixture sound speed, mixture pressure, and mixture total energy. Enabled by setting `config.multiPhaseParams.nPhases >= 2` with per-phase `{gamma, pInf}` in `PhaseEOS`. Volume fractions (`alpha[k]` for k=0..nPhases-2) and partial densities (`alphaRho[k]` for k=0..nPhases-1) are stored in `SolutionState` and reconstructed to faces. At faces, `gammaEff` and `piInfEff` are computed from reconstructed alphas via `MixtureEOS::effectiveGammaAndPiInf()`. Cell-center sound speed uses the full Wood's formula.

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

## Code Style

- C++17, namespace `SemiImplicitFV`
- Headers use `#ifndef` include guards (not `#pragma once`)
- Solver classes take shared pointers to EOS and Riemann solver
- `SolutionState` holds all field data (rho, momentum, energy, sigma, primitives, and for multi-phase: alpha[k], alphaRho[k])
