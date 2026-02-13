# Contributing to SemiImplicitFV

Thank you for your interest in contributing to SemiImplicitFV! This guide will help you understand how the project is organized, how to build and test your changes, and how to add new features.

## Getting Started

### Prerequisites

- CMake 3.14+
- C++17 compiler (GCC or Clang)
- MPI implementation (Open MPI, MPICH, etc.)
- OpenMP (optional)
- ParaView (for visualizing output)

### Building

```bash
git clone <repo-url>
cd SemiImplicitFV
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

For development, use a Debug build which automatically enables AddressSanitizer:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
```

Debug builds add `-fsanitize=address -fno-omit-frame-pointer` so memory errors are caught at runtime.

### Running Examples

Use the convenience script from the project root:

```bash
./run_case.sh 1D_sod_shocktube           # Build and run
./run_case.sh --debug 1D_advection        # Debug build with ASAN
./run_case.sh -n 4 2D_riemann             # Run with 4 MPI ranks
./run_case.sh --build-only 2D_riemann     # Build without running
./run_case.sh --list                      # List all available cases
./run_case.sh --clean 1D_sod_shocktube    # Clean rebuild
```

Output VTK files are written to a `VTK/` directory inside the example folder.

## Project Architecture

### Directory Layout

```
SemiImplicitFV/
├── include/       All header files (.hpp)
├── src/           All implementation files (.cpp)
├── examples/      Standalone example programs (one subdirectory per case)
├── CMakeLists.txt
└── run_case.sh
```

The library is built from all `src/*.cpp` files into a static library (`SemiImplicitFV`). Examples in `examples/` are auto-discovered by CMake — each subdirectory containing `.cpp` files becomes a separate executable linked against the library.

### Core Components

The solver pipeline follows this flow:

```
SimulationConfig  →  Mesh  →  EOS + Riemann Solver  →  Solver  →  Time Loop
```

**Mesh** (`RectilinearMesh`): Uniform rectilinear grid in 1D/2D/3D with ghost cells and boundary conditions.

**Equation of State** (abstract `EquationOfState` base):
- `IdealGasEOS` — ideal gas law
- `StiffenedGasEOS` — stiffened gas law
- `MixtureEOS` namespace — N-phase mixture routines (effective gamma, Wood's sound speed, mixture energy)

**Riemann Solvers** (abstract `RiemannSolver` base):
- `LFSolver` — Lax-Friedrichs
- `RusanovSolver` — Rusanov (local Lax-Friedrichs)
- `HLLCSolver` — HLLC (Toro's 3-wave solver)

**Reconstruction** (`Reconstruction.cpp`): WENO and upwind schemes at 1st, 3rd, and 5th order. Reconstructs primitive variables to cell faces.

**Solvers**:
- `ExplicitSolver` — SSP Runge-Kutta (RK1/2/3) with acoustic CFL constraint
- `SemiImplicitSolver` — advective CFL with implicit pressure correction (Kwatra et al.)

Both solvers share RK utilities from `RKTimeStepping`.

**Physics modules** (called during RHS computation):
- `ViscousFlux` — Newtonian viscous stress tensor
- `SurfaceTension` — capillary stress tensor (Schmidmayer et al. 2017)
- `IGRSolver` — Information Geometric Regularization (entropic pressure)
- Body forces — configured via `BodyForceParams`

**Parallelism**: `Runtime` abstracts MPI/serial execution. `MPIContext` handles Cartesian domain decomposition. `HaloExchange` provides non-blocking ghost cell communication.

**Output**: `VTKWriter` and `VTKSession` produce `.vtr` (serial), `.pvtr` (parallel), and `.pvd` (time series) files for ParaView.

### How the Solver Pipeline Works

Each time step, the solver:

1. **Reconstructs** primitive variables to cell faces (WENO or upwind)
2. **Computes Riemann fluxes** at each face using the chosen Riemann solver
3. **Accumulates** additional physics into the RHS vectors:
   - Viscous fluxes (if `config.hasViscosity()`)
   - Surface tension fluxes (if `config.hasSurfaceTension()`)
   - Body force source terms (if `config.hasBodyForce()`)
4. **Advances** the solution using Runge-Kutta time stepping
5. **Applies** boundary conditions and (optionally) IGR

The key RHS vectors are `rhsRho`, `rhsRhoU`, `rhsRhoV`, `rhsRhoW`, `rhsRhoE`, plus multi-phase arrays `rhsAlpha[k]` and `rhsAlphaRho[k]`.

## Code Style

### General Rules

- **C++17** standard
- **4-space indentation** (no tabs)
- **Namespace**: all code lives in `namespace SemiImplicitFV { ... }`
- **No** `.clang-format` or linter config currently — follow existing patterns

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Classes / Structs | PascalCase | `ExplicitSolver`, `RectilinearMesh` |
| Methods | camelCase | `step()`, `computeRHS()` |
| Member variables | camelCase with trailing underscore | `riemannSolver_`, `eos_`, `halo_` |
| Local variables | camelCase | `dt`, `idx`, `rhsRho` |
| Enums | PascalCase enum class | `ReconstructionOrder::WENO5` |
| File-scope constants | snake_case | `static constexpr double gamma_gas` |

### Header Files

Use `#ifndef` include guards (not `#pragma once`):

```cpp
#ifndef MY_HEADER_HPP
#define MY_HEADER_HPP

namespace SemiImplicitFV {

// ...

} // namespace SemiImplicitFV

#endif // MY_HEADER_HPP
```

### Ownership and Pointers

- **Shared ownership**: `std::shared_ptr` for EOS, Riemann solver, and IGR solver (passed into solver constructors)
- **Non-owning references**: raw pointers with trailing underscore (e.g., `HaloExchange* halo_`)
- **Value types**: `SimulationConfig`, `RectilinearMesh`, `SolutionState` are passed by const reference

## How to Add a New Feature

### Adding a New Physics Module

This is the most common type of contribution. Follow the pattern established by `ViscousFlux` and `SurfaceTension`:

**1. Create the header** (`include/MyFeature.hpp`):

```cpp
#ifndef MY_FEATURE_HPP
#define MY_FEATURE_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <vector>

namespace SemiImplicitFV {

void addMyFeatureFluxes(
    const SimulationConfig& config,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    double myParam,
    std::vector<double>& rhsRhoU,
    std::vector<double>& rhsRhoV,
    std::vector<double>& rhsRhoW,
    std::vector<double>& rhsRhoE);

} // namespace SemiImplicitFV

#endif // MY_FEATURE_HPP
```

The function signature takes the config, mesh, and current state as const inputs, plus references to the RHS vectors to accumulate into. This pattern allows the solver to call multiple physics modules that each add their contributions.

**2. Implement it** (`src/MyFeature.cpp`):

The typical structure is:
- Loop over faces in each active direction (X, and Y/Z if `config.dim >= 2`/`3`)
- Compute gradients at face centers (central differences normal, averaged transverse)
- Calculate flux contributions
- Accumulate into the RHS vectors

**3. Add config parameters** (`include/SimulationConfig.hpp`):

```cpp
struct MyFeatureParams {
    double myParam = 0.0;  // 0 = disabled
};

// Inside SimulationConfig:
MyFeatureParams myFeatureParams;

bool hasMyFeature() const { return myFeatureParams.myParam > 0.0; }
```

Add validation in the `validate()` method.

**4. Integrate into the solvers** (`src/ExplicitSolver.cpp` and `src/SemiImplicitSolver.cpp`):

In the `computeRHS()` method, after existing physics modules:

```cpp
if (config.hasMyFeature())
    addMyFeatureFluxes(config, mesh, state, config.myFeatureParams.myParam,
                       rhsRhoU_, rhsRhoV_, rhsRhoW_, rhsRhoE_);
```

**5. Add the source file to CMakeLists.txt**:

Add `src/MyFeature.cpp` to the `SOURCES` list in the top-level `CMakeLists.txt`.

**6. Write an example** to demonstrate and validate the feature (see below).

### Adding a New Riemann Solver

1. Create `include/MySolver.hpp` and `src/MySolver.cpp`
2. Inherit from `RiemannSolver` and implement `computeFlux()` and `maxWaveSpeed()`
3. Add the source file to the `SOURCES` list in `CMakeLists.txt`

### Adding a New Equation of State

1. Create `include/MyEOS.hpp` and `src/MyEOS.cpp`
2. Inherit from `EquationOfState`
3. Add the source file to the `SOURCES` list in `CMakeLists.txt`

### Adding a New Example

Each example is a standalone program in its own subdirectory. CMake auto-discovers them — no `CMakeLists.txt` edits needed.

**1. Create the directory and source file:**

```bash
mkdir examples/2D_my_case
```

Create `examples/2D_my_case/2D_my_case.cpp`. The filename must match the directory name.

**2. Follow the standard example structure:**

```cpp
#include "Runtime.hpp"
#include "IdealGasEOS.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

using namespace SemiImplicitFV;

int main(int argc, char** argv) {
    // 1. Create runtime
    Runtime rt(argc, argv);

    // 2. Configure simulation
    SimulationConfig config;
    config.dim = 2;
    config.nGhost = 3;
    config.RKOrder = 3;
    config.reconOrder = ReconstructionOrder::WENO5;
    config.explicitParams.cfl = 0.6;
    config.validate();

    // 3. Create mesh and set boundary conditions
    RectilinearMesh mesh = rt.createUniformMesh(config, /*nx=*/100, ...);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow, BoundaryCondition::Outflow);
    // ... set all boundaries

    // 4. Allocate state and set initial conditions
    SolutionState state;
    state.allocate(mesh.totalCells(), config);
    // ... initialize rho, velU, velV, pres, etc.

    // 5. Create EOS, Riemann solver, and solver
    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemann = std::make_shared<HLLCSolver>(eos, config);
    ExplicitSolver solver(mesh, riemann, eos, nullptr, config);
    rt.attachSolver(solver, mesh);

    // 6. Run time loop with VTK output
    VTKSession vtk(rt, "2D_my_case", mesh);
    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = 1.0, .outputInterval = 0.1, .printInterval = 100});

    return 0;
}
```

**3. Build and run:**

```bash
./run_case.sh 2D_my_case
```

The executable is auto-generated. VTK output goes to `examples/2D_my_case/VTK/`.

### Key Things to Remember

- **Ghost cells**: `config.nGhost` must be >= `config.requiredGhostCells()` for the chosen reconstruction order. The `validate()` method checks this.
- **Multi-phase**: Gate multi-phase logic with `config.isMultiPhase()`. Use `MixtureEOS` namespace functions for mixture quantities.
- **Both solvers**: When adding physics, integrate into both `ExplicitSolver` and `SemiImplicitSolver` so the feature works with either time integration method.
- **RHS accumulation**: Physics modules *add to* the RHS vectors; they do not overwrite them. The solver zeroes the vectors at the start of each RHS evaluation.
- **Boundary conditions**: Available types are `Periodic`, `Reflective`, `Outflow`, `SlipWall`, and `NoSlipWall`.

## Building and Verifying Your Changes

### Debug Build (Recommended During Development)

```bash
./run_case.sh --debug <your_example>
```

This enables AddressSanitizer to catch memory errors, buffer overflows, and use-after-free bugs.

### Verification Checklist

Before submitting a contribution:

1. **Clean build succeeds**: `./run_case.sh --clean <your_example>`
2. **Debug build passes**: `./run_case.sh --debug <your_example>` (no ASAN errors)
3. **Existing examples still work**: run a few existing cases to check for regressions
4. **MPI still works**: `./run_case.sh -n 2 <a_2D_example>` (if your changes touch parallel code)
5. **Output is correct**: open VTK files in ParaView and verify results are physically reasonable

### Convergence Testing

For new numerical features, consider adding a convergence study to your example (see `examples/1D_advection/` for the pattern):

- Run at multiple grid resolutions
- Compute L1, L2, and L-infinity error norms against an analytical solution
- Verify the expected convergence order
- Optionally include a Python plotting script and `convergence.png`

## Submitting Changes

1. Create a feature branch from `master`
2. Make your changes following the code style and patterns above
3. Verify your changes (see checklist above)
4. Write a clear commit message describing what was added and why
5. Open a pull request against `master`

## Questions?

If you're unsure about an approach or where something should go, open an issue to discuss before starting work. The architecture section above and the existing code in `src/` are the best references for how things fit together.
