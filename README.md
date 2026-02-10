# SemiImplicitFV

A finite volume solver for the compressible Euler equations on rectilinear meshes in 1D, 2D, and 3D. Supports explicit (SSP Runge-Kutta) and semi-implicit (pressure-split) time integration, high-order WENO and upwind reconstruction, multiple Riemann solvers, and Information Geometric Regularization (IGR). Output is in VTK format for visualization with ParaView.

## Features

- **Explicit and semi-implicit time integration** — SSP-RK1/2/3 for explicit; advective CFL with implicit pressure correction for semi-implicit (Kwatra et al.)
- **High-order spatial reconstruction** — WENO and upwind schemes at 1st, 3rd, and 5th order
- **Riemann solvers** — Lax-Friedrichs, Rusanov, and HLLC
- **Equations of state** — Ideal gas and stiffened gas
- **Information Geometric Regularization (IGR)** — Entropic pressure via elliptic solve for improved stability
- **1D / 2D / 3D** on rectilinear (uniform) meshes with ghost cells
- **Boundary conditions** — Periodic, Reflective, Outflow, Slip Wall, No-Slip Wall
- **MPI parallelism** — Cartesian domain decomposition with non-blocking halo exchange
- **VTK output** — `.vtr` (serial), `.pvtr` (parallel), and `.pvd` (time series) for ParaView

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Run the Sod shock tube example
cd ../examples/1D_sod_shocktube
../../build/1D_sod_shocktube
```

Or use the convenience script:

```bash
./run_case.sh 1D_sod_shocktube
```

Output VTK files are written to a `VTK/` directory inside the example folder. Open the `.pvd` file in [ParaView](https://www.paraview.org/) to visualize the results.

## Building

### Requirements

- CMake 3.14+
- C++17 compiler
- MPI implementation (e.g., Open MPI, MPICH)
- OpenMP (optional)

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_EXAMPLES` | `ON` | Build example programs in `examples/` |
| `BUILD_TESTS` | `OFF` | Build tests |
| `ENABLE_OPENMP` | `OFF` | Enable OpenMP support |

MPI is always required and linked automatically.

```bash
mkdir build && cd build

# Release build (default)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Debug build (enables AddressSanitizer automatically)
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j
```

### Using `run_case.sh`

The `run_case.sh` script handles configuring, building, and running any example:

```bash
./run_case.sh 1D_sod_shocktube              # Build and run (1 MPI rank)
./run_case.sh --debug 1D_advection           # Debug build
./run_case.sh -n 4 1D_sod_shocktube         # Run with 4 MPI ranks
./run_case.sh --build-only 2D_riemann        # Build without running
./run_case.sh --list                         # List available cases
```

## Configuration

All simulation parameters are set through the `SimulationConfig` struct (defined in `include/SimulationConfig.hpp`). Key settings:

```cpp
SimulationConfig config;
config.dim = 2;                              // Spatial dimensions (1, 2, or 3)
config.nGhost = 3;                           // Ghost cells (must match reconstruction order)
config.RKOrder = 3;                          // Runge-Kutta order (1, 2, or 3)
config.reconOrder = ReconstructionOrder::WENO5;  // Reconstruction scheme
config.semiImplicit = false;                 // Use semi-implicit solver
config.useIGR = true;                        // Enable IGR

// Explicit solver parameters
config.explicitParams.cfl = 0.6;
config.explicitParams.maxDt = 1e-3;

// Semi-implicit solver parameters
config.semiImplicitParams.cfl = 0.8;
config.semiImplicitParams.maxPressureIters = 1000;
config.semiImplicitParams.pressureTol = 1e-6;

// IGR parameters
config.igrParams.alphaCoeff = 10.0;
config.igrParams.IGRIters = 5;
```

The `validate()` method checks consistency (e.g., ghost cell count matches reconstruction stencil, semi-implicit requires RK order 1).

### Reconstruction Orders

| Scheme | Order | Ghost Cells Required |
|---|---|---|
| `WENO1` / `UPWIND1` | 1st | 1 |
| `WENO3` / `UPWIND3` | 3rd | 2 |
| `WENO5` / `UPWIND5` | 5th | 3 |

WENO schemes include nonlinear shock-capturing weights. Upwind schemes use standard polynomial reconstruction.

## Writing a New Problem

Each example is a standalone `main.cpp` in its own subdirectory under `examples/`. CMake automatically discovers and builds them. To add a new case:

1. Create `examples/my_case/my_case.cpp`
2. Set up a `SimulationConfig` and create a mesh via `Runtime`
3. Choose an EOS, Riemann solver, and (optionally) IGR and pressure solvers
4. Attach the solver to the runtime and set initial/boundary conditions
5. Run the time loop

```cpp
#include "Runtime.hpp"
#include "IdealGasEOS.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "VTKSession.hpp"

using namespace SemiImplicitFV;

int main(int argc, char* argv[]) {
    Runtime runtime(argc, argv);

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 3;
    config.RKOrder = 3;
    config.reconOrder = ReconstructionOrder::WENO5;
    config.explicitParams.cfl = 0.6;

    auto mesh = runtime.createUniformMesh(100, 0.0, 1.0, config);
    runtime.setBoundaryCondition(Face::XLow, BoundaryCondition::Outflow);
    runtime.setBoundaryCondition(Face::XHigh, BoundaryCondition::Outflow);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0);
    auto riemann = std::make_shared<HLLCSolver>(eos, config);
    auto solver = std::make_shared<ExplicitSolver>(mesh, riemann, eos, nullptr, config);
    runtime.attachSolver(solver);

    SolutionState state;
    state.allocate(mesh);

    // Set initial conditions...
    // for (int i = 0; i < mesh.totalCells(); ++i) { ... }

    state.convertPrimitiveToConservativeVariables(mesh, *eos);

    VTKSession vtk("my_case", "VTK");
    double t = 0.0, tEnd = 0.2, dtOut = 0.01;
    double nextOut = 0.0;

    while (t < tEnd) {
        if (t >= nextOut) {
            vtk.write(mesh, state, t);
            nextOut += dtOut;
        }
        double dt = solver->step(config, mesh, state, tEnd - t);
        t += dt;
    }
    vtk.finalize();
    return 0;
}
```

Rebuild, and the new executable appears automatically:

```bash
./run_case.sh my_case
```

## MPI Execution

MPI is always linked. Run with `mpirun`:

```bash
mpirun -np 4 ./build/2D_riemann
```

Or via the helper script:

```bash
./run_case.sh -n 4 2D_riemann
```

The `Runtime` class handles domain decomposition, halo exchange, and parallel VTK output automatically. Each rank writes its own `.vtr` piece file, and rank 0 writes the `.pvtr` and `.pvd` metadata files.

## Project Structure

```
SemiImplicitFV/
├── include/               Header files
│   ├── SimulationConfig.hpp   Configuration struct
│   ├── SolutionState.hpp      Field data storage
│   ├── RectilinearMesh.hpp    Mesh with ghost cells
│   ├── ExplicitSolver.hpp     SSP-RK explicit solver
│   ├── SemiImplicitSolver.hpp Pressure-split semi-implicit solver
│   ├── RiemannSolver.hpp      Abstract Riemann solver interface
│   ├── LFSolver.hpp           Lax-Friedrichs
│   ├── RusanovSolver.hpp      Rusanov
│   ├── HLLCSolver.hpp         HLLC
│   ├── Reconstruction.hpp     WENO/upwind reconstruction
│   ├── IGR.hpp                Information Geometric Regularization
│   ├── EquationOfState.hpp    Abstract EOS interface
│   ├── IdealGasEOS.hpp        Ideal gas EOS
│   ├── StiffenedGasEOS.hpp    Stiffened gas EOS
│   ├── PressureSolver.hpp     Abstract pressure solver
│   ├── Runtime.hpp            MPI/serial runtime abstraction
│   ├── MPIContext.hpp         MPI domain decomposition
│   ├── HaloExchange.hpp       MPI ghost cell communication
│   ├── VTKWriter.hpp          VTK file I/O
│   └── VTKSession.hpp         VTK time-series management
├── src/                   Implementation files
├── examples/              Example problems
│   ├── 1D_advection/
│   ├── 1D_sod_shocktube/
│   ├── 2D_quasi1D_sod/
│   └── 2D_riemann/
├── CMakeLists.txt
└── run_case.sh            Build & run helper script
```

## Visualization

Output files are VTK XML RectilinearGrid format, viewable in [ParaView](https://www.paraview.org/):

1. Open the `.pvd` file in ParaView to load the full time series
2. Apply a color map to fields like `Density`, `Pressure`, or `Velocity`
3. Use the animation controls to step through time

Fields written per cell: density, velocity (u, v, w), pressure, temperature, and entropic pressure (sigma).

## License

See [LICENSE](LICENSE) for details.
