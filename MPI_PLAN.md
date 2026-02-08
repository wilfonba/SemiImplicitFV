# MPI Parallelization Plan for SemiImplicitFV

## Strategy Overview

The rectilinear grid structure maps naturally to **Cartesian domain decomposition** via `MPI_Cart_create`. Each rank owns a local subdomain of the global grid, creates its own `RectilinearMesh` with local dimensions, and fills inter-rank ghost cells via halo exchanges. Physical boundary conditions remain unchanged — they only apply at ranks that touch the domain boundary.

The VTK writer is already MPI-ready: each rank calls `writeVTR` for its piece, rank 0 calls `writePVTR` + `writePVD`.

## Halo Width

The required halo width equals `nGhost`, which is set to match the reconstruction stencil:

| Reconstruction | Stencil | `nGhost` Required |
|---|---|---|
| WENO1 | `[i-1, i]` | 1 |
| WENO3 | `[i-2, i+1]` | 2 |
| WENO5 | `[i-3, i+2]` | 3+ |

The pressure Laplacian, IGR, velocity gradients, and correction step all use a 6-neighbor stencil (width 1), which is always satisfied by the reconstruction's wider halo. The existing `nGhost=4` accommodates everything.

## Implementation Order

### Phase 1: MPI Infrastructure

**New file: `include/MPIContext.hpp`**

```cpp
#include <mpi.h>
#include <array>

struct MPIContext {
    MPI_Comm cartComm;           // Cartesian communicator
    int rank, size;
    std::array<int, 3> dims;     // Processes per direction {px, py, pz}
    std::array<int, 3> coords;   // This rank's position in grid
    std::array<int, 6> neighbors; // {xLow, xHigh, yLow, yHigh, zLow, zHigh}
                                  // MPI_PROC_NULL at physical boundaries

    // Global grid info
    int globalNx, globalNy, globalNz;

    // Local grid info (this rank's portion)
    int localNx, localNy, localNz;
    std::array<int, 6> localExtent; // {i0,i1,j0,j1,k0,k1} in global coords

    // Local node coordinates (sliced from global)
    std::vector<double> localXNodes, localYNodes, localZNodes;

    static MPIContext create(int globalNx, int globalNy, int globalNz,
                             const std::vector<double>& xNodes,
                             const std::vector<double>& yNodes,
                             const std::vector<double>& zNodes,
                             int dim);

    bool isPhysicalBoundary(int face) const {
        return neighbors[face] == MPI_PROC_NULL;
    }
};
```

**Key decisions in `create()`:**
- Call `MPI_Dims_create(size, dim, dims)` to auto-partition ranks. For 2D, set `dims[2]=1`. For 1D, set `dims[1]=dims[2]=1`.
- Call `MPI_Cart_create` with `periods = {false, false, false}` (periodic handled separately below).
- Call `MPI_Cart_shift` for each active direction to get neighbor ranks.
- Divide cells: `localNx = globalNx / px` with remainder distributed to first `globalNx % px` ranks. Same for y, z.
- Slice the global node arrays: rank at coord `(cx,cy,cz)` gets nodes `[startI, startI + localNx]` (inclusive, `localNx+1` nodes for `localNx` cells).

**Periodic boundaries:** If the global mesh has periodic BCs on a face, pass `periods[dir]=true` to `MPI_Cart_create`. Then `MPI_Cart_shift` returns the wrapped neighbor automatically and `MPI_PROC_NULL` is never returned for that direction, so the physical BC code is never triggered. The halo exchange handles the wrap.

### Phase 2: Halo Exchange

**New file: `include/HaloExchange.hpp`**

This is the core communication primitive. Every time `applyBoundaryConditions` is called, inter-rank boundaries need halo data from neighbors instead of BC fills.

```cpp
class HaloExchange {
public:
    HaloExchange(const MPIContext& mpi, const RectilinearMesh& mesh);

    // Exchange all conserved/primitive fields needed for the current operation
    void exchangeState(SolutionState& state, VarSet varSet) const;

    // Exchange a single scalar field (for pressure solver)
    void exchangeScalar(std::vector<double>& field) const;

private:
    const MPIContext& mpi_;
    const RectilinearMesh& mesh_;

    // Pre-allocated pack/unpack buffers (one per face, sized for halo slab)
    std::vector<double> sendBuf_[6], recvBuf_[6];

    void packFace(int face, const std::vector<double>& field) const;
    void unpackFace(int face, std::vector<double>& field) const;
};
```

**Packing/unpacking geometry (for x-direction, `nGhost=ng`):**

| Buffer | Cells packed | Description |
|---|---|---|
| Send to xLow neighbor | `i in [0, ng)` | First `ng` physical layers |
| Send to xHigh neighbor | `i in [nx-ng, nx)` | Last `ng` physical layers |
| Recv from xLow neighbor | `i in [-ng, 0)` | Low ghost layers |
| Recv from xHigh neighbor | `i in [nx, nx+ng)` | High ghost layers |

Each slab has `ng * nyTotal * nzTotal` cells. For WENO5 with `ng=4` on a 100x100 local grid, each x-slab is `4 * 108 * 1 = 432` doubles ~ 3.4 KB — very small, latency-dominated.

**Communication pattern per direction:**
```cpp
// Non-blocking: overlap both directions simultaneously
MPI_Isend(sendLow,  ..., neighbor[xLow],  tagLow,  ...);
MPI_Irecv(recvLow,  ..., neighbor[xLow],  tagHigh, ...);
MPI_Isend(sendHigh, ..., neighbor[xHigh], tagHigh, ...);
MPI_Irecv(recvHigh, ..., neighbor[xHigh], tagLow,  ...);
MPI_Waitall(4, requests, statuses);
```

Use `MPI_PROC_NULL` for physical boundary faces — MPI treats sends/recvs to `MPI_PROC_NULL` as no-ops, so no special-casing needed.

**For `exchangeState`**: pack all N fields into one contiguous buffer per face for fewer MPI calls. A `SolutionState` has ~11 active fields; batching is much better than exchanging each field separately.

**Onion-peel ordering still applies:** Exchange x-halos first, then y, then z. This correctly fills edge and corner ghosts in multi-D, matching the existing `applyBoundaryConditions` strategy.

### Phase 3: Modify RectilinearMesh Boundary Conditions

The key change: `applyBoundaryConditions` and `fillScalarGhosts` must skip BC fills on faces that have an MPI neighbor and instead let the halo exchange handle those faces.

**Option A (minimal change):** Add an `MPIContext*` member to `RectilinearMesh`. In `fillGhostX/Y/Z`, check `mpi->isPhysicalBoundary(face)` before applying BC logic. Call `HaloExchange::exchangeState()` before or interleaved with BC fills.

**Option B (cleaner):** Create a new method:
```cpp
void RectilinearMesh::applyBoundaryConditions(
    SolutionState& state, VarSet varSet,
    const HaloExchange* halo);  // nullptr = serial (existing behavior)
```

When `halo` is non-null, the method:
1. For each direction (x, then y, then z):
   - Exchange halos for that direction via `halo->exchangeDirection(dir, state)`
   - Apply physical BCs only on faces where `mpi.isPhysicalBoundary(face)`

This preserves the onion-peel ordering while interleaving MPI communication.

**`fillScalarGhosts`** needs the same treatment (called by pressure solver).

### Phase 4: Global Reductions

These are the synchronization points:

**1. CFL timestep computation** (`ExplicitSolver::computeTimeStep`, `SemiImplicitSolver::computeAdvectiveTimeStep`):
```cpp
// After local reduction:
double localMaxSpeed = maxSpeed;
double localMinDx = minDx;
MPI_Allreduce(&localMaxSpeed, &maxSpeed, 1, MPI_DOUBLE, MPI_MAX, cartComm);
MPI_Allreduce(&localMinDx, &minDx, 1, MPI_DOUBLE, MPI_MIN, cartComm);
```

**2. Pressure solver convergence** (`JacobiPressureSolver`, `GaussSeidelPressureSolver`):
```cpp
// After local residual computation:
double localMaxResidual = maxResidual;
MPI_Allreduce(&localMaxResidual, &maxResidual, 1, MPI_DOUBLE, MPI_MAX, cartComm);
```
Plus a `halo->exchangeScalar(pressure)` at the start of each iteration.

**3. Optional diagnostics** (e.g., `max|sigma|` in examples): `MPI_Allreduce` with `MPI_MAX`.

### Phase 5: Modify Solvers

**ExplicitSolver changes:**
- `computeTimeStep()`: add `MPI_Allreduce` after local CFL scan
- `step()`: no structural change — it calls `applyBoundaryConditions` which now handles halos
- `computeRHS()`: no change — operates on local mesh with ghost cells already filled

**SemiImplicitSolver changes:**
- `computeAdvectiveTimeStep()`: add `MPI_Allreduce`
- `advectionStep()`: no change (uses reconstruction + Riemann, local after BC fill)
- `advectPressure()`: needs ghost cells filled before accessing `i+/-1` neighbors — already done by `applyBoundaryConditions`
- `solvePressure()`: modify iterative solver loop:
  ```
  for each iteration:
      halo->exchangeScalar(pressure)      // NEW
      fillScalarGhosts(pressure)           // physical BCs only
      compute new pressure (Jacobi sweep)
      MPI_Allreduce on residual            // NEW
      if converged: break
  ```
- `correctionStep()`: needs pressure ghost cells (already filled by solvePressure)

**Pressure solver note:** Gauss-Seidel is inherently sequential and won't parallelize well. The Jacobi solver is the natural choice for MPI. If convergence is too slow, consider:
- Red-black Gauss-Seidel (parallelizes with 2 halo exchanges per iteration)
- Switching to a preconditioned conjugate gradient solver later

### Phase 6: VTK Output

Already designed for this. Each rank:
```cpp
if (outputTime) {
    std::string vtrFile = "output_" + std::to_string(rank) + "_" + std::to_string(fileNum) + ".vtr";
    VTKWriter::writeVTR("VTK/" + vtrFile, mesh, state, mpi.localExtent);

    if (rank == 0) {
        // Gather all piece filenames and extents (MPI_Gather or hardcode from MPIContext)
        VTKWriter::writePVTR("VTK/output_" + std::to_string(fileNum) + ".pvtr",
                             globalMesh, state, allExtents, allFiles);
        VTKWriter::writePVD("VTK/output.pvd", "a", time,
                            "output_" + std::to_string(fileNum) + ".pvtr");
    }
}
```

### Phase 7: CMake Changes

```cmake
option(ENABLE_MPI "Enable MPI parallelization" OFF)

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(SemiImplicitFV PUBLIC MPI::MPI_CXX)
    target_compile_definitions(SemiImplicitFV PUBLIC ENABLE_MPI)
endif()
```

Guard all MPI code with `#ifdef ENABLE_MPI` so serial builds still work.

### Phase 8: Example Updates

Wrap `main()` with `MPI_Init`/`MPI_Finalize`. Replace `RectilinearMesh::createUniform(globalN, ...)` with local mesh creation from `MPIContext`:
```cpp
MPI_Init(&argc, &argv);
MPIContext mpi = MPIContext::create(globalNx, globalNy, globalNz, ...);

RectilinearMesh mesh = RectilinearMesh(config,
    mpi.localXNodes, mpi.localYNodes, mpi.localZNodes);

// Set BCs only on physical boundary faces
if (mpi.isPhysicalBoundary(RectilinearMesh::XLow))
    mesh.setBoundaryCondition(RectilinearMesh::XLow, BoundaryCondition::Outflow);
// ... etc

HaloExchange halo(mpi, mesh);
// Pass halo to solver or BC routines
```

## Recommended Implementation Sequence

| Step | What | Files | Test |
|---|---|---|---|
| 1 | `MPIContext` + domain splitting | `MPIContext.hpp/.cpp` | Unit test: verify local dims sum to global, neighbor ranks correct |
| 2 | `HaloExchange` | `HaloExchange.hpp/.cpp` | Fill local state with `rank*1000 + localIndex`, exchange, verify ghost values match neighbor's interior |
| 3 | Modify `applyBoundaryConditions` | `RectilinearMesh.hpp/.cpp` | Run 1D sod tube split across 2 ranks, compare to serial |
| 4 | Add CFL `Allreduce` | `ExplicitSolver.cpp` | Run 1D sod on 2+ ranks, verify dt matches serial |
| 5 | VTK parallel output | Examples | Open `.pvtr` in ParaView, verify continuous field across pieces |
| 6 | Pressure solver halos + residual | `JacobiPressureSolver.cpp`, `SemiImplicitSolver.cpp` | Run semi-implicit 1D case on 2+ ranks |
| 7 | IGR halos | `SemiImplicitSolver.cpp` | Enable IGR on multi-rank case |

## Neighbor Access Summary

| Component | Neighbor Range | Width | When Used |
|---|---|---|---|
| Reconstruction WENO1 | `i-1, i` | 1 ghost layer | Always |
| Reconstruction WENO3 | `i-2` to `i+1` | 2 ghost layers | Standard |
| Reconstruction WENO5 | `i-3` to `i+2` | 3 ghost layers | High-order |
| Riemann Solver | Left/Right only | None (post-recon) | Every face |
| Pressure Laplacian | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | Every pressure iter |
| IGR Velocity Grad | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | If IGR enabled |
| IGR Entropic Solve | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | If IGR enabled |
| Pressure Advection | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | Semi-implicit only |
| Divergence | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | Semi-implicit only |
| Correction Step | `i+/-1, j+/-1, k+/-1` | 1 neighbor deep | Semi-implicit only |

## Global Reductions

| Operation | Frequency | Type |
|---|---|---|
| CFL maxSpeed | 1x per timestep | `MPI_Allreduce` `MPI_MAX` |
| CFL minDx | 1x per timestep | `MPI_Allreduce` `MPI_MIN` |
| Pressure residual | 1x per Jacobi/GS iteration | `MPI_Allreduce` `MPI_MAX` |
| Diagnostics (max sigma, etc.) | At output intervals | `MPI_Allreduce` `MPI_MAX` |

## Key Gotchas

1. **Onion-peel ordering matters in multi-D.** Exchange x-halos first, then y, then z — exactly like the existing `applyBoundaryConditions`. If you exchange all directions simultaneously, corner/edge ghosts will be wrong.

2. **`fillScalarGhosts` is called inside the pressure solver loop.** It needs the same MPI-aware treatment as `applyBoundaryConditions`. Every iteration = 1 halo exchange + 1 `Allreduce`.

3. **Gauss-Seidel won't scale.** Use Jacobi for MPI. Consider CG or multigrid if iteration counts become a bottleneck.

4. **`SolutionState::allocate` uses `mesh.totalCells()`** which is already local — no change needed, as long as each rank constructs its own local `RectilinearMesh`.

5. **Reconstruction workspace** (`xLeft_`, `xRight_`, etc. in `ExplicitSolver`) is sized from local mesh dimensions — automatically correct.

6. **`MPI_PROC_NULL` is your friend.** Send/recv to `MPI_PROC_NULL` is a no-op, so you don't need `if (neighbor != MPI_PROC_NULL)` guards around communication calls.

7. **Pack multiple fields into one buffer per face** to minimize latency. A `SolutionState` has ~11 active fields; packing them into one send per face direction is 6x fewer MPI calls than exchanging each field separately.
