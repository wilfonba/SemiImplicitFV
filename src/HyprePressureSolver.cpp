#include "HyprePressureSolver.hpp"
#include "HaloExchange.hpp"

#include <HYPRE_struct_ls.h>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace SemiImplicitFV {

// ---------------------------------------------------------------------------
// Constructors / destructor
// ---------------------------------------------------------------------------

HyprePressureSolver::HyprePressureSolver()
    : comm_(MPI_COMM_SELF)
    , ilower_{0, 0, 0}
    , iupper_{0, 0, 0}
    , periodic_{0, 0, 0}
    , useMPI_(false)
{}

HyprePressureSolver::HyprePressureSolver(
    MPI_Comm comm,
    const std::array<int,6>& localExtent,
    const std::array<int,3>& periodic)
    : comm_(comm)
    , ilower_{localExtent[0], localExtent[2], localExtent[4]}
    , iupper_{localExtent[1] - 1, localExtent[3] - 1, localExtent[5] - 1}
    , periodic_(periodic)
    , useMPI_(true)
{}

HyprePressureSolver::~HyprePressureSolver() {
    destroyHypre();
}

// ---------------------------------------------------------------------------
// Hypre lifecycle
// ---------------------------------------------------------------------------

void HyprePressureSolver::setupHypre(int dim, int nLocalCells) {
    dim_ = dim;

    // ---- Grid ----
    HYPRE_StructGridCreate(comm_, dim_, &grid_);

    // Hypre uses dim-dimensional index arrays
    HYPRE_Int lo[3] = {ilower_[0], ilower_[1], ilower_[2]};
    HYPRE_Int hi[3] = {iupper_[0], iupper_[1], iupper_[2]};
    HYPRE_StructGridSetExtents(grid_, lo, hi);

    // Periodic directions: Hypre expects the *global* period size per direction
    // For serial mode, period = local size in that direction; for MPI, the caller
    // must set the global extent. We store 0/1 flags and compute global sizes.
    if (periodic_[0] || periodic_[1] || periodic_[2]) {
        HYPRE_Int period[3] = {0, 0, 0};
        if (periodic_[0]) {
            // For periodic: need global extent. With FetchContent MPI, the
            // global size is gathered. For serial, iupper - ilower + 1 is the full grid.
            // We use MPI_Allreduce to get the global size.
            int localN = iupper_[0] - ilower_[0] + 1;
            int globalN = 0;
            MPI_Allreduce(&localN, &globalN, 1, MPI_INT, MPI_SUM, comm_);
            period[0] = globalN;
        }
        if (periodic_[1] && dim_ >= 2) {
            int localN = iupper_[1] - ilower_[1] + 1;
            int globalN = 0;
            MPI_Allreduce(&localN, &globalN, 1, MPI_INT, MPI_SUM, comm_);
            period[1] = globalN;
        }
        if (periodic_[2] && dim_ >= 3) {
            int localN = iupper_[2] - ilower_[2] + 1;
            int globalN = 0;
            MPI_Allreduce(&localN, &globalN, 1, MPI_INT, MPI_SUM, comm_);
            period[2] = globalN;
        }
        HYPRE_StructGridSetPeriodic(grid_, period);
    }

    HYPRE_StructGridAssemble(grid_);

    // ---- Stencil ----
    int stencilSize = 1 + 2 * dim_;  // center + 2 per dimension
    HYPRE_StructStencilCreate(dim_, stencilSize, &stencil_);

    // Center entry first
    {
        HYPRE_Int offset[3] = {0, 0, 0};
        HYPRE_StructStencilSetElement(stencil_, 0, offset);
    }
    // X: left, right
    {
        HYPRE_Int offL[3] = {-1, 0, 0};
        HYPRE_Int offR[3] = { 1, 0, 0};
        HYPRE_StructStencilSetElement(stencil_, 1, offL);
        HYPRE_StructStencilSetElement(stencil_, 2, offR);
    }
    if (dim_ >= 2) {
        HYPRE_Int offL[3] = {0, -1, 0};
        HYPRE_Int offR[3] = {0,  1, 0};
        HYPRE_StructStencilSetElement(stencil_, 3, offL);
        HYPRE_StructStencilSetElement(stencil_, 4, offR);
    }
    if (dim_ >= 3) {
        HYPRE_Int offL[3] = {0, 0, -1};
        HYPRE_Int offR[3] = {0, 0,  1};
        HYPRE_StructStencilSetElement(stencil_, 5, offL);
        HYPRE_StructStencilSetElement(stencil_, 6, offR);
    }

    // ---- Matrix & Vectors ----
    HYPRE_StructMatrixCreate(comm_, grid_, stencil_, &matrix_);
    HYPRE_StructMatrixInitialize(matrix_);

    HYPRE_StructVectorCreate(comm_, grid_, &bVec_);
    HYPRE_StructVectorInitialize(bVec_);

    HYPRE_StructVectorCreate(comm_, grid_, &xVec_);
    HYPRE_StructVectorInitialize(xVec_);

    // ---- PFMG preconditioner ----
    HYPRE_StructPFMGCreate(comm_, &precond_);
    HYPRE_StructPFMGSetMaxIter(precond_, 1);       // single V-cycle
    HYPRE_StructPFMGSetRelaxType(precond_, 2);      // weighted Jacobi
    HYPRE_StructPFMGSetNumPreRelax(precond_, 1);
    HYPRE_StructPFMGSetNumPostRelax(precond_, 1);
    HYPRE_StructPFMGSetTol(precond_, 0.0);          // exact preconditioning

    // ---- PCG solver ----
    HYPRE_StructPCGCreate(comm_, &solver_);
    HYPRE_StructPCGSetPrecond(solver_,
        HYPRE_StructPFMGSolve,
        HYPRE_StructPFMGSetup,
        precond_);

    // ---- Scratch buffers ----
    matValues_.resize(static_cast<std::size_t>(stencilSize) * nLocalCells);
    rhsValues_.resize(nLocalCells);
    solValues_.resize(nLocalCells);

    initialized_ = true;
}

void HyprePressureSolver::destroyHypre() {
    if (solver_)  { HYPRE_StructPCGDestroy(solver_);   solver_  = nullptr; }
    if (precond_) { HYPRE_StructPFMGDestroy(precond_);  precond_ = nullptr; }
    if (xVec_)    { HYPRE_StructVectorDestroy(xVec_);   xVec_    = nullptr; }
    if (bVec_)    { HYPRE_StructVectorDestroy(bVec_);   bVec_    = nullptr; }
    if (matrix_)  { HYPRE_StructMatrixDestroy(matrix_);  matrix_  = nullptr; }
    if (stencil_) { HYPRE_StructStencilDestroy(stencil_); stencil_ = nullptr; }
    if (grid_)    { HYPRE_StructGridDestroy(grid_);     grid_    = nullptr; }
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Matrix / vector assembly
// ---------------------------------------------------------------------------

void HyprePressureSolver::assembleSystem(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    const std::vector<double>& pressure,
    double dt,
    HaloExchange* halo)
{
    const double dt2 = dt * dt;
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();
    const int stencilSize = 1 + 2 * dim_;

    int cellIdx = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = mesh.index(i, j, k);
                const double coeff = rhoc2[idx] * dt2;

                double diagL = 0.0;
                double* row = &matValues_[static_cast<std::size_t>(cellIdx) * stencilSize];

                // Initialize all off-diagonals to zero
                for (int s = 0; s < stencilSize; ++s) row[s] = 0.0;

                // --- X direction ---
                {
                    const std::size_t xm = mesh.index(i - 1, j, k);
                    const std::size_t xp = mesh.index(i + 1, j, k);
                    const double rhoL = 0.5 * (rho[idx] + rho[xm]);
                    const double rhoR = 0.5 * (rho[idx] + rho[xp]);
                    const double dL = 0.5 * (mesh.dx(i - 1) + mesh.dx(i));
                    const double dR = 0.5 * (mesh.dx(i) + mesh.dx(i + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dx(i));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dx(i));

                    // Neumann BC: zero the coefficient at physical boundaries
                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (i == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::XLow);
                        physHigh = (i == nx - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::XHigh);
                    } else {
                        physLow  = (i == 0)      && mesh.boundaryCondition(RectilinearMesh::XLow)  != BoundaryCondition::Periodic;
                        physHigh = (i == nx - 1)  && mesh.boundaryCondition(RectilinearMesh::XHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    row[1] = -coeff * cL;   // x-left
                    row[2] = -coeff * cR;   // x-right
                    diagL += cL + cR;
                }

                // --- Y direction ---
                if (dim_ >= 2) {
                    const std::size_t ym = mesh.index(i, j - 1, k);
                    const std::size_t yp = mesh.index(i, j + 1, k);
                    const double rhoL = 0.5 * (rho[idx] + rho[ym]);
                    const double rhoR = 0.5 * (rho[idx] + rho[yp]);
                    const double dL = 0.5 * (mesh.dy(j - 1) + mesh.dy(j));
                    const double dR = 0.5 * (mesh.dy(j) + mesh.dy(j + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dy(j));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dy(j));

                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (j == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::YLow);
                        physHigh = (j == ny - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::YHigh);
                    } else {
                        physLow  = (j == 0)      && mesh.boundaryCondition(RectilinearMesh::YLow)  != BoundaryCondition::Periodic;
                        physHigh = (j == ny - 1)  && mesh.boundaryCondition(RectilinearMesh::YHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    row[3] = -coeff * cL;   // y-left
                    row[4] = -coeff * cR;   // y-right
                    diagL += cL + cR;
                }

                // --- Z direction ---
                if (dim_ >= 3) {
                    const std::size_t zm = mesh.index(i, j, k - 1);
                    const std::size_t zp = mesh.index(i, j, k + 1);
                    const double rhoL = 0.5 * (rho[idx] + rho[zm]);
                    const double rhoR = 0.5 * (rho[idx] + rho[zp]);
                    const double dL = 0.5 * (mesh.dz(k - 1) + mesh.dz(k));
                    const double dR = 0.5 * (mesh.dz(k) + mesh.dz(k + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dz(k));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dz(k));

                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (k == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::ZLow);
                        physHigh = (k == nz - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::ZHigh);
                    } else {
                        physLow  = (k == 0)      && mesh.boundaryCondition(RectilinearMesh::ZLow)  != BoundaryCondition::Periodic;
                        physHigh = (k == nz - 1)  && mesh.boundaryCondition(RectilinearMesh::ZHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    row[5] = -coeff * cL;   // z-left
                    row[6] = -coeff * cR;   // z-right
                    diagL += cL + cR;
                }

                // Diagonal: (1 + dt^2 * rhoc2 * sum_of_laplacian_coeffs)
                row[0] = 1.0 + coeff * diagL;

                // RHS and initial guess
                rhsValues_[cellIdx] = rhs[idx];
                solValues_[cellIdx] = pressure[idx];

                ++cellIdx;
            }
        }
    }

    // Set values into Hypre objects
    HYPRE_Int lo[3] = {ilower_[0], ilower_[1], ilower_[2]};
    HYPRE_Int hi[3] = {iupper_[0], iupper_[1], iupper_[2]};
    int nLocalCells = cellIdx;

    // Set matrix entries (all stencil entries at once)
    std::vector<HYPRE_Int> stencilIndices(stencilSize);
    for (int s = 0; s < stencilSize; ++s) stencilIndices[s] = s;

    HYPRE_StructMatrixSetBoxValues(matrix_, lo, hi,
        stencilSize, stencilIndices.data(), matValues_.data());
    HYPRE_StructMatrixAssemble(matrix_);

    HYPRE_StructVectorSetBoxValues(bVec_, lo, hi, rhsValues_.data());
    HYPRE_StructVectorAssemble(bVec_);

    HYPRE_StructVectorSetBoxValues(xVec_, lo, hi, solValues_.data());
    HYPRE_StructVectorAssemble(xVec_);
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

int HyprePressureSolver::solveInternal(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter,
    HaloExchange* halo)
{
    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();
    const int nLocalCells = nx * ny * nz;

    // Lazy one-time initialization
    if (!initialized_) {
        // For serial mode, set extents from the mesh directly
        if (!useMPI_) {
            ilower_ = {0, 0, 0};
            iupper_ = {nx - 1,
                       (mesh.dim() >= 2) ? ny - 1 : 0,
                       (mesh.dim() >= 3) ? nz - 1 : 0};
            // Detect periodic from mesh BCs
            periodic_[0] = (mesh.boundaryCondition(RectilinearMesh::XLow) == BoundaryCondition::Periodic) ? 1 : 0;
            if (mesh.dim() >= 2)
                periodic_[1] = (mesh.boundaryCondition(RectilinearMesh::YLow) == BoundaryCondition::Periodic) ? 1 : 0;
            if (mesh.dim() >= 3)
                periodic_[2] = (mesh.boundaryCondition(RectilinearMesh::ZLow) == BoundaryCondition::Periodic) ? 1 : 0;
        }
        setupHypre(mesh.dim(), nLocalCells);
    }

    // Fill ghost cells for rho access in stencil computation
    // (rho is const, but we need valid ghosts - the caller should have filled them)

    // Assemble the linear system
    assembleSystem(mesh, rho, rhoc2, rhs, pressure, dt, halo);

    // Configure solver tolerances
    HYPRE_StructPCGSetTol(solver_, tolerance);
    HYPRE_StructPCGSetMaxIter(solver_, maxIter);
    HYPRE_StructPCGSetTwoNorm(solver_, 1);
    HYPRE_StructPCGSetRelChange(solver_, 0);
    HYPRE_StructPCGSetLogging(solver_, 0);

    // Setup and solve
    HYPRE_StructPCGSetup(solver_, matrix_, bVec_, xVec_);
    HYPRE_StructPCGSolve(solver_, matrix_, bVec_, xVec_);

    // Extract solution
    HYPRE_Int lo[3] = {ilower_[0], ilower_[1], ilower_[2]};
    HYPRE_Int hi[3] = {iupper_[0], iupper_[1], iupper_[2]};
    HYPRE_StructVectorGetBoxValues(xVec_, lo, hi, solValues_.data());

    // Copy back to pressure array (interior cells, x-fastest order)
    int cellIdx = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                pressure[mesh.index(i, j, k)] = solValues_[cellIdx];
                ++cellIdx;
            }
        }
    }

    // Return iteration count
    HYPRE_Int numIter = 0;
    HYPRE_StructPCGGetNumIterations(solver_, &numIter);
    return static_cast<int>(numIter);
}

int HyprePressureSolver::solve(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter)
{
    return solveInternal(mesh, rho, rhoc2, rhs, pressure, dt, tolerance, maxIter, nullptr);
}

int HyprePressureSolver::solve(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter,
    HaloExchange& halo)
{
    return solveInternal(mesh, rho, rhoc2, rhs, pressure, dt, tolerance, maxIter, &halo);
}

} // namespace SemiImplicitFV
