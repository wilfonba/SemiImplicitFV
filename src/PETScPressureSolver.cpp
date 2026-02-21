#include "PETScPressureSolver.hpp"
#include "HaloExchange.hpp"

#include <petscksp.h>
#include <petscdmda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace SemiImplicitFV {

// ---------------------------------------------------------------------------
// Constructors / destructor
// ---------------------------------------------------------------------------

PETScPressureSolver::PETScPressureSolver()
    : comm_(MPI_COMM_SELF)
    , ilower_{0, 0, 0}
    , iupper_{0, 0, 0}
    , periodic_{0, 0, 0}
    , procDims_{1, 1, 1}
    , useMPI_(false)
{}

PETScPressureSolver::PETScPressureSolver(
    MPI_Comm comm,
    const std::array<int,6>& localExtent,
    const std::array<int,3>& periodic,
    const std::array<int,3>& procDims)
    : comm_(comm)
    , ilower_{localExtent[0], localExtent[2], localExtent[4]}
    , iupper_{localExtent[1] - 1, localExtent[3] - 1, localExtent[5] - 1}
    , periodic_(periodic)
    , procDims_(procDims)
    , useMPI_(true)
{}

PETScPressureSolver::~PETScPressureSolver() {
    destroyPETSc();
}

// ---------------------------------------------------------------------------
// PETSc lifecycle
// ---------------------------------------------------------------------------

void PETScPressureSolver::setupPETSc(const RectilinearMesh& mesh) {
    dim_ = mesh.dim();

    const int nx = mesh.nx();
    const int ny = mesh.ny();
    const int nz = mesh.nz();

    // Determine boundary types per direction
    DMBoundaryType bx = periodic_[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType by = periodic_[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType bz = periodic_[2] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;

    // Compute global sizes from the exclusive upper bounds of cell extents.
    // Using MPI_MAX is correct regardless of decomposition topology (e.g. 2x2x1).
    int globalNx = nx, globalNy = ny, globalNz = nz;
    if (useMPI_) {
        globalNx = iupper_[0] + 1;
        globalNy = (dim_ >= 2) ? iupper_[1] + 1 : 1;
        globalNz = (dim_ >= 3) ? iupper_[2] + 1 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &globalNx, 1, MPI_INT, MPI_MAX, comm_);
        if (dim_ >= 2) MPI_Allreduce(MPI_IN_PLACE, &globalNy, 1, MPI_INT, MPI_MAX, comm_);
        if (dim_ >= 3) MPI_Allreduce(MPI_IN_PLACE, &globalNz, 1, MPI_INT, MPI_MAX, comm_);
    }

    // --- Remap communicator for DMDA compatibility ---
    // MPI_Cart_create (used by MPIContext) and PETSc's DMDA use DIFFERENT
    // rank-to-coordinate mappings:
    //   MPI:  rank -> (z-fastest) i.e. divides z first, then y, then x
    //   DMDA: rank -> (x-fastest) i.e. petsc_rank = ix + iy*mx + iz*mx*my
    //
    // When the Cartesian communicator has reorder=true (as MPIContext does),
    // rank numbering may differ from both conventions. We must create a new
    // communicator where each rank's number matches DMDA's expected ordering
    // based on that rank's Cartesian coordinates.
    MPI_Comm petscComm = comm_;
    std::vector<PetscInt> lx, ly, lz;

    if (useMPI_) {
        // Get this rank's Cartesian coordinates from the existing communicator
        int coords[3] = {0, 0, 0};
        int myRank = 0;
        MPI_Comm_rank(comm_, &myRank);
        MPI_Cart_coords(comm_, myRank, 3, coords);

        // Compute the rank this process should have in DMDA's x-fastest ordering
        int mx = procDims_[0], my = procDims_[1];
        int petscRank = coords[0] + coords[1] * mx + coords[2] * mx * my;

        // Create a new communicator with rank ordering matching DMDA
        MPI_Comm_split(comm_, 0, petscRank, &petscComm_);
        petscComm = petscComm_;

        // Gather local sizes in DMDA rank order to build lx/ly/lz arrays.
        // In the remapped communicator, ranks are ordered x-fastest, so
        // rank 0..mx-1 are the x-row of the first y,z slab, etc.
        int nProcs = 0;
        MPI_Comm_size(petscComm, &nProcs);

        struct LocalSizes { int nx, ny, nz; };
        LocalSizes mySizes = {nx, ny, nz};
        std::vector<LocalSizes> allSizes(nProcs);
        MPI_Allgather(&mySizes, sizeof(LocalSizes), MPI_BYTE,
                      allSizes.data(), sizeof(LocalSizes), MPI_BYTE, petscComm);

        // Extract per-processor sizes along each dimension.
        // In x-fastest ordering, ranks 0..mx-1 give the x-dimension sizes.
        lx.resize(procDims_[0]);
        for (int i = 0; i < procDims_[0]; ++i)
            lx[i] = allSizes[i].nx;

        if (dim_ >= 2) {
            ly.resize(procDims_[1]);
            for (int j = 0; j < procDims_[1]; ++j)
                ly[j] = allSizes[j * mx].ny;
        }

        if (dim_ >= 3) {
            lz.resize(procDims_[2]);
            for (int k = 0; k < procDims_[2]; ++k)
                lz[k] = allSizes[k * mx * my].nz;
        }
    }

    // Create DMDA on the remapped communicator with explicit local sizes.
    if (dim_ == 1) {
        DMDACreate1d(petscComm, bx, globalNx, 1, 1,
                     lx.empty() ? nullptr : lx.data(), &dm_);
    } else if (dim_ == 2) {
        DMDACreate2d(petscComm, bx, by,
                     DMDA_STENCIL_STAR,
                     globalNx, globalNy,
                     procDims_[0], procDims_[1],
                     1, 1,
                     lx.data(), ly.data(),
                     &dm_);
    } else {
        DMDACreate3d(petscComm, bx, by, bz,
                     DMDA_STENCIL_STAR,
                     globalNx, globalNy, globalNz,
                     procDims_[0], procDims_[1], procDims_[2],
                     1, 1,
                     lx.data(), ly.data(), lz.data(),
                     &dm_);
    }

    DMSetUp(dm_);

    // Validate that DMDA partition matches the local mesh
    {
        PetscInt xs, ys, zs, xm, ym, zm;
        if (dim_ == 1) {
            DMDAGetCorners(dm_, &xs, nullptr, nullptr, &xm, nullptr, nullptr);
            ys = 0; zs = 0; ym = 1; zm = 1;
        } else if (dim_ == 2) {
            DMDAGetCorners(dm_, &xs, &ys, nullptr, &xm, &ym, nullptr);
            zs = 0; zm = 1;
        } else {
            DMDAGetCorners(dm_, &xs, &ys, &zs, &xm, &ym, &zm);
        }

        int myRankPetsc = 0;
        MPI_Comm_rank(petscComm, &myRankPetsc);

        bool mismatch = false;
        if (static_cast<int>(xm) != nx) mismatch = true;
        if (dim_ >= 2 && static_cast<int>(ym) != ny) mismatch = true;
        if (dim_ >= 3 && static_cast<int>(zm) != nz) mismatch = true;
        if (static_cast<int>(xs) != ilower_[0]) mismatch = true;
        if (dim_ >= 2 && static_cast<int>(ys) != ilower_[1]) mismatch = true;
        if (dim_ >= 3 && static_cast<int>(zs) != ilower_[2]) mismatch = true;

        if (mismatch) {
            int myRankOrig = 0;
            MPI_Comm_rank(comm_, &myRankOrig);
            std::fprintf(stderr,
                "PETSc DMDA PARTITION MISMATCH on orig rank %d (petsc rank %d)!\n"
                "  DMDA corners: xs=%d ys=%d zs=%d xm=%d ym=%d zm=%d\n"
                "  Expected:     xs=%d ys=%d zs=%d xm=%d ym=%d zm=%d\n",
                myRankOrig, myRankPetsc,
                (int)xs, (int)ys, (int)zs, (int)xm, (int)ym, (int)zm,
                ilower_[0], ilower_[1], ilower_[2], nx, ny, nz);
            MPI_Abort(petscComm, 1);
        }
    }

    // Create matrix and vectors from DM
    DMCreateMatrix(dm_, &A_);
    DMCreateGlobalVector(dm_, &b_);
    DMCreateGlobalVector(dm_, &x_);

    // Create KSP solver on the same communicator as DMDA
    KSPCreate(petscComm, &ksp_);
    KSPSetDM(ksp_, dm_);
    KSPSetDMActive(ksp_, PETSC_FALSE);  // We assemble the matrix ourselves
    KSPSetType(ksp_, KSPCG);

    // Set GAMG preconditioner
    PC pc;
    KSPGetPC(ksp_, &pc);
    PCSetType(pc, PCGAMG);

    // Use the UNPRECONDITIONED (true) residual norm for convergence testing.
    // With GAMG, the preconditioned residual drops ~1000x in 1 iteration,
    // causing premature convergence while the actual residual is still large.
    KSPSetNormType(ksp_, KSP_NORM_UNPRECONDITIONED);

#ifndef NDEBUG
    // Enable residual monitoring: print true residual norm at each KSP iteration
    PetscOptionsInsertString(nullptr, "-ksp_monitor");
#endif

    // Allow runtime override via command-line options (-ksp_type, -pc_type, etc.)
    KSPSetFromOptions(ksp_);

    // Set tolerances and options that persist across solves
    KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE);

    initialized_ = true;
}

void PETScPressureSolver::destroyPETSc() {
    if (ksp_) { KSPDestroy(&ksp_); ksp_ = nullptr; }
    if (x_)   { VecDestroy(&x_);   x_   = nullptr; }
    if (b_)   { VecDestroy(&b_);   b_   = nullptr; }
    if (A_)   { MatDestroy(&A_);   A_   = nullptr; }
    if (dm_)  { DMDestroy(&dm_);   dm_  = nullptr; }
    if (petscComm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&petscComm_);
        petscComm_ = MPI_COMM_NULL;
    }
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Matrix / vector assembly
// ---------------------------------------------------------------------------

void PETScPressureSolver::assembleSystem(
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

    // Get local portion of the DMDA
    PetscInt xs, ys, zs, xm, ym, zm;
    if (dim_ == 1) {
        DMDAGetCorners(dm_, &xs, nullptr, nullptr, &xm, nullptr, nullptr);
        ys = 0; zs = 0; ym = 1; zm = 1;
    } else if (dim_ == 2) {
        DMDAGetCorners(dm_, &xs, &ys, nullptr, &xm, &ym, nullptr);
        zs = 0; zm = 1;
    } else {
        DMDAGetCorners(dm_, &xs, &ys, &zs, &xm, &ym, &zm);
    }

    for (PetscInt k = zs; k < zs + zm; ++k) {
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                // Map DMDA (i,j,k) to local mesh indices
                const int li = static_cast<int>(i - xs);
                const int lj = static_cast<int>(j - ys);
                const int lk = static_cast<int>(k - zs);
                const std::size_t idx = mesh.index(li, lj, lk);
                const double coeff = rhoc2[idx] * dt2;

                double diagL = 0.0;
                MatStencil row;
                row.i = i; row.j = j; row.k = k;

                // Maximum 7-point stencil (center + 2 per dimension)
                MatStencil cols[7];
                double vals[7];
                int nEntries = 0;

                // --- X direction ---
                {
                    const std::size_t xm_idx = mesh.index(li - 1, lj, lk);
                    const std::size_t xp_idx = mesh.index(li + 1, lj, lk);
                    const double rhoL = 0.5 * (rho[idx] + rho[xm_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[xp_idx]);
                    const double dL = 0.5 * (mesh.dx(li - 1) + mesh.dx(li));
                    const double dR = 0.5 * (mesh.dx(li) + mesh.dx(li + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dx(li));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dx(li));

                    // Neumann BC: zero the coefficient at physical boundaries
                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (li == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::XLow);
                        physHigh = (li == nx - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::XHigh);
                    } else {
                        physLow  = (li == 0)      && mesh.boundaryCondition(RectilinearMesh::XLow)  != BoundaryCondition::Periodic;
                        physHigh = (li == nx - 1)  && mesh.boundaryCondition(RectilinearMesh::XHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    if (cL != 0.0) {
                        cols[nEntries].i = i - 1; cols[nEntries].j = j; cols[nEntries].k = k;
                        vals[nEntries] = -coeff * cL;
                        ++nEntries;
                    }
                    if (cR != 0.0) {
                        cols[nEntries].i = i + 1; cols[nEntries].j = j; cols[nEntries].k = k;
                        vals[nEntries] = -coeff * cR;
                        ++nEntries;
                    }
                    diagL += cL + cR;
                }

                // --- Y direction ---
                if (dim_ >= 2) {
                    const std::size_t ym_idx = mesh.index(li, lj - 1, lk);
                    const std::size_t yp_idx = mesh.index(li, lj + 1, lk);
                    const double rhoL = 0.5 * (rho[idx] + rho[ym_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[yp_idx]);
                    const double dL = 0.5 * (mesh.dy(lj - 1) + mesh.dy(lj));
                    const double dR = 0.5 * (mesh.dy(lj) + mesh.dy(lj + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dy(lj));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dy(lj));

                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (lj == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::YLow);
                        physHigh = (lj == ny - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::YHigh);
                    } else {
                        physLow  = (lj == 0)      && mesh.boundaryCondition(RectilinearMesh::YLow)  != BoundaryCondition::Periodic;
                        physHigh = (lj == ny - 1)  && mesh.boundaryCondition(RectilinearMesh::YHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    if (cL != 0.0) {
                        cols[nEntries].i = i; cols[nEntries].j = j - 1; cols[nEntries].k = k;
                        vals[nEntries] = -coeff * cL;
                        ++nEntries;
                    }
                    if (cR != 0.0) {
                        cols[nEntries].i = i; cols[nEntries].j = j + 1; cols[nEntries].k = k;
                        vals[nEntries] = -coeff * cR;
                        ++nEntries;
                    }
                    diagL += cL + cR;
                }

                // --- Z direction ---
                if (dim_ >= 3) {
                    const std::size_t zm_idx = mesh.index(li, lj, lk - 1);
                    const std::size_t zp_idx = mesh.index(li, lj, lk + 1);
                    const double rhoL = 0.5 * (rho[idx] + rho[zm_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[zp_idx]);
                    const double dL = 0.5 * (mesh.dz(lk - 1) + mesh.dz(lk));
                    const double dR = 0.5 * (mesh.dz(lk) + mesh.dz(lk + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dz(lk));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dz(lk));

                    bool physLow = false, physHigh = false;
                    if (halo) {
                        physLow  = (lk == 0)      && halo->mpi().isPhysicalBoundary(MPIContext::ZLow);
                        physHigh = (lk == nz - 1)  && halo->mpi().isPhysicalBoundary(MPIContext::ZHigh);
                    } else {
                        physLow  = (lk == 0)      && mesh.boundaryCondition(RectilinearMesh::ZLow)  != BoundaryCondition::Periodic;
                        physHigh = (lk == nz - 1)  && mesh.boundaryCondition(RectilinearMesh::ZHigh) != BoundaryCondition::Periodic;
                    }

                    if (physLow)  cL = 0.0;
                    if (physHigh) cR = 0.0;

                    if (cL != 0.0) {
                        cols[nEntries].i = i; cols[nEntries].j = j; cols[nEntries].k = k - 1;
                        vals[nEntries] = -coeff * cL;
                        ++nEntries;
                    }
                    if (cR != 0.0) {
                        cols[nEntries].i = i; cols[nEntries].j = j; cols[nEntries].k = k + 1;
                        vals[nEntries] = -coeff * cR;
                        ++nEntries;
                    }
                    diagL += cL + cR;
                }

                // Diagonal: (1 + dt^2 * rhoc2 * sum_of_laplacian_coeffs)
                cols[nEntries].i = i; cols[nEntries].j = j; cols[nEntries].k = k;
                vals[nEntries] = 1.0 + coeff * diagL;
                ++nEntries;

                MatSetValuesStencil(A_, 1, &row, nEntries, cols, vals, INSERT_VALUES);

                // RHS and initial guess are set via DMDAVecGetArray after this loop
            }
        }
    }

    MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);

    // Set RHS and initial guess via DMDAVecGetArray
    if (dim_ == 1) {
        PetscScalar *bArr, *xArr;
        DMDAVecGetArray(dm_, b_, &bArr);
        DMDAVecGetArray(dm_, x_, &xArr);
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const int li = static_cast<int>(i - xs);
            const std::size_t idx = mesh.index(li, 0, 0);
            bArr[i] = rhs[idx];
            xArr[i] = pressure[idx];
        }
        DMDAVecRestoreArray(dm_, b_, &bArr);
        DMDAVecRestoreArray(dm_, x_, &xArr);
    } else if (dim_ == 2) {
        PetscScalar **bArr, **xArr;
        DMDAVecGetArray(dm_, b_, &bArr);
        DMDAVecGetArray(dm_, x_, &xArr);
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                const int li = static_cast<int>(i - xs);
                const int lj = static_cast<int>(j - ys);
                const std::size_t idx = mesh.index(li, lj, 0);
                bArr[j][i] = rhs[idx];
                xArr[j][i] = pressure[idx];
            }
        }
        DMDAVecRestoreArray(dm_, b_, &bArr);
        DMDAVecRestoreArray(dm_, x_, &xArr);
    } else {
        PetscScalar ***bArr, ***xArr;
        DMDAVecGetArray(dm_, b_, &bArr);
        DMDAVecGetArray(dm_, x_, &xArr);
        for (PetscInt k = zs; k < zs + zm; ++k) {
            for (PetscInt j = ys; j < ys + ym; ++j) {
                for (PetscInt i = xs; i < xs + xm; ++i) {
                    const int li = static_cast<int>(i - xs);
                    const int lj = static_cast<int>(j - ys);
                    const int lk = static_cast<int>(k - zs);
                    const std::size_t idx = mesh.index(li, lj, lk);
                    bArr[k][j][i] = rhs[idx];
                    xArr[k][j][i] = pressure[idx];
                }
            }
        }
        DMDAVecRestoreArray(dm_, b_, &bArr);
        DMDAVecRestoreArray(dm_, x_, &xArr);
    }
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

int PETScPressureSolver::solveInternal(
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

    // Lazy one-time initialization
    if (!initialized_) {
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
        setupPETSc(mesh);
    }

    // Assemble the linear system
    assembleSystem(mesh, rho, rhoc2, rhs, pressure, dt, halo);

    // Tell PETSc about the (re-)assembled matrix every step.
    // The sparsity pattern never changes, so PETSc can reuse symbolic setup,
    // but GAMG must see updated coefficients to remain valid as dt changes.
    KSPSetOperators(ksp_, A_, A_);

    // Use tolerance as ABSOLUTE (matching Hypre PCG behavior with two-norm).
    // Hypre's PCG with SetTwoNorm(1) converges when ||r|| < tol (absolute).
    // PETSc's rtol is relative to initial residual, so we disable it and use atol.
    KSPSetTolerances(ksp_, 1e-50, tolerance, PETSC_DEFAULT, maxIter);

#ifndef NDEBUG
    // Debug: print what PETSc actually has for tolerances
    {
        PetscReal rtol_out, atol_out, dtol_out;
        PetscInt maxits_out;
        KSPGetTolerances(ksp_, &rtol_out, &atol_out, &dtol_out, &maxits_out);
        int rank = 0;
        MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            PetscPrintf(comm_, "  PETSc KSP: rtol=%.2e atol=%.2e dtol=%.2e maxits=%d\n",
                        rtol_out, atol_out, dtol_out, (int)maxits_out);
        }
    }
#endif

    // Solve
    KSPSolve(ksp_, b_, x_);

#ifndef NDEBUG
    // Check convergence reason
    {
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp_, &reason);
        PetscInt numIter;
        KSPGetIterationNumber(ksp_, &numIter);
        PetscReal rnorm;
        KSPGetResidualNorm(ksp_, &rnorm);
        int rank = 0;
        if (useMPI_) MPI_Comm_rank(comm_, &rank);
        if (rank == 0) {
            std::fprintf(stderr, "  KSP: %d its, rnorm=%.3e, reason=%d\n",
                         (int)numIter, (double)rnorm, (int)reason);
        }
    }
#endif

    // Extract solution back to pressure array
    PetscInt xs, ys, zs, xm, ym, zm;
    if (dim_ == 1) {
        DMDAGetCorners(dm_, &xs, nullptr, nullptr, &xm, nullptr, nullptr);
        ys = 0; zs = 0; ym = 1; zm = 1;
    } else if (dim_ == 2) {
        DMDAGetCorners(dm_, &xs, &ys, nullptr, &xm, &ym, nullptr);
        zs = 0; zm = 1;
    } else {
        DMDAGetCorners(dm_, &xs, &ys, &zs, &xm, &ym, &zm);
    }

    if (dim_ == 1) {
        const PetscScalar *xArr;
        DMDAVecGetArrayRead(dm_, x_, &xArr);
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const int li = static_cast<int>(i - xs);
            pressure[mesh.index(li, 0, 0)] = xArr[i];
        }
        DMDAVecRestoreArrayRead(dm_, x_, &xArr);
    } else if (dim_ == 2) {
        const PetscScalar *const *xArr;
        DMDAVecGetArrayRead(dm_, x_, &xArr);
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                const int li = static_cast<int>(i - xs);
                const int lj = static_cast<int>(j - ys);
                pressure[mesh.index(li, lj, 0)] = xArr[j][i];
            }
        }
        DMDAVecRestoreArrayRead(dm_, x_, &xArr);
    } else {
        const PetscScalar *const *const *xArr;
        DMDAVecGetArrayRead(dm_, x_, &xArr);
        for (PetscInt k = zs; k < zs + zm; ++k) {
            for (PetscInt j = ys; j < ys + ym; ++j) {
                for (PetscInt i = xs; i < xs + xm; ++i) {
                    const int li = static_cast<int>(i - xs);
                    const int lj = static_cast<int>(j - ys);
                    const int lk = static_cast<int>(k - zs);
                    pressure[mesh.index(li, lj, lk)] = xArr[k][j][i];
                }
            }
        }
        DMDAVecRestoreArrayRead(dm_, x_, &xArr);
    }

    // Return iteration count
    PetscInt numIter;
    KSPGetIterationNumber(ksp_, &numIter);
    return static_cast<int>(numIter);
}

int PETScPressureSolver::solve(
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

int PETScPressureSolver::solve(
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
