#include "PETScPressureSolver.hpp"
#include "RectilinearMesh.hpp"
#include "HaloExchange.hpp"

#ifdef SIFV_HAS_PETSC

#include <petscksp.h>
#include <petscdmda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                   */
/* ------------------------------------------------------------------ */

static void setupPETSc(struct PETScContext* ctx, const RectilinearMesh* mesh) {
    ctx->dim = mesh->dim;

    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;

    DMBoundaryType bx = ctx->periodic[0] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType by = ctx->periodic[1] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    DMBoundaryType bz = ctx->periodic[2] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;

    int globalNx = nx, globalNy = ny, globalNz = nz;
    MPI_Comm comm = *(MPI_Comm*)ctx->comm;

    if (ctx->useMPI) {
        globalNx = ctx->iupper[0] + 1;
        globalNy = (ctx->dim >= 2) ? ctx->iupper[1] + 1 : 1;
        globalNz = (ctx->dim >= 3) ? ctx->iupper[2] + 1 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &globalNx, 1, MPI_INT, MPI_MAX, comm);
        if (ctx->dim >= 2) MPI_Allreduce(MPI_IN_PLACE, &globalNy, 1, MPI_INT, MPI_MAX, comm);
        if (ctx->dim >= 3) MPI_Allreduce(MPI_IN_PLACE, &globalNz, 1, MPI_INT, MPI_MAX, comm);
    }

    MPI_Comm petscComm = comm;
    std::vector<PetscInt> lx, ly, lz;

    if (ctx->useMPI) {
        int coords[3] = {0, 0, 0};
        int myRank = 0;
        MPI_Comm_rank(comm, &myRank);
        MPI_Cart_coords(comm, myRank, 3, coords);

        int mx = ctx->procDims[0], my = ctx->procDims[1];
        int petscRank = coords[0] + coords[1] * mx + coords[2] * mx * my;

        MPI_Comm newComm;
        MPI_Comm_split(comm, 0, petscRank, &newComm);
        ctx->petscComm = malloc(sizeof(MPI_Comm));
        *(MPI_Comm*)ctx->petscComm = newComm;
        petscComm = newComm;

        int nProcs = 0;
        MPI_Comm_size(petscComm, &nProcs);

        struct { int nx, ny, nz; } mySizes = {nx, ny, nz};
        std::vector<decltype(mySizes)> allSizes(nProcs);
        MPI_Allgather(&mySizes, sizeof(mySizes), MPI_BYTE,
                      allSizes.data(), sizeof(mySizes), MPI_BYTE, petscComm);

        lx.resize(ctx->procDims[0]);
        for (int i = 0; i < ctx->procDims[0]; ++i)
            lx[i] = allSizes[i].nx;

        if (ctx->dim >= 2) {
            ly.resize(ctx->procDims[1]);
            for (int j = 0; j < ctx->procDims[1]; ++j)
                ly[j] = allSizes[j * mx].ny;
        }

        if (ctx->dim >= 3) {
            lz.resize(ctx->procDims[2]);
            for (int k = 0; k < ctx->procDims[2]; ++k)
                lz[k] = allSizes[k * mx * my].nz;
        }
    }

    DM dm = NULL;
    if (ctx->dim == 1) {
        DMDACreate1d(petscComm, bx, globalNx, 1, 1,
                     lx.empty() ? NULL : lx.data(), &dm);
    } else if (ctx->dim == 2) {
        DMDACreate2d(petscComm, bx, by,
                     DMDA_STENCIL_STAR,
                     globalNx, globalNy,
                     ctx->procDims[0], ctx->procDims[1],
                     1, 1,
                     lx.data(), ly.data(),
                     &dm);
    } else {
        DMDACreate3d(petscComm, bx, by, bz,
                     DMDA_STENCIL_STAR,
                     globalNx, globalNy, globalNz,
                     ctx->procDims[0], ctx->procDims[1], ctx->procDims[2],
                     1, 1,
                     lx.data(), ly.data(), lz.data(),
                     &dm);
    }
    ctx->dm = dm;

    DMSetUp(dm);

    /* Validate partition */
    {
        PetscInt xs, ys, zs, xm, ym, zm;
        if (ctx->dim == 1) {
            DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL);
            ys = 0; zs = 0; ym = 1; zm = 1;
        } else if (ctx->dim == 2) {
            DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);
            zs = 0; zm = 1;
        } else {
            DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);
        }

        int myRankPetsc = 0;
        MPI_Comm_rank(petscComm, &myRankPetsc);

        int mismatch = 0;
        if ((int)xm != nx) mismatch = 1;
        if (ctx->dim >= 2 && (int)ym != ny) mismatch = 1;
        if (ctx->dim >= 3 && (int)zm != nz) mismatch = 1;
        if ((int)xs != ctx->ilower[0]) mismatch = 1;
        if (ctx->dim >= 2 && (int)ys != ctx->ilower[1]) mismatch = 1;
        if (ctx->dim >= 3 && (int)zs != ctx->ilower[2]) mismatch = 1;

        if (mismatch) {
            int myRankOrig = 0;
            MPI_Comm_rank(comm, &myRankOrig);
            std::fprintf(stderr,
                "PETSc DMDA PARTITION MISMATCH on orig rank %d (petsc rank %d)!\n"
                "  DMDA corners: xs=%d ys=%d zs=%d xm=%d ym=%d zm=%d\n"
                "  Expected:     xs=%d ys=%d zs=%d xm=%d ym=%d zm=%d\n",
                myRankOrig, myRankPetsc,
                (int)xs, (int)ys, (int)zs, (int)xm, (int)ym, (int)zm,
                ctx->ilower[0], ctx->ilower[1], ctx->ilower[2], nx, ny, nz);
            MPI_Abort(petscComm, 1);
        }
    }

    Mat A = NULL;
    Vec b = NULL, x = NULL;
    DMCreateMatrix(dm, &A);
    DMCreateGlobalVector(dm, &b);
    DMCreateGlobalVector(dm, &x);
    ctx->A = A;
    ctx->b = b;
    ctx->x = x;

    KSP ksp = NULL;
    KSPCreate(petscComm, &ksp);
    KSPSetDM(ksp, dm);
    KSPSetDMActive(ksp, PETSC_FALSE);
    KSPSetType(ksp, KSPCG);

    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);

    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);

#ifndef NDEBUG
    PetscOptionsInsertString(NULL, "-ksp_monitor");
#endif

    KSPSetFromOptions(ksp);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    ctx->ksp = ksp;

    ctx->initialized = 1;
}

static void assembleSystem(
    struct PETScContext* ctx,
    const RectilinearMesh* mesh,
    const double* rho,
    const double* rhoc2,
    const double* rhs,
    const double* pressure,
    double dt,
    HaloExchange* halo)
{
    const double dt2 = dt * dt;
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;

    DM dm = (DM)ctx->dm;
    Mat A = (Mat)ctx->A;
    Vec b = (Vec)ctx->b;
    Vec x = (Vec)ctx->x;

    PetscInt xs, ys, zs, xm, ym, zm;
    if (ctx->dim == 1) {
        DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL);
        ys = 0; zs = 0; ym = 1; zm = 1;
    } else if (ctx->dim == 2) {
        DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);
        zs = 0; zm = 1;
    } else {
        DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);
    }

    for (PetscInt k = zs; k < zs + zm; ++k) {
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                const int li = (int)(i - xs);
                const int lj = (int)(j - ys);
                const int lk = (int)(k - zs);
                const size_t idx = mesh_index(mesh, li, lj, lk);
                const double coeff = rhoc2[idx] * dt2;

                double diagL = 0.0;
                MatStencil row;
                row.i = i; row.j = j; row.k = k;

                MatStencil cols[7];
                double vals[7];
                int nEntries = 0;

                /* --- X direction --- */
                {
                    const size_t xm_idx = mesh_index(mesh, li - 1, lj, lk);
                    const size_t xp_idx = mesh_index(mesh, li + 1, lj, lk);
                    const double rhoL = 0.5 * (rho[idx] + rho[xm_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[xp_idx]);
                    const double dL = 0.5 * (mesh_dx(mesh, li - 1) + mesh_dx(mesh, li));
                    const double dR = 0.5 * (mesh_dx(mesh, li) + mesh_dx(mesh, li + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dx(mesh, li));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dx(mesh, li));

                    int physLow = 0, physHigh = 0;
                    if (halo) {
                        physLow  = (li == 0)       && mpi_is_physical_boundary(halo->mpi, XLOW);
                        physHigh = (li == nx - 1)   && mpi_is_physical_boundary(halo->mpi, XHIGH);
                    } else {
                        physLow  = (li == 0)       && mesh->bc[XLOW]  != BC_PERIODIC;
                        physHigh = (li == nx - 1)   && mesh->bc[XHIGH] != BC_PERIODIC;
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

                /* --- Y direction --- */
                if (ctx->dim >= 2) {
                    const size_t ym_idx = mesh_index(mesh, li, lj - 1, lk);
                    const size_t yp_idx = mesh_index(mesh, li, lj + 1, lk);
                    const double rhoL = 0.5 * (rho[idx] + rho[ym_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[yp_idx]);
                    const double dL = 0.5 * (mesh_dy(mesh, lj - 1) + mesh_dy(mesh, lj));
                    const double dR = 0.5 * (mesh_dy(mesh, lj) + mesh_dy(mesh, lj + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dy(mesh, lj));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dy(mesh, lj));

                    int physLow = 0, physHigh = 0;
                    if (halo) {
                        physLow  = (lj == 0)       && mpi_is_physical_boundary(halo->mpi, YLOW);
                        physHigh = (lj == ny - 1)   && mpi_is_physical_boundary(halo->mpi, YHIGH);
                    } else {
                        physLow  = (lj == 0)       && mesh->bc[YLOW]  != BC_PERIODIC;
                        physHigh = (lj == ny - 1)   && mesh->bc[YHIGH] != BC_PERIODIC;
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

                /* --- Z direction --- */
                if (ctx->dim >= 3) {
                    const size_t zm_idx = mesh_index(mesh, li, lj, lk - 1);
                    const size_t zp_idx = mesh_index(mesh, li, lj, lk + 1);
                    const double rhoL = 0.5 * (rho[idx] + rho[zm_idx]);
                    const double rhoR = 0.5 * (rho[idx] + rho[zp_idx]);
                    const double dL = 0.5 * (mesh_dz(mesh, lk - 1) + mesh_dz(mesh, lk));
                    const double dR = 0.5 * (mesh_dz(mesh, lk) + mesh_dz(mesh, lk + 1));
                    double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dz(mesh, lk));
                    double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dz(mesh, lk));

                    int physLow = 0, physHigh = 0;
                    if (halo) {
                        physLow  = (lk == 0)       && mpi_is_physical_boundary(halo->mpi, ZLOW);
                        physHigh = (lk == nz - 1)   && mpi_is_physical_boundary(halo->mpi, ZHIGH);
                    } else {
                        physLow  = (lk == 0)       && mesh->bc[ZLOW]  != BC_PERIODIC;
                        physHigh = (lk == nz - 1)   && mesh->bc[ZHIGH] != BC_PERIODIC;
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

                cols[nEntries].i = i; cols[nEntries].j = j; cols[nEntries].k = k;
                vals[nEntries] = 1.0 + coeff * diagL;
                ++nEntries;

                MatSetValuesStencil(A, 1, &row, nEntries, cols, vals, INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    /* Set RHS and initial guess */
    if (ctx->dim == 1) {
        PetscScalar *bArr, *xArr;
        DMDAVecGetArray(dm, b, &bArr);
        DMDAVecGetArray(dm, x, &xArr);
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const int li = (int)(i - xs);
            const size_t idx = mesh_index(mesh, li, 0, 0);
            bArr[i] = rhs[idx];
            xArr[i] = pressure[idx];
        }
        DMDAVecRestoreArray(dm, b, &bArr);
        DMDAVecRestoreArray(dm, x, &xArr);
    } else if (ctx->dim == 2) {
        PetscScalar **bArr, **xArr;
        DMDAVecGetArray(dm, b, &bArr);
        DMDAVecGetArray(dm, x, &xArr);
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                const int li = (int)(i - xs);
                const int lj = (int)(j - ys);
                const size_t idx = mesh_index(mesh, li, lj, 0);
                bArr[j][i] = rhs[idx];
                xArr[j][i] = pressure[idx];
            }
        }
        DMDAVecRestoreArray(dm, b, &bArr);
        DMDAVecRestoreArray(dm, x, &xArr);
    } else {
        PetscScalar ***bArr, ***xArr;
        DMDAVecGetArray(dm, b, &bArr);
        DMDAVecGetArray(dm, x, &xArr);
        for (PetscInt k = zs; k < zs + zm; ++k) {
            for (PetscInt j = ys; j < ys + ym; ++j) {
                for (PetscInt i = xs; i < xs + xm; ++i) {
                    const int li = (int)(i - xs);
                    const int lj = (int)(j - ys);
                    const int lk = (int)(k - zs);
                    const size_t idx = mesh_index(mesh, li, lj, lk);
                    bArr[k][j][i] = rhs[idx];
                    xArr[k][j][i] = pressure[idx];
                }
            }
        }
        DMDAVecRestoreArray(dm, b, &bArr);
        DMDAVecRestoreArray(dm, x, &xArr);
    }
}

/* ------------------------------------------------------------------ */
/* Public functions                                                   */
/* ------------------------------------------------------------------ */

void pressure_solver_init_petsc(struct PressureSolverData* ps) {
    memset(ps, 0, sizeof(struct PressureSolverData));
    ps->type = PS_PETSC;

    struct PETScContext* ctx = (struct PETScContext*)calloc(1, sizeof(struct PETScContext));
    ctx->comm = malloc(sizeof(MPI_Comm));
    *(MPI_Comm*)ctx->comm = MPI_COMM_SELF;
    ctx->procDims[0] = 1; ctx->procDims[1] = 1; ctx->procDims[2] = 1;
    ctx->useMPI = 0;
    ps->petsc = ctx;
}

void pressure_solver_init_petsc_mpi(struct PressureSolverData* ps,
                                    void* comm_ptr,
                                    const int localExtent[6],
                                    const int periodic[3],
                                    const int procDims[3])
{
    memset(ps, 0, sizeof(struct PressureSolverData));
    ps->type = PS_PETSC;

    struct PETScContext* ctx = (struct PETScContext*)calloc(1, sizeof(struct PETScContext));
    ctx->comm = malloc(sizeof(MPI_Comm));
    *(MPI_Comm*)ctx->comm = *(MPI_Comm*)comm_ptr;
    ctx->ilower[0] = localExtent[0]; ctx->ilower[1] = localExtent[2]; ctx->ilower[2] = localExtent[4];
    ctx->iupper[0] = localExtent[1] - 1; ctx->iupper[1] = localExtent[3] - 1; ctx->iupper[2] = localExtent[5] - 1;
    memcpy(ctx->periodic, periodic, 3 * sizeof(int));
    memcpy(ctx->procDims, procDims, 3 * sizeof(int));
    ctx->useMPI = 1;
    ps->petsc = ctx;
}

void petsc_pressure_solver_free(struct PETScContext* ctx) {
    if (!ctx) return;

    KSP ksp = (KSP)ctx->ksp;
    Vec x = (Vec)ctx->x;
    Vec b = (Vec)ctx->b;
    Mat A = (Mat)ctx->A;
    DM dm = (DM)ctx->dm;

    if (ksp) { KSPDestroy(&ksp); ctx->ksp = NULL; }
    if (x)   { VecDestroy(&x);   ctx->x   = NULL; }
    if (b)   { VecDestroy(&b);   ctx->b   = NULL; }
    if (A)   { MatDestroy(&A);   ctx->A   = NULL; }
    if (dm)  { DMDestroy(&dm);   ctx->dm  = NULL; }

    if (ctx->petscComm) {
        MPI_Comm petscComm = *(MPI_Comm*)ctx->petscComm;
        if (petscComm != MPI_COMM_NULL) {
            MPI_Comm_free(&petscComm);
        }
        free(ctx->petscComm);
        ctx->petscComm = NULL;
    }

    free(ctx->comm);
    ctx->comm = NULL;
    ctx->initialized = 0;
}

int petsc_pressure_solve(struct PETScContext* ctx,
                         const struct RectilinearMesh* mesh,
                         const double* rho,
                         const double* rhoc2,
                         const double* rhs,
                         double* pressure,
                         size_t fieldSize,
                         double dt,
                         double tolerance,
                         int maxIter,
                         struct HaloExchange* halo)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;

    /* Lazy one-time initialization */
    if (!ctx->initialized) {
        if (!ctx->useMPI) {
            ctx->ilower[0] = 0; ctx->ilower[1] = 0; ctx->ilower[2] = 0;
            ctx->iupper[0] = nx - 1;
            ctx->iupper[1] = (mesh->dim >= 2) ? ny - 1 : 0;
            ctx->iupper[2] = (mesh->dim >= 3) ? nz - 1 : 0;
            ctx->periodic[0] = (mesh->bc[XLOW] == BC_PERIODIC) ? 1 : 0;
            if (mesh->dim >= 2)
                ctx->periodic[1] = (mesh->bc[YLOW] == BC_PERIODIC) ? 1 : 0;
            if (mesh->dim >= 3)
                ctx->periodic[2] = (mesh->bc[ZLOW] == BC_PERIODIC) ? 1 : 0;
        }
        setupPETSc(ctx, mesh);
    }

    DM dm = (DM)ctx->dm;
    Mat A = (Mat)ctx->A;
    Vec b = (Vec)ctx->b;
    Vec x = (Vec)ctx->x;
    KSP ksp = (KSP)ctx->ksp;

    assembleSystem(ctx, mesh, rho, rhoc2, rhs, pressure, dt, halo);

    KSPSetOperators(ksp, A, A);
    KSPSetTolerances(ksp, 1e-50, tolerance, PETSC_DEFAULT, maxIter);

#ifndef NDEBUG
    {
        MPI_Comm dbgComm = *(MPI_Comm*)ctx->comm;
        PetscReal rtol_out, atol_out, dtol_out;
        PetscInt maxits_out;
        KSPGetTolerances(ksp, &rtol_out, &atol_out, &dtol_out, &maxits_out);
        int rank = 0;
        MPI_Comm_rank(dbgComm, &rank);
        if (rank == 0) {
            PetscPrintf(dbgComm, "  PETSc KSP: rtol=%.2e atol=%.2e dtol=%.2e maxits=%d\n",
                        rtol_out, atol_out, dtol_out, (int)maxits_out);
        }
    }
#endif

    KSPSolve(ksp, b, x);

#ifndef NDEBUG
    {
        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        PetscInt numIter;
        KSPGetIterationNumber(ksp, &numIter);
        PetscReal rnorm;
        KSPGetResidualNorm(ksp, &rnorm);
        int rank = 0;
        if (ctx->useMPI) MPI_Comm_rank(*(MPI_Comm*)ctx->comm, &rank);
        if (rank == 0) {
            std::fprintf(stderr, "  KSP: %d its, rnorm=%.3e, reason=%d\n",
                         (int)numIter, (double)rnorm, (int)reason);
        }
    }
#endif

    /* Extract solution */
    PetscInt xs, ys, zs, xm, ym, zm;
    if (ctx->dim == 1) {
        DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL);
        ys = 0; zs = 0; ym = 1; zm = 1;
    } else if (ctx->dim == 2) {
        DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);
        zs = 0; zm = 1;
    } else {
        DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);
    }

    if (ctx->dim == 1) {
        const PetscScalar *xArr;
        DMDAVecGetArrayRead(dm, x, &xArr);
        for (PetscInt i = xs; i < xs + xm; ++i) {
            const int li = (int)(i - xs);
            pressure[mesh_index(mesh, li, 0, 0)] = xArr[i];
        }
        DMDAVecRestoreArrayRead(dm, x, &xArr);
    } else if (ctx->dim == 2) {
        const PetscScalar *const *xArr;
        DMDAVecGetArrayRead(dm, x, &xArr);
        for (PetscInt j = ys; j < ys + ym; ++j) {
            for (PetscInt i = xs; i < xs + xm; ++i) {
                const int li = (int)(i - xs);
                const int lj = (int)(j - ys);
                pressure[mesh_index(mesh, li, lj, 0)] = xArr[j][i];
            }
        }
        DMDAVecRestoreArrayRead(dm, x, &xArr);
    } else {
        const PetscScalar *const *const *xArr;
        DMDAVecGetArrayRead(dm, x, &xArr);
        for (PetscInt k = zs; k < zs + zm; ++k) {
            for (PetscInt j = ys; j < ys + ym; ++j) {
                for (PetscInt i = xs; i < xs + xm; ++i) {
                    const int li = (int)(i - xs);
                    const int lj = (int)(j - ys);
                    const int lk = (int)(k - zs);
                    pressure[mesh_index(mesh, li, lj, lk)] = xArr[k][j][i];
                }
            }
        }
        DMDAVecRestoreArrayRead(dm, x, &xArr);
    }

    PetscInt numIter;
    KSPGetIterationNumber(ksp, &numIter);
    return (int)numIter;
}

#endif /* SIFV_HAS_PETSC */
