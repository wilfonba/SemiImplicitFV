#include "PressureSolver.hpp"
#include "PressureLaplacian.hpp"
#include "RectilinearMesh.hpp"
#include "HaloExchange.hpp"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>

/* ================================================================== */
/* Gauss-Seidel implementation                                        */
/* ================================================================== */

static int gs_solve(const struct RectilinearMesh* mesh,
                    const double* rho,
                    const double* rhoc2,
                    const double* rhs,
                    double* pressure,
                    size_t fieldSize,
                    double dt,
                    double tolerance,
                    int maxIter)
{
    double dt2 = dt * dt;
    double maxResidual = 0.0;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh_fill_scalar_ghosts(mesh, pressure);

        maxResidual = 0.0;

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, &offDiag);

                    double denom = 1.0 + coeff * diagL;
                    double pOld = pressure[idx];
                    pressure[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = fabs(pressure[idx] - pOld);
                    if (residual > maxResidual) maxResidual = residual;
                }
            }
        }

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }
    return maxIter;
}

static int gs_solve_mpi(const struct RectilinearMesh* mesh,
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
    double dt2 = dt * dt;
    double maxResidual = 0.0;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh_fill_scalar_ghosts_mpi(mesh, pressure, halo);

        maxResidual = 0.0;

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, &offDiag);

                    double denom = 1.0 + coeff * diagL;
                    double pOld = pressure[idx];
                    pressure[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = fabs(pressure[idx] - pOld);
                    if (residual > maxResidual) maxResidual = residual;
                }
            }
        }

        /* Global residual reduction */
        MPI_Allreduce(MPI_IN_PLACE, &maxResidual, 1, MPI_DOUBLE, MPI_MAX,
                      halo->mpi->cartComm);

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }
    return maxIter;
}

/* ================================================================== */
/* Jacobi implementation                                              */
/* ================================================================== */

static int jacobi_solve(const struct RectilinearMesh* mesh,
                        const double* rho,
                        const double* rhoc2,
                        const double* rhs,
                        double* pressure,
                        double* pNew,
                        size_t fieldSize,
                        double dt,
                        double tolerance,
                        int maxIter)
{
    double dt2 = dt * dt;
    memset(pNew, 0, fieldSize * sizeof(double));

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh_fill_scalar_ghosts(mesh, pressure);

        double maxResidual = 0.0;

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, &offDiag);

                    double denom = 1.0 + coeff * diagL;
                    pNew[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = fabs(pNew[idx] - pressure[idx]);
                    if (residual > maxResidual) maxResidual = residual;
                }
            }
        }

        for (int k = 0; k < mesh->nz; ++k)
            for (int j = 0; j < mesh->ny; ++j)
                for (int i = 0; i < mesh->nx; ++i)
                    pressure[mesh_index(mesh, i, j, k)] = pNew[mesh_index(mesh, i, j, k)];

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

static int jacobi_solve_mpi(const struct RectilinearMesh* mesh,
                            const double* rho,
                            const double* rhoc2,
                            const double* rhs,
                            double* pressure,
                            double* pNew,
                            size_t fieldSize,
                            double dt,
                            double tolerance,
                            int maxIter,
                            struct HaloExchange* halo)
{
    double dt2 = dt * dt;
    memset(pNew, 0, fieldSize * sizeof(double));

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh_fill_scalar_ghosts_mpi(mesh, pressure, halo);

        double maxResidual = 0.0;

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, &offDiag);

                    double denom = 1.0 + coeff * diagL;
                    pNew[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = fabs(pNew[idx] - pressure[idx]);
                    if (residual > maxResidual) maxResidual = residual;
                }
            }
        }

        for (int k = 0; k < mesh->nz; ++k)
            for (int j = 0; j < mesh->ny; ++j)
                for (int i = 0; i < mesh->nx; ++i)
                    pressure[mesh_index(mesh, i, j, k)] = pNew[mesh_index(mesh, i, j, k)];

        /* Global residual reduction */
        MPI_Allreduce(MPI_IN_PLACE, &maxResidual, 1, MPI_DOUBLE, MPI_MAX,
                      halo->mpi->cartComm);

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

/* ================================================================== */
/* Public API                                                         */
/* ================================================================== */

void pressure_solver_init(struct PressureSolverData* ps, enum PressureSolverType type) {
    memset(ps, 0, sizeof(struct PressureSolverData));
    ps->type = type;
    ps->petsc = NULL;
    ps->scratchBuf = NULL;
    ps->scratchSize = 0;
}

void pressure_solver_free(struct PressureSolverData* ps) {
    if (ps->scratchBuf) {
        free(ps->scratchBuf);
        ps->scratchBuf = NULL;
        ps->scratchSize = 0;
    }
#ifdef SIFV_HAS_PETSC
    if (ps->petsc) {
        /* PETSc cleanup is handled in PETScPressureSolver.cpp */
        extern void petsc_pressure_solver_free(struct PETScContext* ctx);
        petsc_pressure_solver_free(ps->petsc);
        free(ps->petsc);
        ps->petsc = NULL;
    }
#endif
}

/* Ensure scratch buffer is at least fieldSize for Jacobi */
static void ensure_scratch(struct PressureSolverData* ps, size_t fieldSize) {
    if (ps->scratchSize < fieldSize) {
        free(ps->scratchBuf);
        ps->scratchBuf = (double*)calloc(fieldSize, sizeof(double));
        ps->scratchSize = fieldSize;
    }
}

int pressure_solve(struct PressureSolverData* ps,
                   const struct RectilinearMesh* mesh,
                   const double* rho,
                   const double* rhoc2,
                   const double* rhs,
                   double* pressure,
                   size_t fieldSize,
                   double dt,
                   double tolerance,
                   int maxIter)
{
    switch (ps->type) {
        case PS_GAUSS_SEIDEL:
            return gs_solve(mesh, rho, rhoc2, rhs, pressure, fieldSize,
                            dt, tolerance, maxIter);

        case PS_JACOBI:
            ensure_scratch(ps, fieldSize);
            return jacobi_solve(mesh, rho, rhoc2, rhs, pressure, ps->scratchBuf,
                                fieldSize, dt, tolerance, maxIter);

        case PS_PETSC:
#ifdef SIFV_HAS_PETSC
        {
            extern int petsc_pressure_solve(
                struct PETScContext* ctx,
                const struct RectilinearMesh* mesh,
                const double* rho, const double* rhoc2,
                const double* rhs, double* pressure,
                size_t fieldSize,
                double dt, double tolerance, int maxIter,
                struct HaloExchange* halo);
            return petsc_pressure_solve(ps->petsc, mesh, rho, rhoc2,
                                        rhs, pressure, fieldSize,
                                        dt, tolerance, maxIter, NULL);
        }
#else
            return maxIter; /* PETSc not available */
#endif
    }
    return maxIter;
}

int pressure_solve_mpi(struct PressureSolverData* ps,
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
    switch (ps->type) {
        case PS_GAUSS_SEIDEL:
            return gs_solve_mpi(mesh, rho, rhoc2, rhs, pressure, fieldSize,
                                dt, tolerance, maxIter, halo);

        case PS_JACOBI:
            ensure_scratch(ps, fieldSize);
            return jacobi_solve_mpi(mesh, rho, rhoc2, rhs, pressure, ps->scratchBuf,
                                    fieldSize, dt, tolerance, maxIter, halo);

        case PS_PETSC:
#ifdef SIFV_HAS_PETSC
        {
            extern int petsc_pressure_solve(
                struct PETScContext* ctx,
                const struct RectilinearMesh* mesh,
                const double* rho, const double* rhoc2,
                const double* rhs, double* pressure,
                size_t fieldSize,
                double dt, double tolerance, int maxIter,
                struct HaloExchange* halo);
            return petsc_pressure_solve(ps->petsc, mesh, rho, rhoc2,
                                        rhs, pressure, fieldSize,
                                        dt, tolerance, maxIter, halo);
        }
#else
            return maxIter;
#endif
    }
    return maxIter;
}

/* ================================================================== */
/* PETSc init functions (stubs when PETSc unavailable)                */
/* ================================================================== */

#ifndef SIFV_HAS_PETSC

void pressure_solver_init_petsc(struct PressureSolverData* ps) {
    memset(ps, 0, sizeof(struct PressureSolverData));
    ps->type = PS_PETSC;
    ps->petsc = NULL;
}

void pressure_solver_init_petsc_mpi(struct PressureSolverData* ps,
                                    void* /*comm*/,
                                    const int /*localExtent*/[6],
                                    const int /*periodic*/[3],
                                    const int /*procDims*/[3])
{
    memset(ps, 0, sizeof(struct PressureSolverData));
    ps->type = PS_PETSC;
    ps->petsc = NULL;
}

#endif /* !SIFV_HAS_PETSC */
