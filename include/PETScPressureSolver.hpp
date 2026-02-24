#ifndef PETSC_PRESSURE_SOLVER_HPP
#define PETSC_PRESSURE_SOLVER_HPP

#include "PressureSolver.hpp"

#ifdef SIFV_HAS_PETSC

/* These functions are called from PressureSolver.cpp via extern declarations.
   They are implemented in PETScPressureSolver.cpp. */

void pressure_solver_init_petsc(struct PressureSolverData* ps);

void pressure_solver_init_petsc_mpi(struct PressureSolverData* ps,
                                    void* comm,
                                    const int localExtent[6],
                                    const int periodic[3],
                                    const int procDims[3]);

void petsc_pressure_solver_free(struct PETScContext* ctx);

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
                         struct HaloExchange* halo);

#endif /* SIFV_HAS_PETSC */

#endif /* PETSC_PRESSURE_SOLVER_HPP */
