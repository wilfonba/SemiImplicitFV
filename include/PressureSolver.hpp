#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include <stddef.h>

struct RectilinearMesh;
struct HaloExchange;

enum PressureSolverType {
    PS_GAUSS_SEIDEL,
    PS_JACOBI,
    PS_PETSC
};

/* Opaque handles for PETSc objects (only used when SIFV_HAS_PETSC is defined) */
struct PETScContext {
    void* dm;        /* DM */
    void* A;         /* Mat */
    void* b;         /* Vec */
    void* x;         /* Vec */
    void* ksp;       /* KSP */
    void* comm;      /* MPI_Comm (stored as void* for portability) */
    void* petscComm; /* remapped MPI_Comm for DMDA */
    int ilower[3];
    int iupper[3];
    int periodic[3];
    int procDims[3];
    int useMPI;
    int initialized;
    int dim;
};

struct PressureSolverData {
    enum PressureSolverType type;
    struct PETScContext* petsc;  /* NULL unless type == PS_PETSC */
    double* scratchBuf;         /* scratch buffer for Jacobi solver, NULL otherwise */
    size_t scratchSize;         /* size of scratch buffer */
};

/* Initialize a Gauss-Seidel or Jacobi pressure solver */
void pressure_solver_init(struct PressureSolverData* ps, enum PressureSolverType type);

/* Initialize a PETSc pressure solver (serial) */
void pressure_solver_init_petsc(struct PressureSolverData* ps);

/* Initialize a PETSc pressure solver (MPI-aware) */
void pressure_solver_init_petsc_mpi(struct PressureSolverData* ps,
                                    void* comm,
                                    const int localExtent[6],
                                    const int periodic[3],
                                    const int procDims[3]);

/* Free solver resources */
void pressure_solver_free(struct PressureSolverData* ps);

/* Solve the pressure equation (serial) */
int pressure_solve(struct PressureSolverData* ps,
                   const struct RectilinearMesh* mesh,
                   const double* rho,
                   const double* rhoc2,
                   const double* rhs,
                   double* pressure,
                   size_t fieldSize,
                   double dt,
                   double tolerance,
                   int maxIter);

/* Solve the pressure equation (MPI-aware) */
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
                       struct HaloExchange* halo);

#endif /* PRESSURE_SOLVER_HPP */
