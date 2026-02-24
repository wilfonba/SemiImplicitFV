#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include "MPIContext.hpp"
#include "HaloExchange.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"

struct ExplicitSolverWork;
struct SemiImplicitSolverWork;

struct Runtime {
    int rank;
    int size;
    int globalNx;
    int globalNy;
    int globalNz;
    struct MPIContext* mpiCtx;
    struct HaloExchange* halo;
};

void runtime_init(struct Runtime* rt, int* argc, char*** argv);
void runtime_free(struct Runtime* rt);

/* Mesh creation (1D) */
void runtime_create_uniform_mesh_1d(
    struct Runtime* rt,
    struct RectilinearMesh* mesh,
    const struct SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    const int periods[3]);

/* Mesh creation (2D) */
void runtime_create_uniform_mesh_2d(
    struct Runtime* rt,
    struct RectilinearMesh* mesh,
    const struct SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    const int periods[3]);

/* Mesh creation (3D) */
void runtime_create_uniform_mesh_3d(
    struct Runtime* rt,
    struct RectilinearMesh* mesh,
    const struct SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    int globalNz, double zMin, double zMax,
    const int periods[3]);

/* Boundary condition setup (only applies if this rank owns the boundary) */
void runtime_set_bc(struct Runtime* rt, struct RectilinearMesh* mesh,
                    int face, enum BoundaryCondition bc);

/* Solver attachment: creates HaloExchange and sets it on the solver */
void runtime_attach_explicit(struct Runtime* rt, struct ExplicitSolverWork* solver,
                             const struct RectilinearMesh* mesh);
void runtime_attach_semi_implicit(struct Runtime* rt, struct SemiImplicitSolverWork* solver,
                                  const struct RectilinearMesh* mesh);

/* Field smoothing */
void runtime_smooth_fields(struct Runtime* rt, struct SolutionState* state,
                           const struct RectilinearMesh* mesh, int nIters);
void runtime_smooth_fields_config(struct Runtime* rt, struct SolutionState* state,
                                  const struct RectilinearMesh* mesh, int nIters,
                                  const struct SimulationConfig* config);

/* Global reduction */
double runtime_reduce_max(struct Runtime* rt, double localValue);

/* Console output (rank 0 only) */
void runtime_print(struct Runtime* rt, const char* msg);

#endif /* RUNTIME_HPP */
