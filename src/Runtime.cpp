#include "Runtime.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"

#include <mpi.h>
#ifdef SIFV_HAS_PETSC
#include <petsc.h>
#endif
#include <omp.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>

static void linspace(double a, double b, int n, double* out) {
    for (int i = 0; i < n; ++i)
        out[i] = a + (b - a) * i / (n - 1);
}

/* Pin each MPI rank to a distinct GPU.  Without this, every rank's OpenMP
 * target offload defaults to device 0 and ranks serialize on the same GPU,
 * which makes 4-rank runs slower than 1-rank runs.  We pick the local-rank
 * environment variable that the active MPI launcher exports. */
static void bind_rank_to_device(int worldRank) {
    int ndev = omp_get_num_devices();
    if (ndev <= 0) return;

    int localRank = -1;
    const char* envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",  /* OpenMPI */
        "MV2_COMM_WORLD_LOCAL_RANK",   /* MVAPICH2 */
        "SLURM_LOCALID",               /* srun */
        "MPI_LOCALRANKID",             /* Intel MPI */
        "PMI_LOCAL_RANK",              /* MPICH/PMI */
        NULL
    };
    for (int i = 0; envs[i]; ++i) {
        const char* s = std::getenv(envs[i]);
        if (s && *s) { localRank = std::atoi(s); break; }
    }
    if (localRank < 0) localRank = worldRank;  /* single-node fallback */

    int dev = localRank % ndev;
    omp_set_default_device(dev);

    if (worldRank == 0) {
        std::fprintf(stderr,
            "[sifv] GPU binding: %d device(s) visible; rank 0 → device %d\n",
            ndev, dev);
    }
}

void runtime_init(Runtime* rt, int* argc, char*** argv) {
    std::memset(rt, 0, sizeof(Runtime));
    MPI_Init(argc, argv);
#ifdef SIFV_HAS_PETSC
    PetscInitialize(argc, argv, NULL, NULL);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rt->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt->size);
    bind_rank_to_device(rt->rank);
    rt->mpiCtx = NULL;
    rt->halo = NULL;
}

void runtime_free(Runtime* rt) {
    if (rt->halo) {
        halo_free(rt->halo);
        std::free(rt->halo);
        rt->halo = NULL;
    }
    if (rt->mpiCtx) {
        mpi_context_free(rt->mpiCtx);
        std::free(rt->mpiCtx);
        rt->mpiCtx = NULL;
    }
#ifdef SIFV_HAS_PETSC
    PetscFinalize();
#endif
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

void runtime_create_uniform_mesh_1d(
    Runtime* rt,
    RectilinearMesh* mesh,
    const SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    const int periods[3])
{
    rt->globalNx = globalNx;
    rt->globalNy = 1;
    rt->globalNz = 1;

    double* gxNodes = (double*)std::malloc((size_t)(globalNx + 1) * sizeof(double));
    linspace(xMin, xMax, globalNx + 1, gxNodes);
    double gyNodes[2] = {0.0, 1.0};
    double gzNodes[2] = {0.0, 1.0};

    rt->mpiCtx = (MPIContext*)std::malloc(sizeof(MPIContext));
    int procsPerDir[3] = {0, 0, 0};
    mpi_context_create(rt->mpiCtx, globalNx, 1, 1,
                       gxNodes, globalNx + 1, gyNodes, 2, gzNodes, 2,
                       config->dim, periods, procsPerDir);

    rt->rank = rt->mpiCtx->rank;
    rt->size = rt->mpiCtx->size;

    /* Create mesh from local nodes */
    double yNodes1[2] = {0.0, 1.0};
    double zNodes1[2] = {0.0, 1.0};
    mesh_init(mesh, config,
              rt->mpiCtx->localXNodes, rt->mpiCtx->localXNodesSize,
              yNodes1, 2, zNodes1, 2);

    std::free(gxNodes);
}

void runtime_create_uniform_mesh_2d(
    Runtime* rt,
    RectilinearMesh* mesh,
    const SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    const int periods[3])
{
    rt->globalNx = globalNx;
    rt->globalNy = globalNy;
    rt->globalNz = 1;

    double* gxNodes = (double*)std::malloc((size_t)(globalNx + 1) * sizeof(double));
    double* gyNodes = (double*)std::malloc((size_t)(globalNy + 1) * sizeof(double));
    linspace(xMin, xMax, globalNx + 1, gxNodes);
    linspace(yMin, yMax, globalNy + 1, gyNodes);
    double gzNodes[2] = {0.0, 1.0};

    rt->mpiCtx = (MPIContext*)std::malloc(sizeof(MPIContext));
    int procsPerDir[3] = {0, 0, 0};
    mpi_context_create(rt->mpiCtx, globalNx, globalNy, 1,
                       gxNodes, globalNx + 1, gyNodes, globalNy + 1, gzNodes, 2,
                       config->dim, periods, procsPerDir);

    rt->rank = rt->mpiCtx->rank;
    rt->size = rt->mpiCtx->size;

    double zNodes1[2] = {0.0, 1.0};
    mesh_init(mesh, config,
              rt->mpiCtx->localXNodes, rt->mpiCtx->localXNodesSize,
              rt->mpiCtx->localYNodes, rt->mpiCtx->localYNodesSize,
              zNodes1, 2);

    std::free(gxNodes);
    std::free(gyNodes);
}

void runtime_create_uniform_mesh_3d(
    Runtime* rt,
    RectilinearMesh* mesh,
    const SimulationConfig* config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    int globalNz, double zMin, double zMax,
    const int periods[3])
{
    rt->globalNx = globalNx;
    rt->globalNy = globalNy;
    rt->globalNz = globalNz;

    double* gxNodes = (double*)std::malloc((size_t)(globalNx + 1) * sizeof(double));
    double* gyNodes = (double*)std::malloc((size_t)(globalNy + 1) * sizeof(double));
    double* gzNodes = (double*)std::malloc((size_t)(globalNz + 1) * sizeof(double));
    linspace(xMin, xMax, globalNx + 1, gxNodes);
    linspace(yMin, yMax, globalNy + 1, gyNodes);
    linspace(zMin, zMax, globalNz + 1, gzNodes);

    rt->mpiCtx = (MPIContext*)std::malloc(sizeof(MPIContext));
    int procsPerDir[3] = {0, 0, 0};
    mpi_context_create(rt->mpiCtx, globalNx, globalNy, globalNz,
                       gxNodes, globalNx + 1, gyNodes, globalNy + 1,
                       gzNodes, globalNz + 1,
                       config->dim, periods, procsPerDir);

    rt->rank = rt->mpiCtx->rank;
    rt->size = rt->mpiCtx->size;

    mesh_init(mesh, config,
              rt->mpiCtx->localXNodes, rt->mpiCtx->localXNodesSize,
              rt->mpiCtx->localYNodes, rt->mpiCtx->localYNodesSize,
              rt->mpiCtx->localZNodes, rt->mpiCtx->localZNodesSize);

    std::free(gxNodes);
    std::free(gyNodes);
    std::free(gzNodes);
}

void runtime_set_bc(Runtime* rt, RectilinearMesh* mesh,
                    int face, BoundaryCondition bc) {
    if (mpi_is_physical_boundary(rt->mpiCtx, face)) {
        mesh_set_bc(mesh, face, bc);
    }
}

void runtime_attach_explicit(Runtime* rt, ExplicitSolverWork* solver,
                             const RectilinearMesh* mesh) {
    rt->halo = (HaloExchange*)std::malloc(sizeof(HaloExchange));
    halo_init(rt->halo, rt->mpiCtx, mesh);
    solver->halo = rt->halo;
}

void runtime_attach_semi_implicit(Runtime* rt, SemiImplicitSolverWork* solver,
                                  const RectilinearMesh* mesh) {
    rt->halo = (HaloExchange*)std::malloc(sizeof(HaloExchange));
    halo_init(rt->halo, rt->mpiCtx, mesh);
    solver->halo = rt->halo;
}

void runtime_smooth_fields(Runtime* rt, SolutionState* state,
                           const RectilinearMesh* mesh, int nIters) {
    state_smooth_mpi(state, mesh, nIters, rt->halo);
}

void runtime_smooth_fields_config(Runtime* rt, SolutionState* state,
                                  const RectilinearMesh* mesh, int nIters,
                                  const SimulationConfig* config) {
    state_smooth_config_mpi(state, mesh, nIters, rt->halo, config);
}

double runtime_reduce_max(Runtime* rt, double localValue) {
    double globalValue = 0.0;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX, rt->mpiCtx->cartComm);
    return globalValue;
}

void runtime_print(Runtime* rt, const char* msg) {
    if (rt->rank != 0) return;
    std::fputs(msg, stdout);
}
