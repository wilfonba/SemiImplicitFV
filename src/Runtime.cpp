#include "Runtime.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include <vector>

namespace SemiImplicitFV {

#ifdef ENABLE_MPI
static std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = a + (b - a) * i / (n - 1);
    return v;
}
#endif

// ---- Constructor / Destructor ----

Runtime::Runtime(int& argc, char**& argv) {
#ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
    (void)argc; (void)argv;
#endif
}

Runtime::~Runtime() {
#ifdef ENABLE_MPI
    halo_.reset();
    mpiCtx_.reset();
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
#endif
}

// ---- Mesh creation (1D) ----

RectilinearMesh Runtime::createUniformMesh(
    const SimulationConfig& config,
    int globalNx, double xMin, double xMax,
    const std::array<int,3>& periods)
{
    globalNx_ = globalNx;
    globalNy_ = 1;
    globalNz_ = 1;

#ifdef ENABLE_MPI
    std::vector<double> gxNodes = linspace(xMin, xMax, globalNx + 1);
    std::vector<double> gyNodes = {0.0, 1.0};
    std::vector<double> gzNodes = {0.0, 1.0};

    mpiCtx_ = std::make_unique<MPIContext>(
        MPIContext::create(globalNx, 1, 1,
                           gxNodes, gyNodes, gzNodes,
                           config.dim, periods));

    rank_ = mpiCtx_->rank();
    size_ = mpiCtx_->size();

    return RectilinearMesh(config, mpiCtx_->localXNodes());
#else
    (void)periods;
    return RectilinearMesh::createUniform(config, globalNx, xMin, xMax);
#endif
}

// ---- Mesh creation (2D) ----

RectilinearMesh Runtime::createUniformMesh(
    const SimulationConfig& config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    const std::array<int,3>& periods)
{
    globalNx_ = globalNx;
    globalNy_ = globalNy;
    globalNz_ = 1;

#ifdef ENABLE_MPI
    std::vector<double> gxNodes = linspace(xMin, xMax, globalNx + 1);
    std::vector<double> gyNodes = linspace(yMin, yMax, globalNy + 1);
    std::vector<double> gzNodes = {0.0, 1.0};

    mpiCtx_ = std::make_unique<MPIContext>(
        MPIContext::create(globalNx, globalNy, 1,
                           gxNodes, gyNodes, gzNodes,
                           config.dim, periods));

    rank_ = mpiCtx_->rank();
    size_ = mpiCtx_->size();

    return RectilinearMesh(config, mpiCtx_->localXNodes(), mpiCtx_->localYNodes());
#else
    (void)periods;
    return RectilinearMesh::createUniform(config, globalNx, xMin, xMax,
                                           globalNy, yMin, yMax);
#endif
}

// ---- Mesh creation (3D) ----

RectilinearMesh Runtime::createUniformMesh(
    const SimulationConfig& config,
    int globalNx, double xMin, double xMax,
    int globalNy, double yMin, double yMax,
    int globalNz, double zMin, double zMax,
    const std::array<int,3>& periods)
{
    globalNx_ = globalNx;
    globalNy_ = globalNy;
    globalNz_ = globalNz;

#ifdef ENABLE_MPI
    std::vector<double> gxNodes = linspace(xMin, xMax, globalNx + 1);
    std::vector<double> gyNodes = linspace(yMin, yMax, globalNy + 1);
    std::vector<double> gzNodes = linspace(zMin, zMax, globalNz + 1);

    mpiCtx_ = std::make_unique<MPIContext>(
        MPIContext::create(globalNx, globalNy, globalNz,
                           gxNodes, gyNodes, gzNodes,
                           config.dim, periods));

    rank_ = mpiCtx_->rank();
    size_ = mpiCtx_->size();

    return RectilinearMesh(config,
                           mpiCtx_->localXNodes(),
                           mpiCtx_->localYNodes(),
                           mpiCtx_->localZNodes());
#else
    (void)periods;
    return RectilinearMesh::createUniform(config, globalNx, xMin, xMax,
                                           globalNy, yMin, yMax,
                                           globalNz, zMin, zMax);
#endif
}

// ---- Boundary conditions ----

void Runtime::setBoundaryCondition(RectilinearMesh& mesh, int face, BoundaryCondition bc) {
#ifdef ENABLE_MPI
    if (mpiCtx_ && mpiCtx_->isPhysicalBoundary(face)) {
        mesh.setBoundaryCondition(face, bc);
    }
#else
    mesh.setBoundaryCondition(face, bc);
#endif
}

// ---- Solver attachment ----

void Runtime::attachSolver(ExplicitSolver& solver, const RectilinearMesh& mesh) {
#ifdef ENABLE_MPI
    if (mpiCtx_) {
        halo_ = std::make_unique<HaloExchange>(*mpiCtx_, mesh);
        solver.setHaloExchange(halo_.get());
    }
#else
    (void)solver; (void)mesh;
#endif
}

void Runtime::attachSolver(SemiImplicitSolver& solver, const RectilinearMesh& mesh) {
#ifdef ENABLE_MPI
    if (mpiCtx_) {
        halo_ = std::make_unique<HaloExchange>(*mpiCtx_, mesh);
        solver.setHaloExchange(halo_.get());
    }
#else
    (void)solver; (void)mesh;
#endif
}

// ---- Field smoothing ----

void Runtime::smoothFields(SolutionState& state, const RectilinearMesh& mesh, int nIters) {
#ifdef ENABLE_MPI
    if (halo_) {
        state.smoothFields(mesh, nIters, *halo_);
    } else {
        state.smoothFields(mesh, nIters);
    }
#else
    state.smoothFields(mesh, nIters);
#endif
}

// ---- Reductions ----

double Runtime::reduceMax(double localValue) {
#ifdef ENABLE_MPI
    if (mpiCtx_) {
        double globalValue = 0.0;
        MPI_Allreduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_MAX, mpiCtx_->comm());
        return globalValue;
    }
#endif
    return localValue;
}

} // namespace SemiImplicitFV
