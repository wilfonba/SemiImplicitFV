#ifndef RUNTIME_HPP
#define RUNTIME_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include <array>
#include <iostream>
#include <sstream>
#include <memory>
#include "MPIContext.hpp"
#include "HaloExchange.hpp"

namespace SemiImplicitFV {

class ExplicitSolver;
class SemiImplicitSolver;
class ImmersedBoundaryMethod;

/// Unified initialization / utility class that abstracts MPI vs serial.
/// In serial mode it is a thin passthrough.  In MPI mode it owns
/// MPIContext and HaloExchange internally.
class Runtime {
public:
    Runtime(int& argc, char**& argv);
    ~Runtime();

    // Non-copyable, non-movable (owns MPI lifecycle)
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool isRoot() const { return rank_ == 0; }

    // --- Mesh creation (1D) ---
    RectilinearMesh createUniformMesh(
        const SimulationConfig& config,
        int globalNx, double xMin, double xMax,
        const std::array<int,3>& periods = {0,0,0});

    // --- Mesh creation (2D) ---
    RectilinearMesh createUniformMesh(
        const SimulationConfig& config,
        int globalNx, double xMin, double xMax,
        int globalNy, double yMin, double yMax,
        const std::array<int,3>& periods = {0,0,0});

    // --- Mesh creation (3D) ---
    RectilinearMesh createUniformMesh(
        const SimulationConfig& config,
        int globalNx, double xMin, double xMax,
        int globalNy, double yMin, double yMax,
        int globalNz, double zMin, double zMax,
        const std::array<int,3>& periods = {0,0,0});

    // --- BC setup ---
    void setBoundaryCondition(RectilinearMesh& mesh, int face, BoundaryCondition bc);

    // --- Solver attachment ---
    void attachSolver(ExplicitSolver& solver, const RectilinearMesh& mesh);
    void attachSolver(SemiImplicitSolver& solver, const RectilinearMesh& mesh);

    // --- IBM attachment ---
    void attachIBM(ImmersedBoundaryMethod& ibm, ExplicitSolver& solver);

    // --- Field smoothing ---
    void smoothFields(SolutionState& state, const RectilinearMesh& mesh, int nIters);

    // --- Reductions ---
    double reduceMax(double localValue);

    // --- Console output (rank 0 only) ---
    template <typename... Args>
    void print(Args&&... args) {
        if (rank_ != 0) return;
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        std::cout << oss.str();
    }

    // --- Accessors for VTKSession ---
    int globalNx() const { return globalNx_; }
    int globalNy() const { return globalNy_; }
    int globalNz() const { return globalNz_; }

    const MPIContext& mpiContext() const { return *mpiCtx_; }
    HaloExchange* haloExchange() { return halo_.get(); }

private:
    int rank_ = 0;
    int size_ = 1;
    int globalNx_ = 0, globalNy_ = 0, globalNz_ = 0;

    std::unique_ptr<MPIContext> mpiCtx_;
    std::unique_ptr<HaloExchange> halo_;
};

} // namespace SemiImplicitFV

#endif // RUNTIME_HPP
