#ifndef HYPRE_PRESSURE_SOLVER_HPP
#define HYPRE_PRESSURE_SOLVER_HPP

#include "PressureSolver.hpp"
#include "RectilinearMesh.hpp"
#include <array>
#include <vector>
#include <mpi.h>

// Forward-declare Hypre opaque types to avoid leaking HYPRE.h
struct hypre_StructGrid_struct;
struct hypre_StructStencil_struct;
struct hypre_StructMatrix_struct;
struct hypre_StructVector_struct;
struct hypre_StructSolver_struct;

namespace SemiImplicitFV {

/// PCG + PFMG (structured multigrid) pressure solver via Hypre.
///
/// Drop-in replacement for GaussSeidelPressureSolver that provides
/// mesh-independent O(1) convergence for the semi-implicit pressure equation
/// (I + dt^2 * rho*c^2 * L) p = rhs.
class HyprePressureSolver : public PressureSolver {
public:
    /// Serial constructor (uses MPI_COMM_SELF).
    HyprePressureSolver();

    /// MPI-aware constructor.
    /// @param comm       MPI communicator (typically from MPIContext::comm()).
    /// @param localExtent  {i0, i1, j0, j1, k0, k1} global cell extents for this rank.
    /// @param periodic   Periodic flags per direction (0 or 1).
    HyprePressureSolver(MPI_Comm comm,
                        const std::array<int,6>& localExtent,
                        const std::array<int,3>& periodic);

    ~HyprePressureSolver();

    // Non-copyable, non-movable (Hypre handles are not trivially relocatable)
    HyprePressureSolver(const HyprePressureSolver&) = delete;
    HyprePressureSolver& operator=(const HyprePressureSolver&) = delete;

    int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter
    ) override;

    int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter,
        HaloExchange& halo
    ) override;

    std::string name() const override { return "HyprePCG+PFMG"; }

private:
    MPI_Comm comm_;
    std::array<int,3> ilower_;
    std::array<int,3> iupper_;
    std::array<int,3> periodic_;
    bool useMPI_;

    // Hypre handles (opaque pointers)
    hypre_StructGrid_struct*    grid_    = nullptr;
    hypre_StructStencil_struct* stencil_ = nullptr;
    hypre_StructMatrix_struct*  matrix_  = nullptr;
    hypre_StructVector_struct*  bVec_    = nullptr;
    hypre_StructVector_struct*  xVec_    = nullptr;
    hypre_StructSolver_struct*  solver_  = nullptr;
    hypre_StructSolver_struct*  precond_ = nullptr;

    bool initialized_ = false;
    int dim_ = 0;

    // Scratch buffers (avoid per-step allocation)
    std::vector<double> matValues_;
    std::vector<double> rhsValues_;
    std::vector<double> solValues_;

    void setupHypre(int dim, int nLocalCells);
    void destroyHypre();

    void assembleSystem(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        const std::vector<double>& pressure,
        double dt,
        HaloExchange* halo
    );

    int solveInternal(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter,
        HaloExchange* halo
    );
};

} // namespace SemiImplicitFV

#endif // HYPRE_PRESSURE_SOLVER_HPP
