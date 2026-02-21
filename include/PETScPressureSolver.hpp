#ifndef PETSC_PRESSURE_SOLVER_HPP
#define PETSC_PRESSURE_SOLVER_HPP

#include "PressureSolver.hpp"
#include "RectilinearMesh.hpp"
#include <array>
#include <vector>
#include <mpi.h>

// Forward-declare PETSc opaque types to avoid leaking petsc.h
typedef struct _p_DM*  DM;
typedef struct _p_Mat* Mat;
typedef struct _p_Vec* Vec;
typedef struct _p_KSP* KSP;

namespace SemiImplicitFV {

/// CG + GAMG (algebraic multigrid) pressure solver via PETSc.
///
/// Drop-in replacement for GaussSeidelPressureSolver that provides
/// mesh-independent O(1) convergence for the semi-implicit pressure equation
/// (I + dt^2 * rho*c^2 * L) p = rhs.
///
/// Works in 1D, 2D, and 3D. DMDA handles periodic boundary conditions natively.
class PETScPressureSolver : public PressureSolver {
public:
    /// Serial constructor (uses MPI_COMM_SELF).
    PETScPressureSolver();

    /// MPI-aware constructor.
    /// @param comm       MPI communicator (typically from MPIContext::comm()).
    /// @param localExtent  {i0, i1, j0, j1, k0, k1} global cell extents for this rank.
    /// @param periodic   Periodic flags per direction (0 or 1).
    /// @param procDims   Number of MPI ranks per direction (must match domain decomposition).
    PETScPressureSolver(MPI_Comm comm,
                        const std::array<int,6>& localExtent,
                        const std::array<int,3>& periodic,
                        const std::array<int,3>& procDims);

    ~PETScPressureSolver();

    // Non-copyable, non-movable (PETSc handles are not trivially relocatable)
    PETScPressureSolver(const PETScPressureSolver&) = delete;
    PETScPressureSolver& operator=(const PETScPressureSolver&) = delete;

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

    std::string name() const override { return "PETScCG+GAMG"; }

private:
    MPI_Comm comm_;
    MPI_Comm petscComm_ = MPI_COMM_NULL;  // remapped communicator for DMDA
    std::array<int,3> ilower_;
    std::array<int,3> iupper_;
    std::array<int,3> periodic_;
    std::array<int,3> procDims_;
    bool useMPI_;

    // PETSc handles (opaque pointers)
    DM   dm_  = nullptr;
    Mat  A_   = nullptr;
    Vec  b_   = nullptr;
    Vec  x_   = nullptr;
    KSP  ksp_ = nullptr;

    bool initialized_ = false;
    int dim_ = 0;

    void setupPETSc(const RectilinearMesh& mesh);
    void destroyPETSc();

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

#endif // PETSC_PRESSURE_SOLVER_HPP
