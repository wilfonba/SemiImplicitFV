#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "State.hpp"
#include <vector>
#include <cstddef>

namespace SemiImplicitFV {

// ReconstructionOrder enum is defined in SimulationConfig.hpp

class Reconstructor {
public:
    explicit Reconstructor(ReconstructionOrder order = ReconstructionOrder::WENO1,
                           double wenoEps = 1e-6);

    /// Allocate face arrays for the given mesh dimensions.
    void allocate(const RectilinearMesh& mesh);

    /// Perform reconstruction on all faces for all active dimensions.
    /// Precondition: ghost cells in state must already be filled.
    void reconstruct(const SimulationConfig& config,
            const RectilinearMesh& mesh,
            const SolutionState& state);

    /// Minimum ghost cells needed for the chosen order.
    int requiredGhostCells() const;

    /// Access reconstructed face states (X-direction).
    const PrimitiveState& xFaceLeft(std::size_t f)  const { return xLeft_[f]; }
    const PrimitiveState& xFaceRight(std::size_t f) const { return xRight_[f]; }

    /// Access reconstructed face states (Y-direction). Only valid if dim >= 2.
    const PrimitiveState& yFaceLeft(std::size_t f)  const { return yLeft_[f]; }
    const PrimitiveState& yFaceRight(std::size_t f) const { return yRight_[f]; }

    /// Access reconstructed face states (Z-direction). Only valid if dim >= 3.
    const PrimitiveState& zFaceLeft(std::size_t f)  const { return zLeft_[f]; }
    const PrimitiveState& zFaceRight(std::size_t f) const { return zRight_[f]; }

    /// Face index from (i,j,k) triplet.
    /// X-face between cells (i-1,j,k) and (i,j,k): i in [0, nx], j in [0,ny), k in [0,nz)
    std::size_t xFaceIndex(int i, int j, int k) const;
    /// Y-face between cells (i,j-1,k) and (i,j,k): i in [0,nx), j in [0, ny], k in [0,nz)
    std::size_t yFaceIndex(int i, int j, int k) const;
    /// Z-face between cells (i,j,k-1) and (i,j,k): i in [0,nx), j in [0,ny), k in [0, nz]
    std::size_t zFaceIndex(int i, int j, int k) const;

    std::size_t numXFaces() const { return numXFaces_; }
    std::size_t numYFaces() const { return numYFaces_; }
    std::size_t numZFaces() const { return numZFaces_; }

    ReconstructionOrder order() const { return order_; }

private:
    ReconstructionOrder order_;
    double wenoEps_;
    int dim_ = 0;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::size_t numXFaces_ = 0, numYFaces_ = 0, numZFaces_ = 0;

    std::vector<PrimitiveState> xLeft_, xRight_;
    std::vector<PrimitiveState> yLeft_, yRight_;
    std::vector<PrimitiveState> zLeft_, zRight_;

    void reconstructX(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);
    void reconstructY(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);
    void reconstructZ(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state);

    static double weno3Left(const double* v, double eps);
    static double weno3Right(const double* v, double eps);
    static double weno5Left(const double* v, double eps);
    static double weno5Right(const double* v, double eps);
    static double upwind3Left(const double* v, double eps);
    static double upwind3Right(const double* v, double eps);
    static double upwind5Left(const double* v, double eps);
    static double upwind5Right(const double* v, double eps);
};

} // namespace SemiImplicitFV

#endif // RECONSTRUCTION_HPP

