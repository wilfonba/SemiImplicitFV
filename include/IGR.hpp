#ifndef IGR_HPP
#define IGR_HPP

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include <array>
#include <string>

namespace SemiImplicitFV { class HaloExchange; }
namespace SemiImplicitFV { class ImmersedBoundaryMethod; }

namespace SemiImplicitFV {

// Velocity gradient tensor (3x3)
using GradientTensor = std::array<std::array<double, 3>, 3>;

// IGRParams is defined in SimulationConfig.hpp

// Information Geometric Regularization solver
// Computes entropic pressure Σ from the elliptic equation:
//   α[tr(∇u)² + tr²(∇u)] = Σ/ρ - α∇·(∇Σ/ρ)
class IGRSolver {
public:
    explicit IGRSolver(const IGRParams& params = IGRParams());
    ~IGRSolver() = default;

    void setParameters(const IGRParams& params) { params_ = params; }
    void setIBM(ImmersedBoundaryMethod* ibm) { ibm_ = ibm; }
    const IGRParams& parameters() const { return params_; }

    // Compute alpha from mesh spacing
    double computeAlpha(double dx) const;

    // Compute the RHS of the elliptic equation from velocity gradients
    // RHS = α[tr(∇u)² + tr²(∇u)]
    double computeIGRRhs(const SimulationConfig& config, const GradientTensor& gradU, double alpha) const;

    // Solve for entropic pressure Σ at a single cell
    // Uses Jacobi/Gauss-Seidel iteration with warm start
    // Returns the converged value of Σ
    void solveEntropicPressure(const SimulationConfig& config,
            const RectilinearMesh& mesh,
            SolutionState& state,
            std::vector<GradientTensor> gradU);

    void solveEntropicPressure(const SimulationConfig& config,
            const RectilinearMesh& mesh,
            SolutionState& state,
            std::vector<GradientTensor> gradU,
            HaloExchange& halo);

    // Compute velocity gradient tensor from cell-centered velocities
    // Uses central differences
    static GradientTensor computeVelocityGradient(
        const std::array<double, 3>& u_xm,  // u at x-1
        const std::array<double, 3>& u_xp,  // u at x+1
        const std::array<double, 3>& u_ym,  // u at y-1
        const std::array<double, 3>& u_yp,  // u at y+1
        const std::array<double, 3>& u_zm,  // u at z-1
        const std::array<double, 3>& u_zp,  // u at z+1
        double dx, double dy, double dz,
        int dim
    );

private:
    IGRParams params_;
    ImmersedBoundaryMethod* ibm_ = nullptr;
};

} // namespace SemiImplicitFV

#endif // IGR_HPP

