#ifndef IGR_HPP
#define IGR_HPP

#include <array>
#include <string>

namespace SemiImplicitFV {

// Velocity gradient tensor (3x3)
using GradientTensor = std::array<std::array<double, 3>, 3>;

// Parameters for Information Geometric Regularization
struct IGRParams {
    double alphaCoeff;        // Coefficient for alpha = alphaCoeff * dx^2
    int maxIterations;        // Max iterations for elliptic solve
    double tolerance;         // Convergence tolerance for elliptic solve

    IGRParams()
        : alphaCoeff(1.0)
        , maxIterations(5)
        , tolerance(1e-10)
    {}
};

// Information Geometric Regularization solver
// Computes entropic pressure Σ from the elliptic equation:
//   α[tr(∇u)² + tr²(∇u)] = Σ/ρ - α∇·(∇Σ/ρ)
class IGRSolver {
public:
    explicit IGRSolver(const IGRParams& params = IGRParams());
    ~IGRSolver() = default;

    void setParameters(const IGRParams& params) { params_ = params; }
    const IGRParams& parameters() const { return params_; }

    // Compute alpha from mesh spacing
    double computeAlpha(double dx) const;

    // Compute the RHS of the elliptic equation from velocity gradients
    // RHS = α[tr(∇u)² + tr²(∇u)]
    double computeIGRRhs(const GradientTensor& gradU, double alpha) const;

    // Solve for entropic pressure Σ at a single cell
    // Uses Jacobi/Gauss-Seidel iteration with warm start
    // Returns the converged value of Σ
    double solveEntropicPressure(
        double rhs,              // α[tr(∇u)² + tr²(∇u)]
        double rho,              // Density at cell
        double alpha,            // Regularization parameter
        double dx,               // Mesh spacing (for Laplacian discretization)
        double sigmaWarmStart,   // Previous Σ value for warm start
        const std::array<double, 6>& neighborSigmaRho, // Σ/ρ at neighbors (±x, ±y, ±z)
        int nNeighbors = 6       // Number of active neighbors (2*dim)
    ) const;

    // Compute velocity gradient tensor from cell-centered velocities
    // Uses central differences
    static GradientTensor computeVelocityGradient(
        const std::array<double, 3>& u_xm,  // u at x-1
        const std::array<double, 3>& u_xp,  // u at x+1
        const std::array<double, 3>& u_ym,  // u at y-1
        const std::array<double, 3>& u_yp,  // u at y+1
        const std::array<double, 3>& u_zm,  // u at z-1
        const std::array<double, 3>& u_zp,  // u at z+1
        double dx, double dy, double dz
    );

    // Compute trace of a tensor
    static double trace(const GradientTensor& A);

    // Compute tr(A²) = sum of squares of all components
    static double traceSquared(const GradientTensor& A);

private:
    IGRParams params_;
};

} // namespace SemiImplicitFV

#endif // IGR_HPP
