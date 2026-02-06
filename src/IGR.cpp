#include "IGR.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

IGRSolver::IGRSolver(const IGRParams& params)
    : params_(params)
{}

double IGRSolver::computeAlpha(double dx) const {
    return params_.alphaCoeff * dx * dx;
}

double IGRSolver::computeIGRRhs(const GradientTensor& gradU, double alpha) const {
    // Compute: α[tr(∇u)² + tr²(∇u)]
    // tr(∇u)² = sum of squares of all components of ∇u
    // tr²(∇u) = (trace of ∇u)² = (divergence of u)²

    double trSq = traceSquared(gradU);  // tr((∇u)²) = sum of A_ij²
    double tr = trace(gradU);           // tr(∇u) = div(u)
    double trSquared = tr * tr;         // tr²(∇u)

    return alpha * (trSq + trSquared);
}

double IGRSolver::solveEntropicPressure(
    double rhs,
    double rho,
    double alpha,
    double dx,
    double sigmaWarmStart,
    const std::array<double, 6>& neighborSigmaRho,
    int nNeighbors
) const {
    // Solve: rhs = Σ/ρ - α∇·(∇Σ/ρ)
    // Discretize Laplacian: ∇·(∇Σ/ρ) ≈ (1/dx²) * sum of (Σ/ρ)_neighbors - 6*(Σ/ρ)_center
    //
    // Let φ = Σ/ρ, then:
    //   rhs = φ - α * Laplacian(φ)
    //   rhs = φ - α * (1/dx²) * [sum(φ_neighbors) - 6*φ]
    //   rhs = φ + (6*α/dx²)*φ - (α/dx²)*sum(φ_neighbors)
    //   rhs = φ*(1 + 6*α/dx²) - (α/dx²)*sum(φ_neighbors)
    //
    // Solving for φ:
    //   φ = [rhs + (α/dx²)*sum(φ_neighbors)] / (1 + 6*α/dx²)
    //
    // Then Σ = ρ * φ

    double dx2 = dx * dx;
    double coeff = alpha / dx2;
    double diagCoeff = 1.0 + nNeighbors * coeff;

    // Sum of neighbor φ = Σ/ρ values
    // neighborSigmaRho = [x-, x+, y-, y+, z-, z+]
    double neighborSum = 0.0;
    for (int i = 0; i < nNeighbors; ++i) {
        neighborSum += neighborSigmaRho[i];
    }

    // Jacobi iteration with warm start
    double phi = sigmaWarmStart / std::max(rho, 1e-14);

    for (int iter = 0; iter < params_.maxIterations; ++iter) {
        double phiNew = (rhs + coeff * neighborSum) / diagCoeff;

        // Check convergence
        if (std::abs(phiNew - phi) < params_.tolerance) {
            phi = phiNew;
            break;
        }

        phi = phiNew;
    }

    // Convert back to Σ
    return rho * phi;
}

GradientTensor IGRSolver::computeVelocityGradient(
    const std::array<double, 3>& u_xm,
    const std::array<double, 3>& u_xp,
    const std::array<double, 3>& u_ym,
    const std::array<double, 3>& u_yp,
    const std::array<double, 3>& u_zm,
    const std::array<double, 3>& u_zp,
    double dx, double dy, double dz
) {
    GradientTensor grad;

    // grad[i][j] = du_i / dx_j
    // Using central differences

    double invDx = 0.5 / dx;
    double invDy = 0.5 / dy;
    double invDz = 0.5 / dz;

    // d/dx derivatives
    grad[0][0] = (u_xp[0] - u_xm[0]) * invDx;  // du/dx
    grad[1][0] = (u_xp[1] - u_xm[1]) * invDx;  // dv/dx
    grad[2][0] = (u_xp[2] - u_xm[2]) * invDx;  // dw/dx

    // d/dy derivatives
    grad[0][1] = (u_yp[0] - u_ym[0]) * invDy;  // du/dy
    grad[1][1] = (u_yp[1] - u_ym[1]) * invDy;  // dv/dy
    grad[2][1] = (u_yp[2] - u_ym[2]) * invDy;  // dw/dy

    // d/dz derivatives
    grad[0][2] = (u_zp[0] - u_zm[0]) * invDz;  // du/dz
    grad[1][2] = (u_zp[1] - u_zm[1]) * invDz;  // dv/dz
    grad[2][2] = (u_zp[2] - u_zm[2]) * invDz;  // dw/dz

    return grad;
}

double IGRSolver::trace(const GradientTensor& A) {
    // tr(A) = A[0][0] + A[1][1] + A[2][2] = div(u)
    return A[0][0] + A[1][1] + A[2][2];
}

double IGRSolver::traceSquared(const GradientTensor& A) {
    // tr(A²) = sum of A_ij * A_ji
    // For the velocity gradient, this equals sum of all A_ij²
    // when we interpret it as tr((∇u)·(∇u)^T)
    //
    // Actually from the paper: tr(∇u)² means the Frobenius norm squared
    // = sum over all i,j of (du_i/dx_j)²

    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sum += A[i][j] * A[i][j];
        }
    }
    return sum;
}

} // namespace SemiImplicitFV
