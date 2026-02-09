#include "IGR.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

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

void IGRSolver::solveEntropicPressure(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state,
        std::vector<GradientTensor> gradU) {
    // Solve: σ/ρ - α∇·((1/ρ)∇σ) = rhs
    // Gauss-Seidel iteration with warm start
    int maxIters = config.step == 0 ? params_.IGRWarmStartIters : params_.IGRIters;
    double alpha = params_.alphaCoeff * mesh.dx(0) * mesh.dx(0);

    for (int iter = 0; iter < maxIters; ++iter) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double rhs = computeIGRRhs(gradU[idx], alpha);

                    // Diagonal coefficient: 1/ρ_i + α·Σ(1/ρ_neighbor)/dx²
                    double diag = 1.0 / state.rho[idx];

                    // Off-diagonal sum: α·Σ(σ_neighbor/ρ_neighbor)/dx²
                    double offdiag = 0.0;

                    // X-direction
                    std::size_t ixl = mesh.index(i - 1, j, k);
                    std::size_t ixr = mesh.index(i + 1, j, k);
                    double rho_xl = state.rho[ixl];
                    double rho_xr = state.rho[ixr];
                    double dx2 = 1.0 / (mesh.dx(i) * mesh.dx(i));
                    diag += alpha * dx2 * (1.0 / rho_xl + 1.0 / rho_xr);
                    offdiag += alpha * dx2 * (state.sigma[ixl] / rho_xl + state.sigma[ixr] / rho_xr);

                    // Y-direction
                    if (config.dim >= 2) {
                        std::size_t iyl = mesh.index(i, j - 1, k);
                        std::size_t iyr = mesh.index(i, j + 1, k);
                        double rho_yl = state.rho[iyl];
                        double rho_yr = state.rho[iyr];
                        double dy2 = 1.0 / (mesh.dy(j) * mesh.dy(j));
                        diag += alpha * dy2 * (1.0 / rho_yl + 1.0 / rho_yr);
                        offdiag += alpha * dy2 * (state.sigma[iyl] / rho_yl + state.sigma[iyr] / rho_yr);
                    }

                    // Z-direction
                    if (config.dim >= 3) {
                        std::size_t izl = mesh.index(i, j, k - 1);
                        std::size_t izr = mesh.index(i, j, k + 1);
                        double rho_zl = state.rho[izl];
                        double rho_zr = state.rho[izr];
                        double dz2 = 1.0 / (mesh.dz(k) * mesh.dz(k));
                        diag += alpha * dz2 * (1.0 / rho_zl + 1.0 / rho_zr);
                        offdiag += alpha * dz2 * (state.sigma[izl] / rho_zl + state.sigma[izr] / rho_zr);
                    }

                    state.sigma[idx] = (rhs + offdiag) / diag;
                }
            }
        }
        mesh.fillScalarGhosts(state.sigma);
    }
}

GradientTensor IGRSolver::computeVelocityGradient(
    const std::array<double, 3>& u_xm,
    const std::array<double, 3>& u_xp,
    const std::array<double, 3>& u_ym,
    const std::array<double, 3>& u_yp,
    const std::array<double, 3>& u_zm,
    const std::array<double, 3>& u_zp,
    double dx, double dy, double dz,
    int dim
) {
    GradientTensor grad;

    // grad[i][j] = du_i / dx_j
    // Using central differences

    double invDx = 0.5 / dx;
    double invDy = 0.5 / dy;
    double invDz = 0.5 / dz;

    // d/dx derivatives
    grad[0][0] = (u_xp[0] - u_xm[0]) * invDx;  // du/dx

    // d/dy derivatives
    if (dim >= 2) {
        grad[1][0] = (u_xp[1] - u_xm[1]) * invDx;  // dv/dx

        grad[0][1] = (u_yp[0] - u_ym[0]) * invDy;  // du/dy
        grad[1][1] = (u_yp[1] - u_ym[1]) * invDy;  // dv/dy

        if (dim >= 3) {
            grad[2][1] = (u_yp[2] - u_ym[2]) * invDy;  // dw/dy
            grad[2][0] = (u_xp[2] - u_xm[2]) * invDx;  // dw/dx

            grad[0][2] = (u_zp[0] - u_zm[0]) * invDz;  // du/dz
            grad[1][2] = (u_zp[1] - u_zm[1]) * invDz;  // dv/dz
            grad[2][2] = (u_zp[2] - u_zm[2]) * invDz;  // dw/dz
        }
    }

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

