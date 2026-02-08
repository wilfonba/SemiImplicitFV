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
    // Solve: rhs = Σ/ρ - α∇·(∇Σ/ρ)
    // Jacobi iteration with warm start
    int maxIters = config.step == 0 ? params_.IGRWarmStartIters : params_.IGRIters;
    double alpha = params_.alphaCoeff * mesh.dx(0) * mesh.dx(0); // Assuming uniform grid for simplicity
    std::size_t idx; // for indexing current, left, right cells
    std::size_t idxl, idxr;
    std::size_t idyl, idyr;
    std::size_t idzl, idzr;
    double rho_lx, rho_rx, rho_ly, rho_ry, rho_lz, rho_rz;
    double dx2, dy2, dz2;
    double fd_coeff;

    for (int iter = 0; iter < maxIters; ++iter) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {

                    idx = mesh.index(i,j,k);
                    idxl = mesh.index(i - 1, j, k);
                    idxr = mesh.index(i + 1, j, k);

                    rho_lx = state.rho[idxl];
                    rho_rx = state.rho[idxr];
                    dx2 = 1.0 / (mesh.dx(i) * mesh.dx(i));
                    fd_coeff = state.rho[idx];
                    fd_coeff += 1.0 / fd_coeff + alpha *
                        (dx2 * (1.0 / rho_lx + 1.0 / rho_rx));

                    if (config.dim >= 2) {
                        idxl = mesh.index(i, j - 1, k);
                        idxr = mesh.index(i, j + 1, k);
                        rho_ly = state.rho[idxl];
                        rho_ry = state.rho[idxr];
                        dy2 = 1.0 / (mesh.dy(j) * mesh.dy(j));
                        fd_coeff += alpha * dy2 * (1.0 / rho_ly + 1.0 / rho_ry);

                        if (config.dim >= 3) {
                            idxl = mesh.index(i, j, k - 1);
                            idxr = mesh.index(i, j, k + 1);
                            rho_lz = state.rho[idxl];
                            rho_rz = state.rho[idxr];
                            dz2 = 1.0 / (mesh.dz(k) * mesh.dz(k));
                            fd_coeff += alpha * dz2 * (1.0 / rho_lz + 1.0 / rho_rz);
                        }
                    }

                    if (config.dim == 1) {
                        idxl = mesh.index(i - 1, j, k);
                        idxr = mesh.index(i + 1, j, k);

                        state.sigma[idx] = (alpha / fd_coeff) *
                            (dx2 * (state.sigma[idxl] / rho_lx + state.sigma[idxr] / rho_rx) +
                            computeIGRRhs(gradU[idx], alpha) / fd_coeff);
                    }
                    else if (config.dim == 2) {
                        idyl = mesh.index(i, j + 1, k);
                        idyr = mesh.index(i, j - 1, k);
                        idxl = mesh.index(i - 1, j, k);
                        idxr = mesh.index(i + 1, j, k);

                        state.sigma[idx] = (alpha / fd_coeff) *
                            (dx2 * (state.sigma[idxl] / rho_lx + state.sigma[idxr] / rho_rx) +
                             dy2 * (state.sigma[idyl] / rho_ly + state.sigma[idyr] / rho_ry) +
                             computeIGRRhs(gradU[idx], alpha) / fd_coeff);
                    }
                    else if (config.dim == 3) {
                        idzl = mesh.index(i, j, k + 1);
                        idzr = mesh.index(i, j, k - 1);
                        idyl = mesh.index(i, j + 1, k);
                        idyr = mesh.index(i, j - 1, k);
                        idxl = mesh.index(i - 1, j, k);
                        idxr = mesh.index(i + 1, j, k);

                        state.sigma[idx] = (alpha / fd_coeff) *
                            (dx2 * (state.sigma[idxl] / rho_lx + state.sigma[idxr] / rho_rx) +
                             dy2 * (state.sigma[idyl] / rho_ly + state.sigma[idyr] / rho_ry) +
                             dz2 * (state.sigma[idzl] / rho_lz + state.sigma[idzr] / rho_rz) +
                             computeIGRRhs(gradU[idx], alpha) / fd_coeff);
                    }
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

