#include "IGR.hpp"
#include "HaloExchange.hpp"
#include "ImmersedBoundary.hpp"
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

double IGRSolver::computeIGRRhs(const SimulationConfig& config, const GradientTensor& gradU, double alpha) const {
    // Compute: α[tr(∇u)² + tr²(∇u)]

    double trSq{0.0};
    for (int i = 0; i < config.dim; ++i) {
        for (int j = 0; j < config.dim; ++j) {
            trSq += gradU[i][j] * gradU[j][i];
        }
    }

    double trSquared{0.0};
    for (int i = 0; i < config.dim; ++i) {
        trSquared += gradU[i][i];  // Add diagonal components to get trace
    }
    trSquared *= trSquared;  // Square the trace to get (tr(∇u))²

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
                    if (ibm_ && ibm_->isSolid(idx)) continue;

                    double rhs = computeIGRRhs(config, gradU[idx], alpha);

                    // Diagonal coefficient: 1/ρ_i + α·Σ(1/ρ_neighbor)/dx²
                    double diag = 1.0 / state.rho[idx];

                    // Off-diagonal sum: α·Σ(σ_neighbor/ρ_neighbor)/dx²
                    double offdiag = 0.0;

                    // X-direction (face-averaged densities)
                    std::size_t ixl = mesh.index(i - 1, j, k);
                    std::size_t ixr = mesh.index(i + 1, j, k);
                    double rho_i = state.rho[idx];
                    double rho_fxl = 0.5 * (rho_i + state.rho[ixl]);
                    double rho_fxr = 0.5 * (rho_i + state.rho[ixr]);
                    double dx2 = 1.0 / (mesh.dx(i) * mesh.dx(i));
                    diag += alpha * dx2 * (1.0 / rho_fxl + 1.0 / rho_fxr);
                    offdiag += alpha * dx2 * (state.sigma[ixl] / rho_fxl + state.sigma[ixr] / rho_fxr);

                    // Y-direction (face-averaged densities)
                    if (config.dim >= 2) {
                        std::size_t iyl = mesh.index(i, j - 1, k);
                        std::size_t iyr = mesh.index(i, j + 1, k);
                        double rho_fyl = 0.5 * (rho_i + state.rho[iyl]);
                        double rho_fyr = 0.5 * (rho_i + state.rho[iyr]);
                        double dy2 = 1.0 / (mesh.dy(j) * mesh.dy(j));
                        diag += alpha * dy2 * (1.0 / rho_fyl + 1.0 / rho_fyr);
                        offdiag += alpha * dy2 * (state.sigma[iyl] / rho_fyl + state.sigma[iyr] / rho_fyr);
                    }

                    // Z-direction (face-averaged densities)
                    if (config.dim >= 3) {
                        std::size_t izl = mesh.index(i, j, k - 1);
                        std::size_t izr = mesh.index(i, j, k + 1);
                        double rho_fzl = 0.5 * (rho_i + state.rho[izl]);
                        double rho_fzr = 0.5 * (rho_i + state.rho[izr]);
                        double dz2 = 1.0 / (mesh.dz(k) * mesh.dz(k));
                        diag += alpha * dz2 * (1.0 / rho_fzl + 1.0 / rho_fzr);
                        offdiag += alpha * dz2 * (state.sigma[izl] / rho_fzl + state.sigma[izr] / rho_fzr);
                    }

                    state.sigma[idx] = (rhs + offdiag) / diag;
                }
            }
        }
        mesh.fillScalarGhosts(state.sigma);
        if (ibm_) ibm_->applyScalarGhostCells(mesh, state.sigma);
    }
}

void IGRSolver::solveEntropicPressure(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state,
        std::vector<GradientTensor> gradU,
        HaloExchange& halo) {
    int maxIters = config.step == 0 ? params_.IGRWarmStartIters : params_.IGRIters;
    double alpha = params_.alphaCoeff * mesh.dx(0) * mesh.dx(0);

    for (int iter = 0; iter < maxIters; ++iter) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    if (ibm_ && ibm_->isSolid(idx)) continue;

                    double rhs = computeIGRRhs(config, gradU[idx], alpha);

                    double diag = 1.0 / state.rho[idx];
                    double offdiag = 0.0;

                    std::size_t ixl = mesh.index(i - 1, j, k);
                    std::size_t ixr = mesh.index(i + 1, j, k);
                    double rho_i = state.rho[idx];
                    double rho_fxl = 0.5 * (rho_i + state.rho[ixl]);
                    double rho_fxr = 0.5 * (rho_i + state.rho[ixr]);
                    double dx2 = 1.0 / (mesh.dx(i) * mesh.dx(i));
                    diag += alpha * dx2 * (1.0 / rho_fxl + 1.0 / rho_fxr);
                    offdiag += alpha * dx2 * (state.sigma[ixl] / rho_fxl + state.sigma[ixr] / rho_fxr);

                    if (config.dim >= 2) {
                        std::size_t iyl = mesh.index(i, j - 1, k);
                        std::size_t iyr = mesh.index(i, j + 1, k);
                        double rho_fyl = 0.5 * (rho_i + state.rho[iyl]);
                        double rho_fyr = 0.5 * (rho_i + state.rho[iyr]);
                        double dy2 = 1.0 / (mesh.dy(j) * mesh.dy(j));
                        diag += alpha * dy2 * (1.0 / rho_fyl + 1.0 / rho_fyr);
                        offdiag += alpha * dy2 * (state.sigma[iyl] / rho_fyl + state.sigma[iyr] / rho_fyr);
                    }

                    if (config.dim >= 3) {
                        std::size_t izl = mesh.index(i, j, k - 1);
                        std::size_t izr = mesh.index(i, j, k + 1);
                        double rho_fzl = 0.5 * (rho_i + state.rho[izl]);
                        double rho_fzr = 0.5 * (rho_i + state.rho[izr]);
                        double dz2 = 1.0 / (mesh.dz(k) * mesh.dz(k));
                        diag += alpha * dz2 * (1.0 / rho_fzl + 1.0 / rho_fzr);
                        offdiag += alpha * dz2 * (state.sigma[izl] / rho_fzl + state.sigma[izr] / rho_fzr);
                    }

                    state.sigma[idx] = (rhs + offdiag) / diag;
                }
            }
        }
        mesh.fillScalarGhosts(state.sigma, halo);
        if (ibm_) ibm_->applyScalarGhostCells(mesh, state.sigma);
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
    GradientTensor grad{};

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

} // namespace SemiImplicitFV

