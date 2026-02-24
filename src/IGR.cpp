#include "IGR.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "HaloExchange.hpp"
#include <string.h>
#include <math.h>

double igr_compute_alpha(const IGRParams* params, double dx) {
    return params->alphaCoeff * dx * dx;
}

double igr_compute_rhs(const SimulationConfig* config,
                       const GradientTensor gradU,
                       double alpha)
{
    /* Compute: alpha[tr(nabla u)^2 + tr^2(nabla u)] */

    double trSq = 0.0;
    for (int i = 0; i < config->dim; ++i) {
        for (int j = 0; j < config->dim; ++j) {
            trSq += gradU[i][j] * gradU[j][i];
        }
    }

    double trSquared = 0.0;
    for (int i = 0; i < config->dim; ++i) {
        trSquared += gradU[i][i];  /* Add diagonal components to get trace */
    }
    trSquared *= trSquared;  /* Square the trace to get (tr(nabla u))^2 */

    return alpha * (trSq + trSquared);
}

void igr_solve_entropic_pressure(const SimulationConfig* config,
                                 const IGRParams* params,
                                 const struct RectilinearMesh* mesh,
                                 struct SolutionState* state,
                                 const GradientTensor* gradU)
{
    /* Gauss-Seidel iteration with warm start */
    int maxIters = config->step == 0 ? params->IGRWarmStartIters : params->IGRIters;
    double alpha = params->alphaCoeff * mesh_dx(mesh, 0) * mesh_dx(mesh, 0);

    for (int iter = 0; iter < maxIters; ++iter) {
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);

                    double rhs = igr_compute_rhs(config, gradU[idx], alpha);

                    /* Diagonal coefficient: 1/rho_i + alpha * Sum(1/rho_neighbor)/dx^2 */
                    double diag = 1.0 / state->rho[idx];

                    /* Off-diagonal sum: alpha * Sum(sigma_neighbor/rho_neighbor)/dx^2 */
                    double offdiag = 0.0;

                    /* X-direction (face-averaged densities) */
                    size_t ixl = mesh_index(mesh, i - 1, j, k);
                    size_t ixr = mesh_index(mesh, i + 1, j, k);
                    double rho_i = state->rho[idx];
                    double rho_fxl = 0.5 * (rho_i + state->rho[ixl]);
                    double rho_fxr = 0.5 * (rho_i + state->rho[ixr]);
                    double dx2 = 1.0 / (mesh_dx(mesh, i) * mesh_dx(mesh, i));
                    diag += alpha * dx2 * (1.0 / rho_fxl + 1.0 / rho_fxr);
                    offdiag += alpha * dx2 * (state->sigma[ixl] / rho_fxl + state->sigma[ixr] / rho_fxr);

                    /* Y-direction (face-averaged densities) */
                    if (config->dim >= 2) {
                        size_t iyl = mesh_index(mesh, i, j - 1, k);
                        size_t iyr = mesh_index(mesh, i, j + 1, k);
                        double rho_fyl = 0.5 * (rho_i + state->rho[iyl]);
                        double rho_fyr = 0.5 * (rho_i + state->rho[iyr]);
                        double dy2 = 1.0 / (mesh_dy(mesh, j) * mesh_dy(mesh, j));
                        diag += alpha * dy2 * (1.0 / rho_fyl + 1.0 / rho_fyr);
                        offdiag += alpha * dy2 * (state->sigma[iyl] / rho_fyl + state->sigma[iyr] / rho_fyr);
                    }

                    /* Z-direction (face-averaged densities) */
                    if (config->dim >= 3) {
                        size_t izl = mesh_index(mesh, i, j, k - 1);
                        size_t izr = mesh_index(mesh, i, j, k + 1);
                        double rho_fzl = 0.5 * (rho_i + state->rho[izl]);
                        double rho_fzr = 0.5 * (rho_i + state->rho[izr]);
                        double dz2 = 1.0 / (mesh_dz(mesh, k) * mesh_dz(mesh, k));
                        diag += alpha * dz2 * (1.0 / rho_fzl + 1.0 / rho_fzr);
                        offdiag += alpha * dz2 * (state->sigma[izl] / rho_fzl + state->sigma[izr] / rho_fzr);
                    }

                    state->sigma[idx] = (rhs + offdiag) / diag;
                }
            }
        }
        mesh_fill_scalar_ghosts(mesh, state->sigma);
    }
}

void igr_solve_entropic_pressure_mpi(const SimulationConfig* config,
                                     const IGRParams* params,
                                     const struct RectilinearMesh* mesh,
                                     struct SolutionState* state,
                                     const GradientTensor* gradU,
                                     struct HaloExchange* halo)
{
    int maxIters = config->step == 0 ? params->IGRWarmStartIters : params->IGRIters;
    double alpha = params->alphaCoeff * mesh_dx(mesh, 0) * mesh_dx(mesh, 0);

    for (int iter = 0; iter < maxIters; ++iter) {
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);

                    double rhs = igr_compute_rhs(config, gradU[idx], alpha);

                    double diag = 1.0 / state->rho[idx];
                    double offdiag = 0.0;

                    size_t ixl = mesh_index(mesh, i - 1, j, k);
                    size_t ixr = mesh_index(mesh, i + 1, j, k);
                    double rho_i = state->rho[idx];
                    double rho_fxl = 0.5 * (rho_i + state->rho[ixl]);
                    double rho_fxr = 0.5 * (rho_i + state->rho[ixr]);
                    double dx2 = 1.0 / (mesh_dx(mesh, i) * mesh_dx(mesh, i));
                    diag += alpha * dx2 * (1.0 / rho_fxl + 1.0 / rho_fxr);
                    offdiag += alpha * dx2 * (state->sigma[ixl] / rho_fxl + state->sigma[ixr] / rho_fxr);

                    if (config->dim >= 2) {
                        size_t iyl = mesh_index(mesh, i, j - 1, k);
                        size_t iyr = mesh_index(mesh, i, j + 1, k);
                        double rho_fyl = 0.5 * (rho_i + state->rho[iyl]);
                        double rho_fyr = 0.5 * (rho_i + state->rho[iyr]);
                        double dy2 = 1.0 / (mesh_dy(mesh, j) * mesh_dy(mesh, j));
                        diag += alpha * dy2 * (1.0 / rho_fyl + 1.0 / rho_fyr);
                        offdiag += alpha * dy2 * (state->sigma[iyl] / rho_fyl + state->sigma[iyr] / rho_fyr);
                    }

                    if (config->dim >= 3) {
                        size_t izl = mesh_index(mesh, i, j, k - 1);
                        size_t izr = mesh_index(mesh, i, j, k + 1);
                        double rho_fzl = 0.5 * (rho_i + state->rho[izl]);
                        double rho_fzr = 0.5 * (rho_i + state->rho[izr]);
                        double dz2 = 1.0 / (mesh_dz(mesh, k) * mesh_dz(mesh, k));
                        diag += alpha * dz2 * (1.0 / rho_fzl + 1.0 / rho_fzr);
                        offdiag += alpha * dz2 * (state->sigma[izl] / rho_fzl + state->sigma[izr] / rho_fzr);
                    }

                    state->sigma[idx] = (rhs + offdiag) / diag;
                }
            }
        }
        mesh_fill_scalar_ghosts_mpi(mesh, state->sigma, halo);
    }
}

void igr_compute_velocity_gradient(
    const double u_xm[3],
    const double u_xp[3],
    const double u_ym[3],
    const double u_yp[3],
    const double u_zm[3],
    const double u_zp[3],
    double dx, double dy, double dz,
    int dim,
    GradientTensor grad)
{
    memset(grad, 0, sizeof(GradientTensor));

    /* grad[i][j] = du_i / dx_j */
    /* Using central differences */

    double invDx = 0.5 / dx;
    double invDy = 0.5 / dy;
    double invDz = 0.5 / dz;

    /* d/dx derivatives */
    grad[0][0] = (u_xp[0] - u_xm[0]) * invDx;  /* du/dx */

    /* d/dy derivatives */
    if (dim >= 2) {
        grad[1][0] = (u_xp[1] - u_xm[1]) * invDx;  /* dv/dx */

        grad[0][1] = (u_yp[0] - u_ym[0]) * invDy;  /* du/dy */
        grad[1][1] = (u_yp[1] - u_ym[1]) * invDy;  /* dv/dy */

        if (dim >= 3) {
            grad[2][1] = (u_yp[2] - u_ym[2]) * invDy;  /* dw/dy */
            grad[2][0] = (u_xp[2] - u_xm[2]) * invDx;  /* dw/dx */

            grad[0][2] = (u_zp[0] - u_zm[0]) * invDz;  /* du/dz */
            grad[1][2] = (u_zp[1] - u_zm[1]) * invDz;  /* dv/dz */
            grad[2][2] = (u_zp[2] - u_zm[2]) * invDz;  /* dw/dz */
        }
    }
}
