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

/* ---------------------------------------------------------------------------
   GPU variant
   ---------------------------------------------------------------------------

   gradU is stored as a flat array of 9*totalCells doubles, accessed as
   gradU[9*idx + 3*i + j] for the (i,j) component of cell idx.  The host-side
   API uses GradientTensor (double[3][3]) which maps to the same layout in
   memory, so this representation is compatible with the host code path. */

void igr_compute_velocity_gradients_device(
    const SimulationConfig* config,
    const RectilinearMesh* mesh,
    const SolutionState* state,
    double* gradU)
{
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = config->dim;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* velU = state->velU;
    double* velV = state->velV;
    double* velW = state->velW;

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double dxi = xExt[i + ngx + 1] - xExt[i + ngx];
                double invDx = 0.5 / dxi;

                size_t xm = (size_t)((i - 1 + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t xp = (size_t)((i + 1 + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double uxm0 = velU[xm];
                double uxm1 = (dim >= 2) ? velV[xm] : 0.0;
                double uxm2 = (dim >= 3) ? velW[xm] : 0.0;
                double uxp0 = velU[xp];
                double uxp1 = (dim >= 2) ? velV[xp] : 0.0;
                double uxp2 = (dim >= 3) ? velW[xp] : 0.0;

                double uym0 = 0.0, uym1 = 0.0, uym2 = 0.0;
                double uyp0 = 0.0, uyp1 = 0.0, uyp2 = 0.0;
                double invDy = 0.0;
                if (dim >= 2) {
                    double dyj = yExt[j + ngy + 1] - yExt[j + ngy];
                    invDy = 0.5 / dyj;
                    size_t ym = (size_t)((i + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (k + ngz)));
                    size_t yp = (size_t)((i + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (k + ngz)));
                    uym0 = velU[ym]; uym1 = velV[ym];
                    uym2 = (dim >= 3) ? velW[ym] : 0.0;
                    uyp0 = velU[yp]; uyp1 = velV[yp];
                    uyp2 = (dim >= 3) ? velW[yp] : 0.0;
                }

                double uzm0 = 0.0, uzm1 = 0.0, uzm2 = 0.0;
                double uzp0 = 0.0, uzp1 = 0.0, uzp2 = 0.0;
                double invDz = 0.0;
                if (dim >= 3) {
                    double dzk = zExt[k + ngz + 1] - zExt[k + ngz];
                    invDz = 0.5 / dzk;
                    size_t zm = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k - 1 + ngz)));
                    size_t zp = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + 1 + ngz)));
                    uzm0 = velU[zm]; uzm1 = velV[zm]; uzm2 = velW[zm];
                    uzp0 = velU[zp]; uzp1 = velV[zp]; uzp2 = velW[zp];
                }

                /* Match host igr_compute_velocity_gradient conventions. */
                double g[9];
                for (int c = 0; c < 9; ++c) g[c] = 0.0;

                g[0*3 + 0] = (uxp0 - uxm0) * invDx;  /* du/dx */
                if (dim >= 2) {
                    g[1*3 + 0] = (uxp1 - uxm1) * invDx;  /* dv/dx */
                    g[0*3 + 1] = (uyp0 - uym0) * invDy;  /* du/dy */
                    g[1*3 + 1] = (uyp1 - uym1) * invDy;  /* dv/dy */
                    if (dim >= 3) {
                        g[2*3 + 1] = (uyp2 - uym2) * invDy;  /* dw/dy */
                        g[2*3 + 0] = (uxp2 - uxm2) * invDx;  /* dw/dx */
                        g[0*3 + 2] = (uzp0 - uzm0) * invDz;  /* du/dz */
                        g[1*3 + 2] = (uzp1 - uzm1) * invDz;  /* dv/dz */
                        g[2*3 + 2] = (uzp2 - uzm2) * invDz;  /* dw/dz */
                    }
                }

                for (int c = 0; c < 9; ++c) gradU[9 * idx + c] = g[c];
            }
        }
    }
}

/* Jacobi iteration of the IGR elliptic pressure equation.  Uses state->aux
 * as the ping-pong buffer.  Iterations alternate between sigma and aux as
 * source/destination; the final state leaves sigma holding the answer.  The
 * initial warm start step (step == 0) uses params->IGRWarmStartIters,
 * otherwise params->IGRIters. */
void igr_solve_entropic_pressure_mpi_device(const SimulationConfig* config,
                                            const IGRParams* params,
                                            const RectilinearMesh* mesh,
                                            SolutionState* state,
                                            const GradientTensor* gradU,
                                            struct HaloExchange* halo)
{
    int maxIters = config->step == 0 ? params->IGRWarmStartIters : params->IGRIters;
    double alpha = params->alphaCoeff * mesh_dx(mesh, 0) * mesh_dx(mesh, 0);

    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = config->dim;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* rho   = state->rho;
    double* sigma = state->sigma;
    double* aux   = state->aux;
    const double* gU = (const double*)gradU;

    for (int iter = 0; iter < maxIters; ++iter) {
        /* Even iter reads sigma, writes aux; odd iter reads aux, writes sigma. */
        double* src = (iter % 2 == 0) ? sigma : aux;
        double* dst = (iter % 2 == 0) ? aux   : sigma;

        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                    /* RHS = alpha * [tr(grad^2) + tr(grad)^2] */
                    double trSq = 0.0;
                    for (int a = 0; a < dim; ++a)
                        for (int b = 0; b < dim; ++b)
                            trSq += gU[9 * idx + a * 3 + b] * gU[9 * idx + b * 3 + a];
                    double tr = 0.0;
                    for (int a = 0; a < dim; ++a) tr += gU[9 * idx + a * 3 + a];
                    double rhs = alpha * (trSq + tr * tr);

                    double rho_i = rho[idx];
                    double diag = 1.0 / rho_i;
                    double off = 0.0;

                    /* X direction */
                    double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                    double dx2 = 1.0 / (dx * dx);
                    size_t ixl = (size_t)((i - 1 + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    size_t ixr = (size_t)((i + 1 + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    double rho_fxl = 0.5 * (rho_i + rho[ixl]);
                    double rho_fxr = 0.5 * (rho_i + rho[ixr]);
                    diag += alpha * dx2 * (1.0 / rho_fxl + 1.0 / rho_fxr);
                    off  += alpha * dx2 * (src[ixl] / rho_fxl + src[ixr] / rho_fxr);

                    /* Y direction */
                    if (dim >= 2) {
                        double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                        double dy2 = 1.0 / (dy * dy);
                        size_t iyl = (size_t)((i + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (k + ngz)));
                        size_t iyr = (size_t)((i + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (k + ngz)));
                        double rho_fyl = 0.5 * (rho_i + rho[iyl]);
                        double rho_fyr = 0.5 * (rho_i + rho[iyr]);
                        diag += alpha * dy2 * (1.0 / rho_fyl + 1.0 / rho_fyr);
                        off  += alpha * dy2 * (src[iyl] / rho_fyl + src[iyr] / rho_fyr);
                    }

                    /* Z direction */
                    if (dim >= 3) {
                        double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                        double dz2 = 1.0 / (dz * dz);
                        size_t izl = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k - 1 + ngz)));
                        size_t izr = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + 1 + ngz)));
                        double rho_fzl = 0.5 * (rho_i + rho[izl]);
                        double rho_fzr = 0.5 * (rho_i + rho[izr]);
                        diag += alpha * dz2 * (1.0 / rho_fzl + 1.0 / rho_fzr);
                        off  += alpha * dz2 * (src[izl] / rho_fzl + src[izr] / rho_fzr);
                    }

                    dst[idx] = (rhs + off) / diag;
                }
            }
        }

        mesh_fill_scalar_ghosts_mpi_device(mesh, dst, halo);
    }

    /* After an odd number of iterations, the answer lives in aux; copy it
     * back into sigma so downstream code reads the right field. */
    if (maxIters > 0 && (maxIters % 2 == 1)) {
        size_t n = state->totalCells;
        #pragma omp target teams distribute parallel for
        for (size_t ii = 0; ii < n; ++ii) sigma[ii] = aux[ii];
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
