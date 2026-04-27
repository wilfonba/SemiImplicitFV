#include "ViscousFlux.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"

/* Compute face viscosity from the two neighboring cells.
   Arithmetic mixture rule: mu_cell = sum_k(alpha_k * mu_k),
   then face value = average of left and right cells. */
static inline double compute_face_mu(
    size_t idxL, size_t idxR,
    int perPhase, double muConst,
    const SolutionState* state,
    const double* phaseMu, int nPhases)
{
    if (!perPhase) return muConst;
    size_t tc = state->totalCells;
    double muL = 0.0, muR = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        muL += state->alpha[ph * tc + idxL] * phaseMu[ph];
        muR += state->alpha[ph * tc + idxR] * phaseMu[ph];
    }
    return 0.5 * (muL + muR);
}

void add_viscous_fluxes(
    const SimulationConfig* config,
    const RectilinearMesh* mesh,
    const SolutionState* state,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE)
{
    int dim = config->dim;
    const ViscousParams* vp = &config->viscousParams;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    int perPhase = (vp->nPhaseMu > 0);

    /* --- X-direction faces --- */
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i <= mesh->nx; ++i) {
                size_t idxL = mesh_index(mesh, i - 1, j, k);
                size_t idxR = mesh_index(mesh, i, j, k);

                double dxi = 0.5 * (mesh_dx(mesh, i - 1) + mesh_dx(mesh, i));

                double dudx = (state->velU[idxR] - state->velU[idxL]) / dxi;
                double dvdx = 0.0;
                double dwdx = 0.0;
                if (dim >= 2) dvdx = (state->velV[idxR] - state->velV[idxL]) / dxi;
                if (dim >= 3) dwdx = (state->velW[idxR] - state->velW[idxL]) / dxi;

                double dudy = 0.0, dvdy = 0.0;
                if (dim >= 2) {
                    double dyj = mesh_dy(mesh, j);
                    size_t Ljm = mesh_index(mesh, i - 1, j - 1, k);
                    size_t Ljp = mesh_index(mesh, i - 1, j + 1, k);
                    size_t Rjm = mesh_index(mesh, i, j - 1, k);
                    size_t Rjp = mesh_index(mesh, i, j + 1, k);

                    double dudy_L = (state->velU[Ljp] - state->velU[Ljm]) / (2.0 * dyj);
                    double dudy_R = (state->velU[Rjp] - state->velU[Rjm]) / (2.0 * dyj);
                    dudy = 0.5 * (dudy_L + dudy_R);

                    double dvdy_L = (state->velV[Ljp] - state->velV[Ljm]) / (2.0 * dyj);
                    double dvdy_R = (state->velV[Rjp] - state->velV[Rjm]) / (2.0 * dyj);
                    dvdy = 0.5 * (dvdy_L + dvdy_R);
                }

                double dudz = 0.0, dwdz = 0.0;
                if (dim >= 3) {
                    double dzk = mesh_dz(mesh, k);
                    size_t Lkm = mesh_index(mesh, i - 1, j, k - 1);
                    size_t Lkp = mesh_index(mesh, i - 1, j, k + 1);
                    size_t Rkm = mesh_index(mesh, i, j, k - 1);
                    size_t Rkp = mesh_index(mesh, i, j, k + 1);

                    double dudz_L = (state->velU[Lkp] - state->velU[Lkm]) / (2.0 * dzk);
                    double dudz_R = (state->velU[Rkp] - state->velU[Rkm]) / (2.0 * dzk);
                    dudz = 0.5 * (dudz_L + dudz_R);

                    double dwdz_L = (state->velW[Lkp] - state->velW[Lkm]) / (2.0 * dzk);
                    double dwdz_R = (state->velW[Rkp] - state->velW[Rkm]) / (2.0 * dzk);
                    dwdz = 0.5 * (dwdz_L + dwdz_R);
                }

                double divU = dudx + dvdy + dwdz;

                double muF = compute_face_mu(idxL, idxR, perPhase, vp->mu, state, vp->phaseMu, mp->nPhases);
                double tau_xx = muF * (2.0 * dudx - (2.0 / 3.0) * divU);
                double tau_xy = muF * (dvdx + dudy);
                double tau_xz = muF * (dwdx + dudz);

                double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                double vFace = 0.0;
                double wFace = 0.0;
                if (dim >= 2) vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                if (dim >= 3) wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                double work = tau_xx * uFace + tau_xy * vFace + tau_xz * wFace;

                double area = mesh_faceAreaX(mesh, j, k);

                if (i >= 1) {
                    double coeff = area / mesh_cell_volume(mesh, i - 1, j, k);
                    rhsRhoU[idxL] += coeff * tau_xx;
                    if (dim >= 2) rhsRhoV[idxL] += coeff * tau_xy;
                    if (dim >= 3) rhsRhoW[idxL] += coeff * tau_xz;
                    rhsRhoE[idxL] += coeff * work;
                }

                if (i < mesh->nx) {
                    double coeff = area / mesh_cell_volume(mesh, i, j, k);
                    rhsRhoU[idxR] -= coeff * tau_xx;
                    if (dim >= 2) rhsRhoV[idxR] -= coeff * tau_xy;
                    if (dim >= 3) rhsRhoW[idxR] -= coeff * tau_xz;
                    rhsRhoE[idxR] -= coeff * work;
                }
            }
        }
    }

    /* --- Y-direction faces --- */
    if (dim >= 2) {
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j <= mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idxL = mesh_index(mesh, i, j - 1, k);
                    size_t idxR = mesh_index(mesh, i, j, k);

                    double dyj = 0.5 * (mesh_dy(mesh, j - 1) + mesh_dy(mesh, j));

                    double dudy = (state->velU[idxR] - state->velU[idxL]) / dyj;
                    double dvdy = (state->velV[idxR] - state->velV[idxL]) / dyj;
                    double dwdy = 0.0;
                    if (dim >= 3) dwdy = (state->velW[idxR] - state->velW[idxL]) / dyj;

                    double dxi = mesh_dx(mesh, i);
                    size_t Lim = mesh_index(mesh, i - 1, j - 1, k);
                    size_t Lip = mesh_index(mesh, i + 1, j - 1, k);
                    size_t Rim = mesh_index(mesh, i - 1, j, k);
                    size_t Rip = mesh_index(mesh, i + 1, j, k);

                    double dudx_L = (state->velU[Lip] - state->velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (state->velU[Rip] - state->velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);

                    double dvdx_L = (state->velV[Lip] - state->velV[Lim]) / (2.0 * dxi);
                    double dvdx_R = (state->velV[Rip] - state->velV[Rim]) / (2.0 * dxi);
                    double dvdx = 0.5 * (dvdx_L + dvdx_R);

                    double dvdz = 0.0, dwdz = 0.0;
                    if (dim >= 3) {
                        double dzk = mesh_dz(mesh, k);
                        size_t Lkm = mesh_index(mesh, i, j - 1, k - 1);
                        size_t Lkp = mesh_index(mesh, i, j - 1, k + 1);
                        size_t Rkm = mesh_index(mesh, i, j, k - 1);
                        size_t Rkp = mesh_index(mesh, i, j, k + 1);

                        double dvdz_L = (state->velV[Lkp] - state->velV[Lkm]) / (2.0 * dzk);
                        double dvdz_R = (state->velV[Rkp] - state->velV[Rkm]) / (2.0 * dzk);
                        dvdz = 0.5 * (dvdz_L + dvdz_R);

                        double dwdz_L = (state->velW[Lkp] - state->velW[Lkm]) / (2.0 * dzk);
                        double dwdz_R = (state->velW[Rkp] - state->velW[Rkm]) / (2.0 * dzk);
                        dwdz = 0.5 * (dwdz_L + dwdz_R);
                    }

                    double divU = dudx + dvdy + dwdz;

                    double muF = compute_face_mu(idxL, idxR, perPhase, vp->mu, state, vp->phaseMu, mp->nPhases);
                    double tau_yx = muF * (dudy + dvdx);
                    double tau_yy = muF * (2.0 * dvdy - (2.0 / 3.0) * divU);
                    double tau_yz = muF * (dwdy + dvdz);

                    double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                    double vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                    double wFace = 0.0;
                    if (dim >= 3) wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                    double work = tau_yx * uFace + tau_yy * vFace + tau_yz * wFace;

                    double area = mesh_faceAreaY(mesh, i, k);

                    if (j >= 1) {
                        double coeff = area / mesh_cell_volume(mesh, i, j - 1, k);
                        rhsRhoU[idxL] += coeff * tau_yx;
                        rhsRhoV[idxL] += coeff * tau_yy;
                        if (dim >= 3) rhsRhoW[idxL] += coeff * tau_yz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (j < mesh->ny) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
                        rhsRhoU[idxR] -= coeff * tau_yx;
                        rhsRhoV[idxR] -= coeff * tau_yy;
                        if (dim >= 3) rhsRhoW[idxR] -= coeff * tau_yz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }

    /* --- Z-direction faces --- */
    if (dim >= 3) {
        for (int k = 0; k <= mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idxL = mesh_index(mesh, i, j, k - 1);
                    size_t idxR = mesh_index(mesh, i, j, k);

                    double dzk = 0.5 * (mesh_dz(mesh, k - 1) + mesh_dz(mesh, k));

                    double dudz = (state->velU[idxR] - state->velU[idxL]) / dzk;
                    double dvdz = (state->velV[idxR] - state->velV[idxL]) / dzk;
                    double dwdz = (state->velW[idxR] - state->velW[idxL]) / dzk;

                    double dxi = mesh_dx(mesh, i);
                    size_t Lim = mesh_index(mesh, i - 1, j, k - 1);
                    size_t Lip = mesh_index(mesh, i + 1, j, k - 1);
                    size_t Rim = mesh_index(mesh, i - 1, j, k);
                    size_t Rip = mesh_index(mesh, i + 1, j, k);

                    double dudx_L = (state->velU[Lip] - state->velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (state->velU[Rip] - state->velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);

                    double dwdx_L = (state->velW[Lip] - state->velW[Lim]) / (2.0 * dxi);
                    double dwdx_R = (state->velW[Rip] - state->velW[Rim]) / (2.0 * dxi);
                    double dwdx = 0.5 * (dwdx_L + dwdx_R);

                    double dyj = mesh_dy(mesh, j);
                    size_t Ljm = mesh_index(mesh, i, j - 1, k - 1);
                    size_t Ljp = mesh_index(mesh, i, j + 1, k - 1);
                    size_t Rjm = mesh_index(mesh, i, j - 1, k);
                    size_t Rjp = mesh_index(mesh, i, j + 1, k);

                    double dvdy_L = (state->velV[Ljp] - state->velV[Ljm]) / (2.0 * dyj);
                    double dvdy_R = (state->velV[Rjp] - state->velV[Rjm]) / (2.0 * dyj);
                    double dvdy = 0.5 * (dvdy_L + dvdy_R);

                    double dwdy_L = (state->velW[Ljp] - state->velW[Ljm]) / (2.0 * dyj);
                    double dwdy_R = (state->velW[Rjp] - state->velW[Rjm]) / (2.0 * dyj);
                    double dwdy = 0.5 * (dwdy_L + dwdy_R);

                    double divU = dudx + dvdy + dwdz;

                    double muF = compute_face_mu(idxL, idxR, perPhase, vp->mu, state, vp->phaseMu, mp->nPhases);
                    double tau_zx = muF * (dudz + dwdx);
                    double tau_zy = muF * (dvdz + dwdy);
                    double tau_zz = muF * (2.0 * dwdz - (2.0 / 3.0) * divU);

                    double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                    double vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                    double wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                    double work = tau_zx * uFace + tau_zy * vFace + tau_zz * wFace;

                    double area = mesh_faceAreaZ(mesh, i, j);

                    if (k >= 1) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k - 1);
                        rhsRhoU[idxL] += coeff * tau_zx;
                        rhsRhoV[idxL] += coeff * tau_zy;
                        rhsRhoW[idxL] += coeff * tau_zz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (k < mesh->nz) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
                        rhsRhoU[idxR] -= coeff * tau_zx;
                        rhsRhoV[idxR] -= coeff * tau_zy;
                        rhsRhoW[idxR] -= coeff * tau_zz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   GPU device variant
   --------------------------------------------------------------------------- */

/* Cell-based parallel pattern.  For each physical cell (i,j,k), compute the
 * viscous flux at the left (face index i) and right (face index i+1) X-faces
 * and add contributions to this cell's RHS using this cell's own volume.  A
 * face between two physical cells is therefore evaluated twice, once per
 * neighbour cell — same trade-off as in the Riemann flux kernel. */

void add_viscous_fluxes_device(
    const SimulationConfig* config,
    const RectilinearMesh* mesh,
    const SolutionState* state,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE)
{
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = config->dim;
    const ViscousParams* vp = &config->viscousParams;
    const int perPhase = (vp->nPhaseMu > 0);
    const int nPhases = config->multiPhaseParams.nPhases;
    const double muConst = vp->mu;
    const size_t tc = state->totalCells;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* velU  = state->velU;
    double* velV  = state->velV;
    double* velW  = state->velW;
    double* alpha = state->alpha;

    double phaseMu[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph)
        phaseMu[ph] = (perPhase && ph < nPhases) ? vp->phaseMu[ph] : 0.0;

    /* ---------- X-direction face contributions ---------- */
    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseMu[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double dxc = xExt[i + ngx + 1] - xExt[i + ngx];
                double dyc = yExt[j + ngy + 1] - yExt[j + ngy];
                double dzc = zExt[k + ngz + 1] - zExt[k + ngz];
                double vol = dxc * dyc * dzc;
                double area = dyc * dzc;
                double coeff = area / vol;

                double tauL_xx = 0.0, tauL_xy = 0.0, tauL_xz = 0.0, workL = 0.0;
                double tauR_xx = 0.0, tauR_xy = 0.0, tauR_xz = 0.0, workR = 0.0;

                /* side = 0 → left face (between i-1 and i)
                   side = 1 → right face (between i and i+1) */
                for (int side = 0; side < 2; ++side) {
                    int iL = (side == 0) ? (i - 1) : i;
                    int iR = (side == 0) ? i : (i + 1);
                    size_t idxL = (size_t)((iL + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    size_t idxR = (size_t)((iR + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                    double dxiFace = 0.5 * ((xExt[iL + ngx + 1] - xExt[iL + ngx]) +
                                            (xExt[iR + ngx + 1] - xExt[iR + ngx]));

                    double dudx = (velU[idxR] - velU[idxL]) / dxiFace;
                    double dvdx = 0.0, dwdx = 0.0;
                    if (dim >= 2) dvdx = (velV[idxR] - velV[idxL]) / dxiFace;
                    if (dim >= 3) dwdx = (velW[idxR] - velW[idxL]) / dxiFace;

                    double dudy = 0.0, dvdy = 0.0;
                    if (dim >= 2) {
                        double dyj = dyc;
                        size_t Ljm = (size_t)((iL + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (k + ngz)));
                        size_t Ljp = (size_t)((iL + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (k + ngz)));
                        size_t Rjm = (size_t)((iR + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (k + ngz)));
                        size_t Rjp = (size_t)((iR + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (k + ngz)));
                        double dudy_L = (velU[Ljp] - velU[Ljm]) / (2.0 * dyj);
                        double dudy_R = (velU[Rjp] - velU[Rjm]) / (2.0 * dyj);
                        dudy = 0.5 * (dudy_L + dudy_R);
                        double dvdy_L = (velV[Ljp] - velV[Ljm]) / (2.0 * dyj);
                        double dvdy_R = (velV[Rjp] - velV[Rjm]) / (2.0 * dyj);
                        dvdy = 0.5 * (dvdy_L + dvdy_R);
                    }

                    double dudz = 0.0, dwdz = 0.0;
                    if (dim >= 3) {
                        double dzk = dzc;
                        size_t Lkm = (size_t)((iL + ngx) + nxTot * ((j + ngy) + nyTot * (k - 1 + ngz)));
                        size_t Lkp = (size_t)((iL + ngx) + nxTot * ((j + ngy) + nyTot * (k + 1 + ngz)));
                        size_t Rkm = (size_t)((iR + ngx) + nxTot * ((j + ngy) + nyTot * (k - 1 + ngz)));
                        size_t Rkp = (size_t)((iR + ngx) + nxTot * ((j + ngy) + nyTot * (k + 1 + ngz)));
                        double dudz_L = (velU[Lkp] - velU[Lkm]) / (2.0 * dzk);
                        double dudz_R = (velU[Rkp] - velU[Rkm]) / (2.0 * dzk);
                        dudz = 0.5 * (dudz_L + dudz_R);
                        double dwdz_L = (velW[Lkp] - velW[Lkm]) / (2.0 * dzk);
                        double dwdz_R = (velW[Rkp] - velW[Rkm]) / (2.0 * dzk);
                        dwdz = 0.5 * (dwdz_L + dwdz_R);
                    }

                    double divU = dudx + dvdy + dwdz;

                    double muF = muConst;
                    if (perPhase) {
                        double muL = 0.0, muR = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            muL += alpha[(size_t)ph * tc + idxL] * phaseMu[ph];
                            muR += alpha[(size_t)ph * tc + idxR] * phaseMu[ph];
                        }
                        muF = 0.5 * (muL + muR);
                    }
                    double tau_xx = muF * (2.0 * dudx - (2.0 / 3.0) * divU);
                    double tau_xy = muF * (dvdx + dudy);
                    double tau_xz = muF * (dwdx + dudz);

                    double uFace = 0.5 * (velU[idxL] + velU[idxR]);
                    double vFace = (dim >= 2) ? 0.5 * (velV[idxL] + velV[idxR]) : 0.0;
                    double wFace = (dim >= 3) ? 0.5 * (velW[idxL] + velW[idxR]) : 0.0;
                    double work = tau_xx * uFace + tau_xy * vFace + tau_xz * wFace;

                    if (side == 0) {
                        tauL_xx = tau_xx; tauL_xy = tau_xy; tauL_xz = tau_xz; workL = work;
                    } else {
                        tauR_xx = tau_xx; tauR_xy = tau_xy; tauR_xz = tau_xz; workR = work;
                    }
                }

                rhsRhoU[idx] += coeff * (tauR_xx - tauL_xx);
                if (dim >= 2) rhsRhoV[idx] += coeff * (tauR_xy - tauL_xy);
                if (dim >= 3) rhsRhoW[idx] += coeff * (tauR_xz - tauL_xz);
                rhsRhoE[idx] += coeff * (workR - workL);
            }
        }
    }

    if (dim < 2) return;

    /* ---------- Y-direction face contributions ---------- */
    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseMu[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double dxc = xExt[i + ngx + 1] - xExt[i + ngx];
                double dyc = yExt[j + ngy + 1] - yExt[j + ngy];
                double dzc = zExt[k + ngz + 1] - zExt[k + ngz];
                double vol = dxc * dyc * dzc;
                double area = dxc * dzc;
                double coeff = area / vol;

                double tauL_yx = 0.0, tauL_yy = 0.0, tauL_yz = 0.0, workL = 0.0;
                double tauR_yx = 0.0, tauR_yy = 0.0, tauR_yz = 0.0, workR = 0.0;

                for (int side = 0; side < 2; ++side) {
                    int jL = (side == 0) ? (j - 1) : j;
                    int jR = (side == 0) ? j : (j + 1);
                    size_t idxL = (size_t)((i + ngx) + nxTot * ((jL + ngy) + nyTot * (k + ngz)));
                    size_t idxR = (size_t)((i + ngx) + nxTot * ((jR + ngy) + nyTot * (k + ngz)));

                    double dyjFace = 0.5 * ((yExt[jL + ngy + 1] - yExt[jL + ngy]) +
                                            (yExt[jR + ngy + 1] - yExt[jR + ngy]));

                    double dudy = (velU[idxR] - velU[idxL]) / dyjFace;
                    double dvdy = (velV[idxR] - velV[idxL]) / dyjFace;
                    double dwdy = 0.0;
                    if (dim >= 3) dwdy = (velW[idxR] - velW[idxL]) / dyjFace;

                    double dxi = dxc;
                    size_t Lim = (size_t)((i - 1 + ngx) + nxTot * ((jL + ngy) + nyTot * (k + ngz)));
                    size_t Lip = (size_t)((i + 1 + ngx) + nxTot * ((jL + ngy) + nyTot * (k + ngz)));
                    size_t Rim = (size_t)((i - 1 + ngx) + nxTot * ((jR + ngy) + nyTot * (k + ngz)));
                    size_t Rip = (size_t)((i + 1 + ngx) + nxTot * ((jR + ngy) + nyTot * (k + ngz)));
                    double dudx_L = (velU[Lip] - velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (velU[Rip] - velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);
                    double dvdx_L = (velV[Lip] - velV[Lim]) / (2.0 * dxi);
                    double dvdx_R = (velV[Rip] - velV[Rim]) / (2.0 * dxi);
                    double dvdx = 0.5 * (dvdx_L + dvdx_R);

                    double dvdz = 0.0, dwdz = 0.0;
                    if (dim >= 3) {
                        double dzk = dzc;
                        size_t Lkm = (size_t)((i + ngx) + nxTot * ((jL + ngy) + nyTot * (k - 1 + ngz)));
                        size_t Lkp = (size_t)((i + ngx) + nxTot * ((jL + ngy) + nyTot * (k + 1 + ngz)));
                        size_t Rkm = (size_t)((i + ngx) + nxTot * ((jR + ngy) + nyTot * (k - 1 + ngz)));
                        size_t Rkp = (size_t)((i + ngx) + nxTot * ((jR + ngy) + nyTot * (k + 1 + ngz)));
                        double dvdz_L = (velV[Lkp] - velV[Lkm]) / (2.0 * dzk);
                        double dvdz_R = (velV[Rkp] - velV[Rkm]) / (2.0 * dzk);
                        dvdz = 0.5 * (dvdz_L + dvdz_R);
                        double dwdz_L = (velW[Lkp] - velW[Lkm]) / (2.0 * dzk);
                        double dwdz_R = (velW[Rkp] - velW[Rkm]) / (2.0 * dzk);
                        dwdz = 0.5 * (dwdz_L + dwdz_R);
                    }

                    double divU = dudx + dvdy + dwdz;

                    double muF = muConst;
                    if (perPhase) {
                        double muL = 0.0, muR = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            muL += alpha[(size_t)ph * tc + idxL] * phaseMu[ph];
                            muR += alpha[(size_t)ph * tc + idxR] * phaseMu[ph];
                        }
                        muF = 0.5 * (muL + muR);
                    }
                    double tau_yx = muF * (dudy + dvdx);
                    double tau_yy = muF * (2.0 * dvdy - (2.0 / 3.0) * divU);
                    double tau_yz = muF * (dwdy + dvdz);

                    double uFace = 0.5 * (velU[idxL] + velU[idxR]);
                    double vFace = 0.5 * (velV[idxL] + velV[idxR]);
                    double wFace = (dim >= 3) ? 0.5 * (velW[idxL] + velW[idxR]) : 0.0;
                    double work = tau_yx * uFace + tau_yy * vFace + tau_yz * wFace;

                    if (side == 0) {
                        tauL_yx = tau_yx; tauL_yy = tau_yy; tauL_yz = tau_yz; workL = work;
                    } else {
                        tauR_yx = tau_yx; tauR_yy = tau_yy; tauR_yz = tau_yz; workR = work;
                    }
                }

                rhsRhoU[idx] += coeff * (tauR_yx - tauL_yx);
                rhsRhoV[idx] += coeff * (tauR_yy - tauL_yy);
                if (dim >= 3) rhsRhoW[idx] += coeff * (tauR_yz - tauL_yz);
                rhsRhoE[idx] += coeff * (workR - workL);
            }
        }
    }

    if (dim < 3) return;

    /* ---------- Z-direction face contributions ---------- */
    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseMu[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double dxc = xExt[i + ngx + 1] - xExt[i + ngx];
                double dyc = yExt[j + ngy + 1] - yExt[j + ngy];
                double dzc = zExt[k + ngz + 1] - zExt[k + ngz];
                double vol = dxc * dyc * dzc;
                double area = dxc * dyc;
                double coeff = area / vol;

                double tauL_zx = 0.0, tauL_zy = 0.0, tauL_zz = 0.0, workL = 0.0;
                double tauR_zx = 0.0, tauR_zy = 0.0, tauR_zz = 0.0, workR = 0.0;

                for (int side = 0; side < 2; ++side) {
                    int kL = (side == 0) ? (k - 1) : k;
                    int kR = (side == 0) ? k : (k + 1);
                    size_t idxL = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (kL + ngz)));
                    size_t idxR = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (kR + ngz)));

                    double dzkFace = 0.5 * ((zExt[kL + ngz + 1] - zExt[kL + ngz]) +
                                            (zExt[kR + ngz + 1] - zExt[kR + ngz]));

                    double dudz = (velU[idxR] - velU[idxL]) / dzkFace;
                    double dvdz = (velV[idxR] - velV[idxL]) / dzkFace;
                    double dwdz = (velW[idxR] - velW[idxL]) / dzkFace;

                    double dxi = dxc;
                    size_t Lim = (size_t)((i - 1 + ngx) + nxTot * ((j + ngy) + nyTot * (kL + ngz)));
                    size_t Lip = (size_t)((i + 1 + ngx) + nxTot * ((j + ngy) + nyTot * (kL + ngz)));
                    size_t Rim = (size_t)((i - 1 + ngx) + nxTot * ((j + ngy) + nyTot * (kR + ngz)));
                    size_t Rip = (size_t)((i + 1 + ngx) + nxTot * ((j + ngy) + nyTot * (kR + ngz)));
                    double dudx_L = (velU[Lip] - velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (velU[Rip] - velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);
                    double dwdx_L = (velW[Lip] - velW[Lim]) / (2.0 * dxi);
                    double dwdx_R = (velW[Rip] - velW[Rim]) / (2.0 * dxi);
                    double dwdx = 0.5 * (dwdx_L + dwdx_R);

                    double dyj = dyc;
                    size_t Ljm = (size_t)((i + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (kL + ngz)));
                    size_t Ljp = (size_t)((i + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (kL + ngz)));
                    size_t Rjm = (size_t)((i + ngx) + nxTot * ((j - 1 + ngy) + nyTot * (kR + ngz)));
                    size_t Rjp = (size_t)((i + ngx) + nxTot * ((j + 1 + ngy) + nyTot * (kR + ngz)));
                    double dvdy_L = (velV[Ljp] - velV[Ljm]) / (2.0 * dyj);
                    double dvdy_R = (velV[Rjp] - velV[Rjm]) / (2.0 * dyj);
                    double dvdy = 0.5 * (dvdy_L + dvdy_R);
                    double dwdy_L = (velW[Ljp] - velW[Ljm]) / (2.0 * dyj);
                    double dwdy_R = (velW[Rjp] - velW[Rjm]) / (2.0 * dyj);
                    double dwdy = 0.5 * (dwdy_L + dwdy_R);

                    double divU = dudx + dvdy + dwdz;

                    double muF = muConst;
                    if (perPhase) {
                        double muL = 0.0, muR = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            muL += alpha[(size_t)ph * tc + idxL] * phaseMu[ph];
                            muR += alpha[(size_t)ph * tc + idxR] * phaseMu[ph];
                        }
                        muF = 0.5 * (muL + muR);
                    }
                    double tau_zx = muF * (dudz + dwdx);
                    double tau_zy = muF * (dvdz + dwdy);
                    double tau_zz = muF * (2.0 * dwdz - (2.0 / 3.0) * divU);

                    double uFace = 0.5 * (velU[idxL] + velU[idxR]);
                    double vFace = 0.5 * (velV[idxL] + velV[idxR]);
                    double wFace = 0.5 * (velW[idxL] + velW[idxR]);
                    double work = tau_zx * uFace + tau_zy * vFace + tau_zz * wFace;

                    if (side == 0) {
                        tauL_zx = tau_zx; tauL_zy = tau_zy; tauL_zz = tau_zz; workL = work;
                    } else {
                        tauR_zx = tau_zx; tauR_zy = tau_zy; tauR_zz = tau_zz; workR = work;
                    }
                }

                rhsRhoU[idx] += coeff * (tauR_zx - tauL_zx);
                rhsRhoV[idx] += coeff * (tauR_zy - tauL_zy);
                rhsRhoW[idx] += coeff * (tauR_zz - tauL_zz);
                rhsRhoE[idx] += coeff * (workR - workL);
            }
        }
    }
}
