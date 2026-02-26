#include "ViscousFlux.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"

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
