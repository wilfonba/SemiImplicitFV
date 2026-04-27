#include "Reconstruction.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "MixtureEOS.hpp"
#include "NvtxRange.hpp"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cmath>
#include <algorithm>

/* Device-safe mesh index: same computation as mesh_index(), but takes the
 * mesh geometry as loose scalars so the expression can be used from inside
 * an `omp target` region without dereferencing the host mesh struct. */
#define CELL_IDX(ii, jj, kk) \
    ((size_t)(((ii) + ngx) + nxTot * (((jj) + ngy) + nyTot * ((kk) + ngz))))

/* ---- WENO / Upwind stencil functions and dispatcher ----
 * All helpers and the `reconstructScalar` dispatcher are inside an
 * `omp declare target` region so they can be called from the reconstruction
 * kernels running on the GPU.  The dispatcher replaces the original function
 * pointer (ReconFn) scheme with an enum-switch, since function pointers are
 * not portable to device code. */

#pragma omp declare target

static double weno3Left(const double* v, double eps) {
    double p0 = -0.5 * v[0] + 1.5 * v[1];
    double p1 =  0.5 * v[1] + 0.5 * v[2];

    double b0 = (v[1] - v[0]) * (v[1] - v[0]);
    double b1 = (v[2] - v[1]) * (v[2] - v[1]);

    const double d0 = 1.0 / 3.0;
    const double d1 = 2.0 / 3.0;

    double a0 = d0 / ((eps + b0) * (eps + b0));
    double a1 = d1 / ((eps + b1) * (eps + b1));
    double aSum = a0 + a1;

    return (a0 * p0 + a1 * p1) / aSum;
}

static double weno3Right(const double* v, double eps) {
    double p0 =  0.5 * v[0] + 0.5 * v[1];
    double p1 =  1.5 * v[1] - 0.5 * v[2];

    double b0 = (v[1] - v[0]) * (v[1] - v[0]);
    double b1 = (v[2] - v[1]) * (v[2] - v[1]);

    const double d0 = 2.0 / 3.0;
    const double d1 = 1.0 / 3.0;

    double a0 = d0 / ((eps + b0) * (eps + b0));
    double a1 = d1 / ((eps + b1) * (eps + b1));
    double aSum = a0 + a1;

    return (a0 * p0 + a1 * p1) / aSum;
}

static double weno5Left(const double* v, double eps) {
    double p0 = (1.0/3.0)*v[0] - (7.0/6.0)*v[1] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[1] + (5.0/6.0)*v[2] + (1.0/3.0)*v[3];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[3] - (1.0/6.0)*v[4];

    double b0 = (13.0/12.0)*(v[0] - 2.0*v[1] + v[2])*(v[0] - 2.0*v[1] + v[2])
              + (1.0/4.0)*(v[0] - 4.0*v[1] + 3.0*v[2])*(v[0] - 4.0*v[1] + 3.0*v[2]);
    double b1 = (13.0/12.0)*(v[1] - 2.0*v[2] + v[3])*(v[1] - 2.0*v[2] + v[3])
              + (1.0/4.0)*(v[1] - v[3])*(v[1] - v[3]);
    double b2 = (13.0/12.0)*(v[2] - 2.0*v[3] + v[4])*(v[2] - 2.0*v[3] + v[4])
              + (1.0/4.0)*(3.0*v[2] - 4.0*v[3] + v[4])*(3.0*v[2] - 4.0*v[3] + v[4]);

    const double d0 = 1.0 / 10.0;
    const double d1 = 6.0 / 10.0;
    const double d2 = 3.0 / 10.0;

    double a0 = d0 / ((eps + b0) * (eps + b0));
    double a1 = d1 / ((eps + b1) * (eps + b1));
    double a2 = d2 / ((eps + b2) * (eps + b2));
    double aSum = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / aSum;
}

static double weno5Right(const double* v, double eps) {
    double p0 = (1.0/3.0)*v[4] - (7.0/6.0)*v[3] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[3] + (5.0/6.0)*v[2] + (1.0/3.0)*v[1];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[1] - (1.0/6.0)*v[0];

    double b0 = (13.0/12.0)*(v[4] - 2.0*v[3] + v[2])*(v[4] - 2.0*v[3] + v[2])
              + (1.0/4.0)*(v[4] - 4.0*v[3] + 3.0*v[2])*(v[4] - 4.0*v[3] + 3.0*v[2]);
    double b1 = (13.0/12.0)*(v[3] - 2.0*v[2] + v[1])*(v[3] - 2.0*v[2] + v[1])
              + (1.0/4.0)*(v[3] - v[1])*(v[3] - v[1]);
    double b2 = (13.0/12.0)*(v[2] - 2.0*v[1] + v[0])*(v[2] - 2.0*v[1] + v[0])
              + (1.0/4.0)*(3.0*v[2] - 4.0*v[1] + v[0])*(3.0*v[2] - 4.0*v[1] + v[0]);

    const double d0 = 1.0 / 10.0;
    const double d1 = 6.0 / 10.0;
    const double d2 = 3.0 / 10.0;

    double a0 = d0 / ((eps + b0) * (eps + b0));
    double a1 = d1 / ((eps + b1) * (eps + b1));
    double a2 = d2 / ((eps + b2) * (eps + b2));
    double aSum = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / aSum;
}

static double upwind3Left(const double* v, double /*eps*/) {
    double p0 = -0.5 * v[0] + 1.5 * v[1];
    double p1 =  0.5 * v[1] + 0.5 * v[2];
    return (1.0/3.0) * p0 + (2.0/3.0) * p1;
}

static double upwind3Right(const double* v, double /*eps*/) {
    double p0 =  0.5 * v[0] + 0.5 * v[1];
    double p1 =  1.5 * v[1] - 0.5 * v[2];
    return (2.0/3.0) * p0 + (1.0/3.0) * p1;
}

static double upwind5Left(const double* v, double /*eps*/) {
    double p0 = (1.0/3.0)*v[0] - (7.0/6.0)*v[1] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[1] + (5.0/6.0)*v[2] + (1.0/3.0)*v[3];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[3] - (1.0/6.0)*v[4];
    return (1.0/10.0) * p0 + (6.0/10.0) * p1 + (3.0/10.0) * p2;
}

static double upwind5Right(const double* v, double /*eps*/) {
    double p0 = (1.0/3.0)*v[4] - (7.0/6.0)*v[3] + (11.0/6.0)*v[2];
    double p1 = -(1.0/6.0)*v[3] + (5.0/6.0)*v[2] + (1.0/3.0)*v[1];
    double p2 = (1.0/3.0)*v[2] + (5.0/6.0)*v[1] - (1.0/6.0)*v[0];
    return (1.0/10.0) * p0 + (6.0/10.0) * p1 + (3.0/10.0) * p2;
}

static inline void reconstructScalar(
    const double* field,
    const size_t* cells,
    int stencilSize,
    enum ReconstructionOrder order,
    double eps,
    double* outLeft,
    double* outRight)
{
    double vL[5], vR[5];
    for (int s = 0; s < stencilSize; ++s) {
        vL[s] = field[cells[s]];
        vR[s] = field[cells[s + 1]];
    }
    switch (order) {
        case WENO3:
            *outLeft  = weno3Left(vL, eps);
            *outRight = weno3Right(vR, eps);
            break;
        case UPWIND3:
            *outLeft  = upwind3Left(vL, eps);
            *outRight = upwind3Right(vR, eps);
            break;
        case WENO5:
            *outLeft  = weno5Left(vL, eps);
            *outRight = weno5Right(vR, eps);
            break;
        case UPWIND5:
            *outLeft  = upwind5Left(vL, eps);
            *outRight = upwind5Right(vR, eps);
            break;
        default:
            *outLeft  = vL[stencilSize / 2];
            *outRight = vR[stencilSize / 2];
            break;
    }
}

#pragma omp end declare target

/* ---- Direction-specific reconstruction sweeps ---- */

static void reconstructX(struct ReconstructorData* r,
                          const struct SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          const struct SolutionState* state)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int recNx = r->nx, recNy = r->ny;
    const int rdim = r->dim;
    const enum ReconstructionOrder order = r->order;
    const double wenoEps = r->wenoEps;
    const double rGamma = r->gamma;
    const double rPInf  = r->pInf;

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* pres = state->pres;
    const double* sig  = state->sigma;
    const double* velV = (rdim >= 2) ? state->velV : state->velU;  /* unused if rdim < 2 */
    const double* velW = (rdim >= 3) ? state->velW : state->velU;  /* unused if rdim < 3 */
    PrimitiveState* xLeftA  = r->xLeft;
    PrimitiveState* xRightA = r->xRight;

    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const int useIGR = config->useIGR;
    const size_t totalCells = state->totalCells;
    const double* alphaPtr = state->alpha;  /* NULL for single-phase, fine */

    /* Per-phase EOS coeffs as flat arrays so the kernel doesn't need to
     * dereference MultiPhaseParams::phases (a host struct) on device. */
    double phaseGamma[MAX_PHASES], phasePinf[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph) { phaseGamma[ph] = 0.0; phasePinf[ph] = 0.0; }
    if (multiPhase) {
        for (int ph = 0; ph < nAlphas; ++ph) {
            phaseGamma[ph] = config->multiPhaseParams.phases[ph].gamma;
            phasePinf[ph]  = config->multiPhaseParams.phases[ph].pInf;
        }
    }

    /* GPU kernel: every (i, j, k) face is independent (each writes to its
     * own xLeft[f] / xRight[f]).  Note i ranges 0..nx inclusive — the face
     * count in X is (nx+1) * ny * nz.  `useIGR` is uniform per launch — the
     * sigma load + reconstruct + store is skipped entirely when IGR is off,
     * since HLLC/Rusanov/LF only read sigma in their useIGR branch. */
    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseGamma[0:MAX_PHASES], phasePinf[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                size_t fIdx = (size_t)(i + (recNx + 1) * (j + recNy * k));
                PrimitiveState* left  = &xLeftA[fIdx];
                PrimitiveState* right = &xRightA[fIdx];

                if (order == WENO1 || order == UPWIND1) {
                    size_t idxL = CELL_IDX(i - 1, j, k);
                    size_t idxR = CELL_IDX(i,     j, k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->p     = pres[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->p     = pres[idxR];
                    if (useIGR) {
                        left->sigma  = sig[idxL];
                        right->sigma = sig[idxR];
                    }
                    if (rdim >= 2) {
                        left->u[1]  = velV[idxL];
                        right->u[1] = velV[idxR];
                    }
                    if (rdim >= 3) {
                        left->u[2]  = velW[idxL];
                        right->u[2] = velW[idxR];
                    }
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph]  = alphaPtr[(size_t)ph * totalCells + idxL];
                            right->alpha[ph] = alphaPtr[(size_t)ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else if (order == WENO3 || order == UPWIND3) {
                    size_t c[4];
                    c[0] = CELL_IDX(i - 2, j, k);
                    c[1] = CELL_IDX(i - 1, j, k);
                    c[2] = CELL_IDX(i,     j, k);
                    c[3] = CELL_IDX(i + 1, j, k);

                    reconstructScalar(rho,  c, 3, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 3, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(pres, c, 3, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 3, order, wenoEps, &left->sigma, &right->sigma);
                    if (rdim >= 2)
                        reconstructScalar(velV, c, 3, order, wenoEps, &left->u[1], &right->u[1]);
                    if (rdim >= 3)
                        reconstructScalar(velW, c, 3, order, wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 3, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = CELL_IDX(i - 3, j, k);
                    c[1] = CELL_IDX(i - 2, j, k);
                    c[2] = CELL_IDX(i - 1, j, k);
                    c[3] = CELL_IDX(i,     j, k);
                    c[4] = CELL_IDX(i + 1, j, k);
                    c[5] = CELL_IDX(i + 2, j, k);

                    reconstructScalar(rho,  c, 5, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 5, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(pres, c, 5, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 5, order, wenoEps, &left->sigma, &right->sigma);
                    if (rdim >= 2)
                        reconstructScalar(velV, c, 5, order, wenoEps, &left->u[1], &right->u[1]);
                    if (rdim >= 3)
                        reconstructScalar(velW, c, 5, order, wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 5, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
            }
        }
    }
}

static void reconstructY(struct ReconstructorData* r,
                          const struct SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          const struct SolutionState* state)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int recNx = r->nx, recNy = r->ny;
    const int rdim = r->dim;
    const enum ReconstructionOrder order = r->order;
    const double wenoEps = r->wenoEps;
    const double rGamma = r->gamma, rPInf = r->pInf;

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* velV = state->velV;
    const double* pres = state->pres;
    const double* sig  = state->sigma;
    const double* velW = (rdim >= 3) ? state->velW : state->velU;
    const int useIGR = config->useIGR;
    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const size_t totalCells = state->totalCells;
    const double* alphaPtr = state->alpha;
    PrimitiveState* yLeftA  = r->yLeft;
    PrimitiveState* yRightA = r->yRight;

    double phaseGamma[MAX_PHASES], phasePinf[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph) { phaseGamma[ph] = 0.0; phasePinf[ph] = 0.0; }
    if (multiPhase) {
        for (int ph = 0; ph < nAlphas; ++ph) {
            phaseGamma[ph] = config->multiPhaseParams.phases[ph].gamma;
            phasePinf[ph]  = config->multiPhaseParams.phases[ph].pInf;
        }
    }

    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseGamma[0:MAX_PHASES], phasePinf[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fIdx = (size_t)(i + recNx * (j + (recNy + 1) * k));
                PrimitiveState* left  = &yLeftA[fIdx];
                PrimitiveState* right = &yRightA[fIdx];

                if (order == WENO1 || order == UPWIND1) {
                    size_t idxL = CELL_IDX(i, j - 1, k);
                    size_t idxR = CELL_IDX(i, j,     k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->u[1]  = velV[idxL];
                    left->p     = pres[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->u[1]  = velV[idxR];
                    right->p     = pres[idxR];
                    if (useIGR) {
                        left->sigma  = sig[idxL];
                        right->sigma = sig[idxR];
                    }
                    if (rdim >= 3) {
                        left->u[2]  = velW[idxL];
                        right->u[2] = velW[idxR];
                    }
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph]  = alphaPtr[(size_t)ph * totalCells + idxL];
                            right->alpha[ph] = alphaPtr[(size_t)ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else if (order == WENO3 || order == UPWIND3) {
                    size_t c[4];
                    c[0] = CELL_IDX(i, j - 2, k);
                    c[1] = CELL_IDX(i, j - 1, k);
                    c[2] = CELL_IDX(i, j,     k);
                    c[3] = CELL_IDX(i, j + 1, k);

                    reconstructScalar(rho,  c, 3, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 3, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(velV, c, 3, order, wenoEps, &left->u[1],  &right->u[1]);
                    reconstructScalar(pres, c, 3, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 3, order, wenoEps, &left->sigma, &right->sigma);
                    if (rdim >= 3)
                        reconstructScalar(velW, c, 3, order, wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 3, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = CELL_IDX(i, j - 3, k);
                    c[1] = CELL_IDX(i, j - 2, k);
                    c[2] = CELL_IDX(i, j - 1, k);
                    c[3] = CELL_IDX(i, j,     k);
                    c[4] = CELL_IDX(i, j + 1, k);
                    c[5] = CELL_IDX(i, j + 2, k);

                    reconstructScalar(rho,  c, 5, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 5, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(velV, c, 5, order, wenoEps, &left->u[1],  &right->u[1]);
                    reconstructScalar(pres, c, 5, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 5, order, wenoEps, &left->sigma, &right->sigma);
                    if (rdim >= 3)
                        reconstructScalar(velW, c, 5, order, wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 5, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
            }
        }
    }
}

static void reconstructZ(struct ReconstructorData* r,
                          const struct SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          const struct SolutionState* state)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int recNx = r->nx, recNy = r->ny;
    const enum ReconstructionOrder order = r->order;
    const double wenoEps = r->wenoEps;
    const double rGamma = r->gamma, rPInf = r->pInf;

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* velV = state->velV;
    const double* velW = state->velW;
    const double* pres = state->pres;
    const double* sig  = state->sigma;
    const int useIGR = config->useIGR;
    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const size_t totalCells = state->totalCells;
    const double* alphaPtr = state->alpha;
    PrimitiveState* zLeftA  = r->zLeft;
    PrimitiveState* zRightA = r->zRight;

    double phaseGamma[MAX_PHASES], phasePinf[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph) { phaseGamma[ph] = 0.0; phasePinf[ph] = 0.0; }
    if (multiPhase) {
        for (int ph = 0; ph < nAlphas; ++ph) {
            phaseGamma[ph] = config->multiPhaseParams.phases[ph].gamma;
            phasePinf[ph]  = config->multiPhaseParams.phases[ph].pInf;
        }
    }

    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseGamma[0:MAX_PHASES], phasePinf[0:MAX_PHASES])
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fIdx = (size_t)(i + recNx * (j + recNy * k));
                PrimitiveState* left  = &zLeftA[fIdx];
                PrimitiveState* right = &zRightA[fIdx];

                if (order == WENO1 || order == UPWIND1) {
                    size_t idxL = CELL_IDX(i, j, k - 1);
                    size_t idxR = CELL_IDX(i, j, k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->u[1]  = velV[idxL];
                    left->u[2]  = velW[idxL];
                    left->p     = pres[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->u[1]  = velV[idxR];
                    right->u[2]  = velW[idxR];
                    right->p     = pres[idxR];
                    if (useIGR) {
                        left->sigma  = sig[idxL];
                        right->sigma = sig[idxR];
                    }
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph]  = alphaPtr[(size_t)ph * totalCells + idxL];
                            right->alpha[ph] = alphaPtr[(size_t)ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else if (order == WENO3 || order == UPWIND3) {
                    size_t c[4];
                    c[0] = CELL_IDX(i, j, k - 2);
                    c[1] = CELL_IDX(i, j, k - 1);
                    c[2] = CELL_IDX(i, j, k);
                    c[3] = CELL_IDX(i, j, k + 1);

                    reconstructScalar(rho,  c, 3, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 3, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(velV, c, 3, order, wenoEps, &left->u[1],  &right->u[1]);
                    reconstructScalar(velW, c, 3, order, wenoEps, &left->u[2],  &right->u[2]);
                    reconstructScalar(pres, c, 3, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 3, order, wenoEps, &left->sigma, &right->sigma);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 3, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = CELL_IDX(i, j, k - 3);
                    c[1] = CELL_IDX(i, j, k - 2);
                    c[2] = CELL_IDX(i, j, k - 1);
                    c[3] = CELL_IDX(i, j, k);
                    c[4] = CELL_IDX(i, j, k + 1);
                    c[5] = CELL_IDX(i, j, k + 2);

                    reconstructScalar(rho,  c, 5, order, wenoEps, &left->rho,   &right->rho);
                    reconstructScalar(velU, c, 5, order, wenoEps, &left->u[0],  &right->u[0]);
                    reconstructScalar(velV, c, 5, order, wenoEps, &left->u[1],  &right->u[1]);
                    reconstructScalar(velW, c, 5, order, wenoEps, &left->u[2],  &right->u[2]);
                    reconstructScalar(pres, c, 5, order, wenoEps, &left->p,     &right->p);
                    if (useIGR)
                        reconstructScalar(sig,  c, 5, order, wenoEps, &left->sigma, &right->sigma);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(alphaPtr + (size_t)ph * totalCells, c, 5, order,
                                              wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf_arr(left->alpha,  nAlphas,
                            phaseGamma, phasePinf, &left->gammaEff,  &left->piInfEff);
                        effective_gamma_and_pi_inf_arr(right->alpha, nAlphas,
                            phaseGamma, phasePinf, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = rGamma;   left->piInfEff = rPInf;
                        right->gammaEff = rGamma;  right->piInfEff = rPInf;
                    }
                }
            }
        }
    }
}

/* ---- MTHINC (Multi-dimensional THINC) interface compression ---- */
/*
 * Multi-dimensional THINC reconstructs a hyperbolic tangent profile oriented
 * along the interface normal (computed from the gradient of alpha across all
 * dimensions), then integrates that profile over each cell face to obtain
 * face-averaged volume fractions.  This prevents the grid-aligned staircase
 * artifacts that arise from dimension-by-dimension THINC.
 *
 * The THINC function in normalised cell coordinates xi in [-1/2, 1/2]^dim:
 *
 *   H(xi) = 0.5 * (1 + tanh( beta * (n_hat . xi + d) ))
 *
 * where n_hat is the unit interface normal in cell-scaled coordinates, beta
 * controls the sharpness, and d is determined by the conservation constraint:
 *
 *   integral_cell H(xi) dxi = alpha_cell
 *
 * The conservation constraint is solved with Newton iteration using Gaussian
 * quadrature for the multi-dimensional volume integral.  Face-averaged values
 * are obtained by analytically integrating along one direction and using
 * Gauss quadrature for the remaining transverse direction(s).
 *
 * Reference: Xie & Xiao, J. Comput. Phys., 349, 415-440 (2017).
 */

/* 3-point Gauss-Legendre quadrature on [-1/2, 1/2] */
static const double GQ3_PTS[3] = {
    -0.5 * 0.7745966692414834,   /* -sqrt(3/5) / 2 */
     0.0,
     0.5 * 0.7745966692414834    /*  sqrt(3/5) / 2 */
};
static const double GQ3_WTS[3] = {
    5.0 / 18.0,
    8.0 / 18.0,
    5.0 / 18.0
};

/*
 * Stable computation of ln(cosh(x)).
 * For large |x| the naive formula overflows; we use
 *   ln(cosh(x)) = |x| + ln(1 + exp(-2|x|)) - ln(2)
 */
static double log_cosh(double x)
{
    double ax = std::fabs(x);
    if (ax > 20.0)
        return ax - 0.6931471805599453;  /* ax - ln(2) */
    return ax + std::log(1.0 + std::exp(-2.0 * ax)) - 0.6931471805599453;
}

/*
 * Analytical 1-D integral:
 *   I(a, b) = integral_{-1/2}^{1/2}  0.5 * (1 + tanh(a + b t))  dt
 *
 * If |b| ~ 0 the integral degenerates to a point evaluation.
 */
static double thinc_integral_1d(double a, double b)
{
    if (std::fabs(b) < 1e-14)
        return 0.5 * (1.0 + std::tanh(a));
    return 0.5 + (log_cosh(a + 0.5 * b) - log_cosh(a - 0.5 * b)) / (2.0 * b);
}

/*
 * Volume integral of H over [-1/2, 1/2]^dim.
 * Integrates analytically along dimension 0, uses 3-pt Gauss quadrature
 * for the remaining transverse directions.
 */
static double mthinc_volume_integral(const double n[3], double d,
                                      double beta, int dim)
{
    if (dim == 1) {
        return thinc_integral_1d(beta * d, beta * n[0]);
    }
    else if (dim == 2) {
        double sum = 0.0;
        for (int q = 0; q < 3; ++q) {
            double a = beta * (n[1] * GQ3_PTS[q] + d);
            sum += GQ3_WTS[q] * thinc_integral_1d(a, beta * n[0]);
        }
        return sum;
    }
    else { /* dim == 3 */
        double sum = 0.0;
        for (int q1 = 0; q1 < 3; ++q1) {
            for (int q2 = 0; q2 < 3; ++q2) {
                double a = beta * (n[1] * GQ3_PTS[q1]
                                 + n[2] * GQ3_PTS[q2] + d);
                sum += GQ3_WTS[q1] * GQ3_WTS[q2]
                     * thinc_integral_1d(a, beta * n[0]);
            }
        }
        return sum;
    }
}

/*
 * Derivative dV/dd of the volume integral (for Newton iteration).
 *   dH/dd = 0.5 * beta * sech^2(beta * (n.xi + d))
 * Evaluated with Gauss quadrature over all dimensions.
 */
static double mthinc_volume_integral_dd(const double n[3], double d,
                                         double beta, int dim)
{
    double sum = 0.0;
    if (dim == 1) {
        for (int q = 0; q < 3; ++q) {
            double th = std::tanh(beta * (n[0] * GQ3_PTS[q] + d));
            sum += GQ3_WTS[q] * (1.0 - th * th);
        }
    }
    else if (dim == 2) {
        for (int q1 = 0; q1 < 3; ++q1) {
            for (int q2 = 0; q2 < 3; ++q2) {
                double th = std::tanh(beta * (n[0] * GQ3_PTS[q1]
                                            + n[1] * GQ3_PTS[q2] + d));
                sum += GQ3_WTS[q1] * GQ3_WTS[q2] * (1.0 - th * th);
            }
        }
    }
    else { /* dim == 3 */
        for (int q1 = 0; q1 < 3; ++q1) {
            for (int q2 = 0; q2 < 3; ++q2) {
                for (int q3 = 0; q3 < 3; ++q3) {
                    double th = std::tanh(beta * (n[0] * GQ3_PTS[q1]
                                                + n[1] * GQ3_PTS[q2]
                                                + n[2] * GQ3_PTS[q3] + d));
                    sum += GQ3_WTS[q1] * GQ3_WTS[q2] * GQ3_WTS[q3]
                         * (1.0 - th * th);
                }
            }
        }
    }
    return 0.5 * beta * sum;
}

/*
 * Solve for the interface-position parameter d such that
 *   V(d) = integral_cell H(xi; n, d, beta) dxi = alpha_cell
 * using Newton iteration.
 */
static double mthinc_solve_d(const double n[3], double beta,
                              double alpha_cell, int dim)
{
    double d = 0.0;
    for (int iter = 0; iter < 30; ++iter) {
        double V   = mthinc_volume_integral(n, d, beta, dim);
        double res = V - alpha_cell;
        if (std::fabs(res) < 1e-14) break;
        double dV  = mthinc_volume_integral_dd(n, d, beta, dim);
        if (std::fabs(dV) < 1e-14) break;
        d -= res / dV;
    }
    return d;
}

/*
 * Face-averaged H at a cell face.
 *
 *   face_dir  : 0 = x, 1 = y, 2 = z
 *   face_pos  : -0.5 (low face) or +0.5 (high face) in normalised coords
 *
 * The face coordinate in face_dir is fixed; the remaining directions are
 * integrated over [-1/2, 1/2] (analytically along one, Gauss along others).
 */
static double mthinc_face_average(const double n[3], double d, double beta,
                                   int face_dir, double face_pos, int dim)
{
    if (dim == 1) {
        return 0.5 * (1.0 + std::tanh(beta * (n[0] * face_pos + d)));
    }
    else if (dim == 2) {
        int trans = 1 - face_dir;
        double a = beta * (n[face_dir] * face_pos + d);
        return thinc_integral_1d(a, beta * n[trans]);
    }
    else { /* dim == 3 */
        /* Two transverse directions */
        int t0 = -1, t1 = -1;
        for (int dd = 0; dd < 3; ++dd)
            if (dd != face_dir) { if (t0 < 0) t0 = dd; else t1 = dd; }
        /* Gauss along t0, analytical along t1 */
        double sum = 0.0;
        for (int q = 0; q < 3; ++q) {
            double a = beta * (n[face_dir] * face_pos
                             + n[t0] * GQ3_PTS[q] + d);
            sum += GQ3_WTS[q] * thinc_integral_1d(a, beta * n[t1]);
        }
        return sum;
    }
}

/*
 * Apply MTHINC to all interior cells.
 *
 * For each interface cell the routine:
 *   1. Computes the interface normal from the gradient of alpha (central
 *      differences in all dimensions, yielding the normal in cell-scaled
 *      coordinates so that non-square cells are handled correctly).
 *   2. Solves for the interface-position parameter d from conservation.
 *   3. Evaluates the face-averaged THINC function at every face of the cell
 *      and overwrites the corresponding alpha in the face arrays.
 *   4. Recomputes gammaEff / piInfEff at modified faces.
 */
static void mthinc_apply(struct ReconstructorData* r,
                          const struct SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          const struct SolutionState* state)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;
    const int dim = mesh->dim;
    const int nPhases = config->multiPhaseParams.nPhases;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    const size_t tc = state->totalCells;
    const double beta = config->mthincParams.beta;
    const double eps  = 1e-6;  /* interface detection threshold */

    /* Extend loop by one ghost cell in each direction so that ghost cells
     * at MPI boundaries also get MTHINC treatment.  This ensures both
     * processors agree on the face states at shared boundaries.
     * With nGhost >= 2, the gradient stencil (i-1, i+1) is valid for
     * ghost cells at i=-1 and i=nx.  Face writes are bounds-checked. */
    for (int k = -1; k <= nz; ++k) {
        for (int j = -1; j <= ny; ++j) {
            for (int i = -1; i <= nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                int any_modified = 0;

                for (int ph = 0; ph < nPhases; ++ph) {
                    double ac = state->alpha[ph * tc + idx];

                    /* Skip pure cells */
                    if (ac < eps || ac > 1.0 - eps) continue;

                    /* --- Interface normal in cell-scaled coordinates ---
                     * n_xi = dAlpha/dxi where xi = (x - x_c)/dx, so the
                     * aspect ratio is embedded automatically.  For a uniform
                     * mesh: n_xi = (alpha_{i+1} - alpha_{i-1}) / 2.       */
                    double n_raw[3] = {0.0, 0.0, 0.0};
                    {
                        size_t im = mesh_index(mesh, i - 1, j, k);
                        size_t ip = mesh_index(mesh, i + 1, j, k);
                        n_raw[0] = (state->alpha[ph * tc + ip]
                                  - state->alpha[ph * tc + im]) * 0.5;
                    }
                    if (dim >= 2) {
                        size_t jm = mesh_index(mesh, i, j - 1, k);
                        size_t jp = mesh_index(mesh, i, j + 1, k);
                        n_raw[1] = (state->alpha[ph * tc + jp]
                                  - state->alpha[ph * tc + jm]) * 0.5;
                    }
                    if (dim >= 3) {
                        size_t km = mesh_index(mesh, i, j, k - 1);
                        size_t kp = mesh_index(mesh, i, j, k + 1);
                        n_raw[2] = (state->alpha[ph * tc + kp]
                                  - state->alpha[ph * tc + km]) * 0.5;
                    }

                    double nmag = 0.0;
                    for (int d = 0; d < dim; ++d) nmag += n_raw[d] * n_raw[d];
                    nmag = std::sqrt(nmag);
                    if (nmag < 1e-14) continue;

                    double n_hat[3] = {0.0, 0.0, 0.0};
                    for (int d = 0; d < dim; ++d) n_hat[d] = n_raw[d] / nmag;

                    /* Solve for d from conservation */
                    double d_param = mthinc_solve_d(n_hat, beta, ac, dim);

                    /* --- Write face-averaged alpha to face arrays ---
                     *
                     * Face indexing convention:
                     *   x-face i is between cells (i-1) and (i).
                     *   xLeft[face] holds the state from cell (i-1) (its right face).
                     *   xRight[face] holds the state from cell (i) (its left face).
                     *
                     * So cell (ic, jc, kc):
                     *   right x-face -> xLeft[ x_face_index(ic+1, jc, kc) ]
                     *   left  x-face -> xRight[ x_face_index(ic,  jc, kc) ]
                     *
                     * Bounds: x-faces valid for i in [0, nx], j in [0, ny), k in [0, nz)
                     *         y-faces valid for i in [0, nx), j in [0, ny], k in [0, nz)
                     *         z-faces valid for i in [0, nx), j in [0, ny), k in [0, nz]
                     */

                    int jInY = (j >= 0 && j < ny);
                    int kInZ = (k >= 0 && k < nz);
                    int iInX = (i >= 0 && i < nx);

                    /* X faces: valid when j in [0,ny), k in [0,nz) */
                    if (jInY && kInZ) {
                        if (i + 1 >= 0 && i + 1 <= nx) {
                            size_t fR = x_face_index(r, i + 1, j, k);
                            r->xLeft[fR].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 0, 0.5, dim),
                                0.0, 1.0);
                        }
                        if (i >= 0 && i <= nx) {
                            size_t fL = x_face_index(r, i, j, k);
                            r->xRight[fL].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 0, -0.5, dim),
                                0.0, 1.0);
                        }
                    }

                    /* Y faces: valid when i in [0,nx), k in [0,nz) */
                    if (dim >= 2 && iInX && kInZ) {
                        if (j + 1 >= 0 && j + 1 <= ny) {
                            size_t fR = y_face_index(r, i, j + 1, k);
                            r->yLeft[fR].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 1, 0.5, dim),
                                0.0, 1.0);
                        }
                        if (j >= 0 && j <= ny) {
                            size_t fL = y_face_index(r, i, j, k);
                            r->yRight[fL].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 1, -0.5, dim),
                                0.0, 1.0);
                        }
                    }

                    /* Z faces: valid when i in [0,nx), j in [0,ny) */
                    if (dim >= 3 && iInX && jInY) {
                        if (k + 1 >= 0 && k + 1 <= nz) {
                            size_t fR = z_face_index(r, i, j, k + 1);
                            r->zLeft[fR].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 2, 0.5, dim),
                                0.0, 1.0);
                        }
                        if (k >= 0 && k <= nz) {
                            size_t fL = z_face_index(r, i, j, k);
                            r->zRight[fL].alpha[ph] = std::clamp(
                                mthinc_face_average(n_hat, d_param, beta, 2, -0.5, dim),
                                0.0, 1.0);
                        }
                    }

                    any_modified = 1;
                } /* end phase loop */

                /* Recompute effective EOS at all faces touched by this cell */
                if (any_modified) {
                    int jInY = (j >= 0 && j < ny);
                    int kInZ = (k >= 0 && k < nz);
                    int iInX = (i >= 0 && i < nx);

                    if (jInY && kInZ) {
                        if (i + 1 >= 0 && i + 1 <= nx) {
                            size_t fR = x_face_index(r, i + 1, j, k);
                            effective_gamma_and_pi_inf(r->xLeft[fR].alpha, nPhases, mp,
                                &r->xLeft[fR].gammaEff, &r->xLeft[fR].piInfEff);
                        }
                        if (i >= 0 && i <= nx) {
                            size_t fL = x_face_index(r, i, j, k);
                            effective_gamma_and_pi_inf(r->xRight[fL].alpha, nPhases, mp,
                                &r->xRight[fL].gammaEff, &r->xRight[fL].piInfEff);
                        }
                    }
                    if (dim >= 2 && iInX && kInZ) {
                        if (j + 1 >= 0 && j + 1 <= ny) {
                            size_t fR = y_face_index(r, i, j + 1, k);
                            effective_gamma_and_pi_inf(r->yLeft[fR].alpha, nPhases, mp,
                                &r->yLeft[fR].gammaEff, &r->yLeft[fR].piInfEff);
                        }
                        if (j >= 0 && j <= ny) {
                            size_t fL = y_face_index(r, i, j, k);
                            effective_gamma_and_pi_inf(r->yRight[fL].alpha, nPhases, mp,
                                &r->yRight[fL].gammaEff, &r->yRight[fL].piInfEff);
                        }
                    }
                    if (dim >= 3 && iInX && jInY) {
                        if (k + 1 >= 0 && k + 1 <= nz) {
                            size_t fR = z_face_index(r, i, j, k + 1);
                            effective_gamma_and_pi_inf(r->zLeft[fR].alpha, nPhases, mp,
                                &r->zLeft[fR].gammaEff, &r->zLeft[fR].piInfEff);
                        }
                        if (k >= 0 && k <= nz) {
                            size_t fL = z_face_index(r, i, j, k);
                            effective_gamma_and_pi_inf(r->zRight[fL].alpha, nPhases, mp,
                                &r->zRight[fL].gammaEff, &r->zRight[fL].piInfEff);
                        }
                    }
                }
            }
        }
    }
}

/* ---- Public API ---- */

void reconstructor_init(struct ReconstructorData* r,
                        enum ReconstructionOrder order,
                        double wenoEps, double gamma, double pInf)
{
    memset(r, 0, sizeof(struct ReconstructorData));
    r->order = order;
    r->wenoEps = wenoEps;
    r->gamma = gamma;
    r->pInf = pInf;
}

void reconstructor_allocate(struct ReconstructorData* r,
                            const struct RectilinearMesh* mesh)
{
    r->dim = mesh->dim;
    r->nx = mesh->nx;
    r->ny = mesh->ny;
    r->nz = mesh->nz;

    r->numXFaces = (size_t)(r->nx + 1) * r->ny * r->nz;
    r->xLeft  = (PrimitiveState*)calloc(r->numXFaces, sizeof(PrimitiveState));
    r->xRight = (PrimitiveState*)calloc(r->numXFaces, sizeof(PrimitiveState));

    if (r->dim >= 2) {
        r->numYFaces = (size_t)r->nx * (r->ny + 1) * r->nz;
        r->yLeft  = (PrimitiveState*)calloc(r->numYFaces, sizeof(PrimitiveState));
        r->yRight = (PrimitiveState*)calloc(r->numYFaces, sizeof(PrimitiveState));
    }

    if (r->dim >= 3) {
        r->numZFaces = (size_t)r->nx * r->ny * (r->nz + 1);
        r->zLeft  = (PrimitiveState*)calloc(r->numZFaces, sizeof(PrimitiveState));
        r->zRight = (PrimitiveState*)calloc(r->numZFaces, sizeof(PrimitiveState));
    }

    /* Device-side face arrays.  These are scratch buffers — filled each step
     * by the reconstruction kernels and consumed by the Riemann flux loop —
     * so `map(alloc:...)` is sufficient (no host copy needed). */
    PrimitiveState* xL = r->xLeft;  PrimitiveState* xR = r->xRight;
    size_t nxF = r->numXFaces;
    #pragma omp target enter data map(alloc: xL[0:nxF], xR[0:nxF])
    if (r->dim >= 2) {
        PrimitiveState* yL = r->yLeft;  PrimitiveState* yR = r->yRight;
        size_t nyF = r->numYFaces;
        #pragma omp target enter data map(alloc: yL[0:nyF], yR[0:nyF])
    }
    if (r->dim >= 3) {
        PrimitiveState* zL = r->zLeft;  PrimitiveState* zR = r->zRight;
        size_t nzF = r->numZFaces;
        #pragma omp target enter data map(alloc: zL[0:nzF], zR[0:nzF])
    }
}

void reconstructor_free(struct ReconstructorData* r) {
    PrimitiveState* xL = r->xLeft;  PrimitiveState* xR = r->xRight;
    size_t nxF = r->numXFaces;
    if (xL) {
        #pragma omp target exit data map(delete: xL[0:nxF], xR[0:nxF])
    }
    if (r->dim >= 2 && r->yLeft) {
        PrimitiveState* yL = r->yLeft;  PrimitiveState* yR = r->yRight;
        size_t nyF = r->numYFaces;
        #pragma omp target exit data map(delete: yL[0:nyF], yR[0:nyF])
    }
    if (r->dim >= 3 && r->zLeft) {
        PrimitiveState* zL = r->zLeft;  PrimitiveState* zR = r->zRight;
        size_t nzF = r->numZFaces;
        #pragma omp target exit data map(delete: zL[0:nzF], zR[0:nzF])
    }

    free(r->xLeft);  r->xLeft = NULL;
    free(r->xRight); r->xRight = NULL;
    free(r->yLeft);  r->yLeft = NULL;
    free(r->yRight); r->yRight = NULL;
    free(r->zLeft);  r->zLeft = NULL;
    free(r->zRight); r->zRight = NULL;
}

int reconstructor_required_ghost_cells(const struct ReconstructorData* r) {
    switch (r->order) {
        case WENO1:   return 1;
        case WENO3:   return 2;
        case WENO5:   return 3;
        case UPWIND1: return 1;
        case UPWIND3: return 2;
        case UPWIND5: return 3;
    }
    return 1;
}

void reconstruct(struct ReconstructorData* r,
                 const struct SimulationConfig* config,
                 const struct RectilinearMesh* mesh,
                 const struct SolutionState* state)
{
    NVTX_PUSH("Reconstruction");
    assert(config->nGhost >= reconstructor_required_ghost_cells(r));
    NVTX_PUSH("Recon::X");
    reconstructX(r, config, mesh, state);
    NVTX_POP();
    if (r->dim >= 2) {
        NVTX_PUSH("Recon::Y");
        reconstructY(r, config, mesh, state);
        NVTX_POP();
    }
    if (r->dim >= 3) {
        NVTX_PUSH("Recon::Z");
        reconstructZ(r, config, mesh, state);
        NVTX_POP();
    }

    /* MTHINC: overwrite alpha face values with multi-dimensional THINC
     * reconstruction at interface cells */
    if (config->mthincParams.enabled && config_is_multi_phase(config)) {
        NVTX_PUSH("MTHINC");
        mthinc_apply(r, config, mesh, state);
        NVTX_POP();
    }

    NVTX_POP();
}
