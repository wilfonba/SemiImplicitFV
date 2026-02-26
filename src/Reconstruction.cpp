#include "Reconstruction.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "MixtureEOS.hpp"
#include "NvtxRange.hpp"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* No using declarations needed - all functions are free functions */

/* ---- Reconstruction function pointer type ---- */
typedef double (*ReconFn)(const double*, double);

static void reconstructScalar(
    const double* field,
    const size_t* cells,
    int stencilSize,
    ReconFn leftFn,
    ReconFn rightFn,
    double eps,
    double* outLeft,
    double* outRight)
{
    double vL[5], vR[5];
    for (int s = 0; s < stencilSize; ++s) {
        vL[s] = field[cells[s]];
        vR[s] = field[cells[s + 1]];
    }
    *outLeft  = leftFn(vL, eps);
    *outRight = rightFn(vR, eps);
}

/* ---- WENO / Upwind stencil functions ---- */

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

/* ---- Helper to zero-initialize a PrimitiveState ---- */
static void zero_primitive(PrimitiveState* s) {
    memset(s, 0, sizeof(PrimitiveState));
}

/* ---- Direction-specific reconstruction sweeps ---- */

static void reconstructX(struct ReconstructorData* r,
                          const struct SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          const struct SolutionState* state)
{
    const int nx = mesh->nx;
    const int ny = mesh->ny;
    const int nz = mesh->nz;

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* pres = state->pres;
    const double* sig  = state->sigma;
    const double* velV = (r->dim >= 2) ? state->velV : NULL;
    const double* velW = (r->dim >= 3) ? state->velW : NULL;

    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    const size_t totalCells = state->totalCells;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                size_t fIdx = x_face_index(r, i, j, k);
                PrimitiveState* left  = &r->xLeft[fIdx];
                PrimitiveState* right = &r->xRight[fIdx];
                zero_primitive(left);
                zero_primitive(right);

                if (r->order == WENO1 || r->order == UPWIND1) {
                    size_t idxL = mesh_index(mesh,i - 1, j, k);
                    size_t idxR = mesh_index(mesh,i, j, k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->p     = pres[idxL];
                    left->sigma = sig[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->p     = pres[idxR];
                    right->sigma = sig[idxR];
                    if (r->dim >= 2) {
                        left->u[1]  = velV[idxL];
                        right->u[1] = velV[idxR];
                    }
                    if (r->dim >= 3) {
                        left->u[2]  = velW[idxL];
                        right->u[2] = velW[idxR];
                    }
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph] = state->alpha[ph * totalCells + idxL];
                            right->alpha[ph] = state->alpha[ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else if (r->order == WENO3 || r->order == UPWIND3) {
                    size_t c[4];
                    c[0] = mesh_index(mesh,i - 2, j, k);
                    c[1] = mesh_index(mesh,i - 1, j, k);
                    c[2] = mesh_index(mesh,i,     j, k);
                    c[3] = mesh_index(mesh,i + 1, j, k);

                    ReconFn lFn = (r->order == WENO3) ? weno3Left  : upwind3Left;
                    ReconFn rFn = (r->order == WENO3) ? weno3Right : upwind3Right;
                    reconstructScalar(rho,  c, 3, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 3, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(pres, c, 3, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 3, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (r->dim >= 2)
                        reconstructScalar(velV, c, 3, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    if (r->dim >= 3)
                        reconstructScalar(velW, c, 3, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 3, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = mesh_index(mesh,i - 3, j, k);
                    c[1] = mesh_index(mesh,i - 2, j, k);
                    c[2] = mesh_index(mesh,i - 1, j, k);
                    c[3] = mesh_index(mesh,i,     j, k);
                    c[4] = mesh_index(mesh,i + 1, j, k);
                    c[5] = mesh_index(mesh,i + 2, j, k);

                    ReconFn lFn = (r->order == WENO5) ? weno5Left  : upwind5Left;
                    ReconFn rFn = (r->order == WENO5) ? weno5Right : upwind5Right;
                    reconstructScalar(rho,  c, 5, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 5, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(pres, c, 5, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 5, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (r->dim >= 2)
                        reconstructScalar(velV, c, 5, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    if (r->dim >= 3)
                        reconstructScalar(velW, c, 5, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 5, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
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

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* velV = state->velV;
    const double* pres = state->pres;
    const double* sig  = state->sigma;
    const double* velW = (r->dim >= 3) ? state->velW : NULL;

    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    const size_t totalCells = state->totalCells;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fIdx = y_face_index(r, i, j, k);
                PrimitiveState* left  = &r->yLeft[fIdx];
                PrimitiveState* right = &r->yRight[fIdx];
                zero_primitive(left);
                zero_primitive(right);

                if (r->order == WENO1 || r->order == UPWIND1) {
                    size_t idxL = mesh_index(mesh,i, j - 1, k);
                    size_t idxR = mesh_index(mesh,i, j, k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->u[1]  = velV[idxL];
                    left->p     = pres[idxL];
                    left->sigma = sig[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->u[1]  = velV[idxR];
                    right->p     = pres[idxR];
                    right->sigma = sig[idxR];
                    if (r->dim >= 3) {
                        left->u[2]  = velW[idxL];
                        right->u[2] = velW[idxR];
                    }
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph] = state->alpha[ph * totalCells + idxL];
                            right->alpha[ph] = state->alpha[ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else if (r->order == WENO3 || r->order == UPWIND3) {
                    size_t c[4];
                    c[0] = mesh_index(mesh,i, j - 2, k);
                    c[1] = mesh_index(mesh,i, j - 1, k);
                    c[2] = mesh_index(mesh,i, j,     k);
                    c[3] = mesh_index(mesh,i, j + 1, k);

                    ReconFn lFn = (r->order == WENO3) ? weno3Left  : upwind3Left;
                    ReconFn rFn = (r->order == WENO3) ? weno3Right : upwind3Right;
                    reconstructScalar(rho,  c, 3, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 3, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(velV, c, 3, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    reconstructScalar(pres, c, 3, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 3, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (r->dim >= 3)
                        reconstructScalar(velW, c, 3, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 3, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = mesh_index(mesh,i, j - 3, k);
                    c[1] = mesh_index(mesh,i, j - 2, k);
                    c[2] = mesh_index(mesh,i, j - 1, k);
                    c[3] = mesh_index(mesh,i, j,     k);
                    c[4] = mesh_index(mesh,i, j + 1, k);
                    c[5] = mesh_index(mesh,i, j + 2, k);

                    ReconFn lFn = (r->order == WENO5) ? weno5Left  : upwind5Left;
                    ReconFn rFn = (r->order == WENO5) ? weno5Right : upwind5Right;
                    reconstructScalar(rho,  c, 5, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 5, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(velV, c, 5, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    reconstructScalar(pres, c, 5, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 5, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (r->dim >= 3)
                        reconstructScalar(velW, c, 5, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 5, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
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

    const double* rho  = state->rho;
    const double* velU = state->velU;
    const double* velV = state->velV;
    const double* velW = state->velW;
    const double* pres = state->pres;
    const double* sig  = state->sigma;

    const int multiPhase = config_is_multi_phase(config);
    const int nAlphas = multiPhase ? config->multiPhaseParams.nPhases : 0;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    const size_t totalCells = state->totalCells;

    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fIdx = z_face_index(r, i, j, k);
                PrimitiveState* left  = &r->zLeft[fIdx];
                PrimitiveState* right = &r->zRight[fIdx];
                zero_primitive(left);
                zero_primitive(right);

                if (r->order == WENO1 || r->order == UPWIND1) {
                    size_t idxL = mesh_index(mesh,i, j, k - 1);
                    size_t idxR = mesh_index(mesh,i, j, k);
                    left->rho   = rho[idxL];
                    left->u[0]  = velU[idxL];
                    left->u[1]  = velV[idxL];
                    left->u[2]  = velW[idxL];
                    left->p     = pres[idxL];
                    left->sigma = sig[idxL];
                    right->rho   = rho[idxR];
                    right->u[0]  = velU[idxR];
                    right->u[1]  = velV[idxR];
                    right->u[2]  = velW[idxR];
                    right->p     = pres[idxR];
                    right->sigma = sig[idxR];
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph) {
                            left->alpha[ph] = state->alpha[ph * totalCells + idxL];
                            right->alpha[ph] = state->alpha[ph * totalCells + idxR];
                        }
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else if (r->order == WENO3 || r->order == UPWIND3) {
                    size_t c[4];
                    c[0] = mesh_index(mesh,i, j, k - 2);
                    c[1] = mesh_index(mesh,i, j, k - 1);
                    c[2] = mesh_index(mesh,i, j, k);
                    c[3] = mesh_index(mesh,i, j, k + 1);

                    ReconFn lFn = (r->order == WENO3) ? weno3Left  : upwind3Left;
                    ReconFn rFn = (r->order == WENO3) ? weno3Right : upwind3Right;
                    reconstructScalar(rho,  c, 3, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 3, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(velV, c, 3, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    reconstructScalar(velW, c, 3, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    reconstructScalar(pres, c, 3, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 3, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 3, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
                    }
                }
                else { /* WENO5 or UPWIND5 */
                    size_t c[6];
                    c[0] = mesh_index(mesh,i, j, k - 3);
                    c[1] = mesh_index(mesh,i, j, k - 2);
                    c[2] = mesh_index(mesh,i, j, k - 1);
                    c[3] = mesh_index(mesh,i, j, k);
                    c[4] = mesh_index(mesh,i, j, k + 1);
                    c[5] = mesh_index(mesh,i, j, k + 2);

                    ReconFn lFn = (r->order == WENO5) ? weno5Left  : upwind5Left;
                    ReconFn rFn = (r->order == WENO5) ? weno5Right : upwind5Right;
                    reconstructScalar(rho,  c, 5, lFn, rFn, r->wenoEps, &left->rho,  &right->rho);
                    reconstructScalar(velU, c, 5, lFn, rFn, r->wenoEps, &left->u[0], &right->u[0]);
                    reconstructScalar(velV, c, 5, lFn, rFn, r->wenoEps, &left->u[1], &right->u[1]);
                    reconstructScalar(velW, c, 5, lFn, rFn, r->wenoEps, &left->u[2], &right->u[2]);
                    reconstructScalar(pres, c, 5, lFn, rFn, r->wenoEps, &left->p,    &right->p);
                    reconstructScalar(sig,  c, 5, lFn, rFn, r->wenoEps, &left->sigma, &right->sigma);
                    if (multiPhase) {
                        for (int ph = 0; ph < nAlphas; ++ph)
                            reconstructScalar(state->alpha + ph * totalCells, c, 5, lFn, rFn, r->wenoEps, &left->alpha[ph], &right->alpha[ph]);
                        effective_gamma_and_pi_inf(left->alpha, nAlphas, mp, &left->gammaEff, &left->piInfEff);
                        effective_gamma_and_pi_inf(right->alpha, nAlphas, mp, &right->gammaEff, &right->piInfEff);
                    } else {
                        left->gammaEff = r->gamma;   left->piInfEff = r->pInf;
                        right->gammaEff = r->gamma;  right->piInfEff = r->pInf;
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
}

void reconstructor_free(struct ReconstructorData* r) {
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
    reconstructX(r, config, mesh, state);
    if (r->dim >= 2) reconstructY(r, config, mesh, state);
    if (r->dim >= 3) reconstructZ(r, config, mesh, state);
    NVTX_POP();
}
