#include "MixtureEOS.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <cmath>
#include <algorithm>

#pragma omp declare target
double effective_gamma(const double* alphas, int nPhases,
                       const MultiPhaseParams* mp) {
    double sumInvGm1 = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        sumInvGm1 += alphas[ph] / (mp->phases[ph].gamma - 1.0);
    }
    return 1.0 + 1.0 / sumInvGm1;
}

void effective_gamma_and_pi_inf(const double* alphas, int nPhases,
                                const MultiPhaseParams* mp,
                                double* gammaEff, double* piInfEff) {
    double sumInvGm1 = 0.0;
    double sumPiInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gk = mp->phases[ph].gamma;
        double gm1 = gk - 1.0;
        sumInvGm1 += alphas[ph] / gm1;
        sumPiInfTerm += alphas[ph] * gk * mp->phases[ph].pInf / gm1;
    }
    *gammaEff = 1.0 + 1.0 / sumInvGm1;
    *piInfEff = (*gammaEff - 1.0) / *gammaEff * sumPiInfTerm;
}

double mixture_pressure(double rhoE_internal,
                        const double* alphas, int nPhases,
                        const PhaseEOS* phases) {
    double sumInvGm1 = 0.0;
    double sumPInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gm1 = phases[ph].gamma - 1.0;
        sumInvGm1 += alphas[ph] / gm1;
        sumPInfTerm += alphas[ph] * phases[ph].gamma * phases[ph].pInf / gm1;
    }
    return (rhoE_internal - sumPInfTerm) / sumInvGm1;
}

double mixture_sound_speed(double rho, double p,
                           const double* alphas,
                           const double* alphaRhos,
                           int nPhases, const PhaseEOS* phases) {
    double sumInvRhoc2 = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double a = alphas[ph];
        double rho_k = std::max(alphaRhos[ph], 1e-14) / std::max(a, 1e-14);
        double gk = phases[ph].gamma;
        double pInfk = phases[ph].pInf;
        double ck2 = gk * (p + pInfk) / rho_k;
        sumInvRhoc2 += a / (rho_k * std::max(ck2, 1e-14));
    }
    double c2 = 1.0 / (rho * std::max(sumInvRhoc2, 1e-30));
    return std::sqrt(std::max(c2, 0.0));
}

double mixture_total_energy(double /*rho*/, double p,
                            const double* alphas, int nPhases,
                            double ke, const PhaseEOS* phases) {
    double result = ke;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gm1 = phases[ph].gamma - 1.0;
        result += alphas[ph] * (p + phases[ph].gamma * phases[ph].pInf) / gm1;
    }
    return result;
}
#pragma omp end declare target

void mixture_cons_to_prim(const RectilinearMesh* mesh,
                          SolutionState* state,
                          const MultiPhaseParams* mp) {
    int dim = state->dim;
    int nPhases = mp->nPhases;
    size_t tc = state->totalCells;

    /* Pre-allocate scratch outside the loop */
    double alphas[MAX_PHASES];

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                /* Density from sum of partial densities */
                double rho = 0.0;
                for (int ph = 0; ph < nPhases; ++ph)
                    rho += state->alphaRho[ph * tc + idx];
                state->rho[idx] = std::max(rho, 1e-14);

                /* Velocity */
                double rhoSafe = state->rho[idx];
                state->velU[idx] = state->rhoU[idx] / rhoSafe;
                if (dim >= 2) state->velV[idx] = state->rhoV[idx] / rhoSafe;
                if (dim >= 3) state->velW[idx] = state->rhoW[idx] / rhoSafe;

                /* Kinetic energy */
                double ke = 0.5 * rhoSafe * state->velU[idx] * state->velU[idx];
                if (dim >= 2) ke += 0.5 * rhoSafe * state->velV[idx] * state->velV[idx];
                if (dim >= 3) ke += 0.5 * rhoSafe * state->velW[idx] * state->velW[idx];

                /* Gather volume fractions */
                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = state->alpha[ph * tc + idx];

                /* Internal energy */
                double rhoE_internal = state->rhoE[idx] - ke;

                /* Pressure */
                double p = mixture_pressure(rhoE_internal, alphas, nPhases, mp->phases);
                state->pres[idx] = p;
            }
        }
    }
}

void mixture_cons_to_prim_device(const RectilinearMesh* mesh,
                                 SolutionState* state,
                                 const MultiPhaseParams* mp)
{
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = state->dim;
    const int nPhases = mp->nPhases;
    const size_t tc = state->totalCells;
    double* rho   = state->rho;
    double* rhoU  = state->rhoU;
    double* rhoV  = state->rhoV;
    double* rhoW  = state->rhoW;
    double* rhoE  = state->rhoE;
    double* velU  = state->velU;
    double* velV  = state->velV;
    double* velW  = state->velW;
    double* pres  = state->pres;
    double* alpha = state->alpha;
    double* alphaRho = state->alphaRho;

    double phaseGamma[MAX_PHASES], phasePinf[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph) { phaseGamma[ph] = 0.0; phasePinf[ph] = 0.0; }
    for (int ph = 0; ph < nPhases; ++ph) {
        phaseGamma[ph] = mp->phases[ph].gamma;
        phasePinf[ph]  = mp->phases[ph].pInf;
    }

    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: phaseGamma[0:MAX_PHASES], phasePinf[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                /* rho = sum(alphaRho[ph]) */
                double rhoSum = 0.0;
                for (int ph = 0; ph < nPhases; ++ph)
                    rhoSum += alphaRho[(size_t)ph * tc + idx];
                if (rhoSum < 1e-14) rhoSum = 1e-14;
                rho[idx] = rhoSum;

                double u = rhoU[idx] / rhoSum;
                double v = (dim >= 2) ? rhoV[idx] / rhoSum : 0.0;
                double w = (dim >= 3) ? rhoW[idx] / rhoSum : 0.0;
                velU[idx] = u;
                if (dim >= 2) velV[idx] = v;
                if (dim >= 3) velW[idx] = w;

                double ke = 0.5 * rhoSum * (u * u + v * v + w * w);
                double rhoE_internal = rhoE[idx] - ke;

                double alphas[MAX_PHASES];
                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = alpha[(size_t)ph * tc + idx];
                pres[idx] = mixture_pressure_arr(rhoE_internal, alphas, nPhases,
                                                 phaseGamma, phasePinf);
            }
        }
    }
}

void mixture_prim_to_cons(const RectilinearMesh* mesh,
                          SolutionState* state,
                          const MultiPhaseParams* mp) {
    int dim = state->dim;
    int nPhases = mp->nPhases;
    size_t tc = state->totalCells;

    double alphas[MAX_PHASES];

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double rho = state->rho[idx];
                state->rhoU[idx] = rho * state->velU[idx];
                if (dim >= 2) state->rhoV[idx] = rho * state->velV[idx];
                if (dim >= 3) state->rhoW[idx] = rho * state->velW[idx];

                double ke = 0.5 * rho * state->velU[idx] * state->velU[idx];
                if (dim >= 2) ke += 0.5 * rho * state->velV[idx] * state->velV[idx];
                if (dim >= 3) ke += 0.5 * rho * state->velW[idx] * state->velW[idx];

                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = state->alpha[ph * tc + idx];

                state->rhoE[idx] = mixture_total_energy(rho, state->pres[idx],
                                                        alphas, nPhases, ke, mp->phases);
            }
        }
    }
}
