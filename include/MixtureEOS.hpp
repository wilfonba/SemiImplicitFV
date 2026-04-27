#ifndef MIXTURE_EOS_HPP
#define MIXTURE_EOS_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

#pragma omp declare target
/* Effective gamma from volume fractions:
   1/(g_mix - 1) = sum(alpha_k / (g_k - 1)) */
double effective_gamma(const double* alphas, int nPhases,
                       const struct MultiPhaseParams* mp);

/* Compute both gammaEff and piInfEff from raw alpha array. */
void effective_gamma_and_pi_inf(const double* alphas, int nPhases,
                                const struct MultiPhaseParams* mp,
                                double* gammaEff, double* piInfEff);

/* Mixture pressure from internal energy and volume fractions:
   p = (rhoE_internal - sum(alpha_k * g_k * pInf_k / (g_k - 1))) / sum(alpha_k / (g_k - 1)) */
double mixture_pressure(double rhoE_internal,
                        const double* alphas, int nPhases,
                        const struct PhaseEOS* phases);

/* Wood's mixture sound speed:
   1/(rho*c^2) = sum(alpha_k / (rho_k * c_k^2)) */
double mixture_sound_speed(double rho, double p,
                           const double* alphas,
                           const double* alphaRhos,
                           int nPhases, const struct PhaseEOS* phases);

/* Mixture total energy from pressure:
   rhoE = sum(alpha_k * (p + g_k * pInf_k) / (g_k - 1)) + ke */
double mixture_total_energy(double rho, double p,
                            const double* alphas, int nPhases,
                            double ke, const struct PhaseEOS* phases);

/* Device-friendly variants taking flat double arrays for per-phase gamma /
 * pInf instead of a host PhaseEOS struct pointer.  GPU kernels can map
 * these arrays via map(to:) without needing to map the SimulationConfig
 * struct itself. */
static inline void effective_gamma_and_pi_inf_arr(const double* alphas,
                                                  int nPhases,
                                                  const double* phaseGamma,
                                                  const double* phasePinf,
                                                  double* gammaEff,
                                                  double* piInfEff) {
    double sumInvGm1 = 0.0;
    double sumPiInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gk = phaseGamma[ph];
        double gm1 = gk - 1.0;
        sumInvGm1   += alphas[ph] / gm1;
        sumPiInfTerm += alphas[ph] * gk * phasePinf[ph] / gm1;
    }
    *gammaEff = 1.0 + 1.0 / sumInvGm1;
    *piInfEff = (*gammaEff - 1.0) / *gammaEff * sumPiInfTerm;
}

static inline double mixture_pressure_arr(double rhoE_internal,
                                          const double* alphas, int nPhases,
                                          const double* phaseGamma,
                                          const double* phasePinf) {
    double sumInvGm1 = 0.0;
    double sumPInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gm1 = phaseGamma[ph] - 1.0;
        sumInvGm1   += alphas[ph] / gm1;
        sumPInfTerm += alphas[ph] * phaseGamma[ph] * phasePinf[ph] / gm1;
    }
    return (rhoE_internal - sumPInfTerm) / sumInvGm1;
}
#pragma omp end declare target

/* Full mesh loop: conservative -> primitive for multi-phase */
void mixture_cons_to_prim(const struct RectilinearMesh* mesh,
                          struct SolutionState* state,
                          const struct MultiPhaseParams* mp);

/* Device-resident variant of mixture_cons_to_prim — runs on GPU. */
void mixture_cons_to_prim_device(const struct RectilinearMesh* mesh,
                                 struct SolutionState* state,
                                 const struct MultiPhaseParams* mp);

/* Full mesh loop: primitive -> conservative for multi-phase */
void mixture_prim_to_cons(const struct RectilinearMesh* mesh,
                          struct SolutionState* state,
                          const struct MultiPhaseParams* mp);

#endif /* MIXTURE_EOS_HPP */
