#ifndef MIXTURE_EOS_HPP
#define MIXTURE_EOS_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

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

/* Full mesh loop: conservative -> primitive for multi-phase */
void mixture_cons_to_prim(const struct RectilinearMesh* mesh,
                          struct SolutionState* state,
                          const struct MultiPhaseParams* mp);

/* Full mesh loop: primitive -> conservative for multi-phase */
void mixture_prim_to_cons(const struct RectilinearMesh* mesh,
                          struct SolutionState* state,
                          const struct MultiPhaseParams* mp);

#endif /* MIXTURE_EOS_HPP */
