#ifndef MIXTURE_EOS_HPP
#define MIXTURE_EOS_HPP

#include "SimulationConfig.hpp"
#include "SolutionState.hpp"
#include "RectilinearMesh.hpp"
#include <vector>

namespace SemiImplicitFV {
namespace MixtureEOS {

// Effective gamma from volume fractions:
//   1/(g_mix - 1) = sum(alpha_k / (g_k - 1))
double effectiveGamma(const std::vector<double>& alphas,
                      const MultiPhaseParams& mp);

// Allocation-free version: compute both gammaEff and piInfEff from raw alpha array.
// alphas has nAlphas = nPhases entries (all N volume fractions).
// Math: 1/(g_mix-1) = sum(alpha_k/(g_k-1))
//       piInf_mix = (g_mix-1)/g_mix * sum(alpha_k * g_k * pInf_k / (g_k-1))
void effectiveGammaAndPiInf(const double* alphas, int nAlphas,
                             const MultiPhaseParams& mp,
                             double& gammaEff, double& piInfEff);

// Mixture pressure from internal energy and volume fractions:
//   p = (rho*e - sum(alpha_k * g_k * pInf_k / (g_k - 1))) / sum(alpha_k / (g_k - 1))
double mixturePressure(double rhoE_internal,
                       const std::vector<double>& alphas,
                       const MultiPhaseParams& mp);

// Raw-pointer overload (GPU-ready, no STL)
double mixturePressure(double rhoE_internal,
                       const double* alphas, int nPhases,
                       const PhaseEOS* phases);

// Wood's mixture sound speed:
//   1/(rho*c^2) = sum(alpha_k / (rho_k * c_k^2))
double mixtureSoundSpeed(double rho, double p,
                         const std::vector<double>& alphas,
                         const std::vector<double>& alphaRhos,
                         const MultiPhaseParams& mp);

// Raw-pointer overload (GPU-ready, no STL)
double mixtureSoundSpeed(double rho, double p,
                         const double* alphas,
                         const double* alphaRhos,
                         int nPhases, const PhaseEOS* phases);

// Mixture total energy from pressure:
//   rhoE = sum(alpha_k * (p + g_k * pInf_k) / (g_k - 1)) + ke
double mixtureTotalEnergy(double rho, double p,
                          const std::vector<double>& alphas,
                          double ke,
                          const MultiPhaseParams& mp);

// Raw-pointer overload (GPU-ready, no STL)
double mixtureTotalEnergy(double rho, double p,
                          const double* alphas, int nPhases,
                          double ke, const PhaseEOS* phases);

// Full mesh loop: conservative -> primitive for multi-phase
void convertConservativeToPrimitive(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp);

// Full mesh loop: primitive -> conservative for multi-phase
void convertPrimitiveToConservative(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp);

} // namespace MixtureEOS
} // namespace SemiImplicitFV

#endif // MIXTURE_EOS_HPP
