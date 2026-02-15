#include "MixtureEOS.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {
namespace MixtureEOS {

double effectiveGamma(const std::vector<double>& alphas,
                      const MultiPhaseParams& mp) {
    int nPhases = mp.nPhases;
    double sumInvGm1 = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        sumInvGm1 += alphas[ph] / (mp.phases[ph].gamma - 1.0);
    }
    return 1.0 + 1.0 / sumInvGm1;
}

void effectiveGammaAndPiInf(const double* alphas, int nAlphas,
                             const MultiPhaseParams& mp,
                             double& gammaEff, double& piInfEff) {
    double sumInvGm1 = 0.0;
    double sumPiInfTerm = 0.0;
    for (int ph = 0; ph < nAlphas; ++ph) {
        double gk = mp.phases[ph].gamma;
        double gm1 = gk - 1.0;
        sumInvGm1 += alphas[ph] / gm1;
        sumPiInfTerm += alphas[ph] * gk * mp.phases[ph].pInf / gm1;
    }
    gammaEff = 1.0 + 1.0 / sumInvGm1;
    piInfEff = (gammaEff - 1.0) / gammaEff * sumPiInfTerm;
}

// ---- Raw-pointer implementations (GPU-ready) ----

double mixturePressure(double rhoE_internal,
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

double mixtureSoundSpeed(double rho, double p,
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

double mixtureTotalEnergy(double /*rho*/, double p,
                          const double* alphas, int nPhases,
                          double ke, const PhaseEOS* phases) {
    double result = ke;
    for (int ph = 0; ph < nPhases; ++ph) {
        double gm1 = phases[ph].gamma - 1.0;
        result += alphas[ph] * (p + phases[ph].gamma * phases[ph].pInf) / gm1;
    }
    return result;
}

// ---- std::vector wrappers ----

double mixturePressure(double rhoE_internal,
                       const std::vector<double>& alphas,
                       const MultiPhaseParams& mp) {
    return mixturePressure(rhoE_internal, alphas.data(), mp.nPhases, mp.phases.data());
}

double mixtureSoundSpeed(double rho, double p,
                         const std::vector<double>& alphas,
                         const std::vector<double>& alphaRhos,
                         const MultiPhaseParams& mp) {
    return mixtureSoundSpeed(rho, p, alphas.data(), alphaRhos.data(),
                             mp.nPhases, mp.phases.data());
}

double mixtureTotalEnergy(double rho, double p,
                          const std::vector<double>& alphas,
                          double ke,
                          const MultiPhaseParams& mp) {
    return mixtureTotalEnergy(rho, p, alphas.data(), mp.nPhases, ke, mp.phases.data());
}

void convertConservativeToPrimitive(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp) {
    int dim = state.dim();
    int nPhases = mp.nPhases;

    // Pre-allocate outside the loop to avoid per-cell heap allocations
    std::vector<double> alphas(nPhases);

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                // Density from sum of partial densities
                double rho = 0.0;
                for (int ph = 0; ph < nPhases; ++ph)
                    rho += state.alphaRho[ph][idx];
                state.rho[idx] = std::max(rho, 1e-14);

                // Velocity
                double rhoSafe = state.rho[idx];
                state.velU[idx] = state.rhoU[idx] / rhoSafe;
                if (dim >= 2) state.velV[idx] = state.rhoV[idx] / rhoSafe;
                if (dim >= 3) state.velW[idx] = state.rhoW[idx] / rhoSafe;

                // Kinetic energy
                double ke = 0.5 * rhoSafe * state.velU[idx] * state.velU[idx];
                if (dim >= 2) ke += 0.5 * rhoSafe * state.velV[idx] * state.velV[idx];
                if (dim >= 3) ke += 0.5 * rhoSafe * state.velW[idx] * state.velW[idx];

                // Gather volume fractions
                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = state.alpha[ph][idx];

                // Internal energy
                double rhoE_internal = state.rhoE[idx] - ke;

                // Pressure
                double p = mixturePressure(rhoE_internal, alphas, mp);
                state.pres[idx] = p;
            }
        }
    }
}

void convertPrimitiveToConservative(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp) {
    int dim = state.dim();
    int nPhases = mp.nPhases;

    // Pre-allocate outside the loop to avoid per-cell heap allocations
    std::vector<double> alphas(nPhases);

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                double rho = state.rho[idx];
                state.rhoU[idx] = rho * state.velU[idx];
                if (dim >= 2) state.rhoV[idx] = rho * state.velV[idx];
                if (dim >= 3) state.rhoW[idx] = rho * state.velW[idx];

                double ke = 0.5 * rho * state.velU[idx] * state.velU[idx];
                if (dim >= 2) ke += 0.5 * rho * state.velV[idx] * state.velV[idx];
                if (dim >= 3) ke += 0.5 * rho * state.velW[idx] * state.velW[idx];

                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = state.alpha[ph][idx];

                state.rhoE[idx] = mixtureTotalEnergy(rho, state.pres[idx],
                                                      alphas, ke, mp);
            }
        }
    }
}

} // namespace MixtureEOS
} // namespace SemiImplicitFV
