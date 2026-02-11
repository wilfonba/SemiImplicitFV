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
        double a = (ph < nPhases - 1) ? alphas[ph]
                 : 1.0 - [&]() { double s = 0.0; for (int q = 0; q < nPhases - 1; ++q) s += alphas[q]; return s; }();
        sumInvGm1 += a / (mp.phases[ph].gamma - 1.0);
    }
    return 1.0 + 1.0 / sumInvGm1;
}

void effectiveGammaAndPiInf(const double* alphas, int nAlphas,
                             const MultiPhaseParams& mp,
                             double& gammaEff, double& piInfEff) {
    int nPhases = mp.nPhases;
    double alphaSum = 0.0;
    for (int ph = 0; ph < nAlphas; ++ph)
        alphaSum += alphas[ph];
    double alphaN = 1.0 - alphaSum;

    double sumInvGm1 = 0.0;
    double sumPiInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double a = (ph < nAlphas) ? alphas[ph] : alphaN;
        double gk = mp.phases[ph].gamma;
        double gm1 = gk - 1.0;
        sumInvGm1 += a / gm1;
        sumPiInfTerm += a * gk * mp.phases[ph].pInf / gm1;
    }
    gammaEff = 1.0 + 1.0 / sumInvGm1;
    // piInf_mix = (g_mix-1)/g_mix * sum(alpha_k * g_k * pInf_k / (g_k-1))
    piInfEff = (gammaEff - 1.0) / gammaEff * sumPiInfTerm;
}

double mixturePressure(double rhoE_internal,
                       const std::vector<double>& alphas,
                       const MultiPhaseParams& mp) {
    int nPhases = mp.nPhases;

    // Compute alpha_N from constraint
    double alphaSum = 0.0;
    for (int ph = 0; ph < nPhases - 1; ++ph)
        alphaSum += alphas[ph];
    double alphaN = 1.0 - alphaSum;

    double sumInvGm1 = 0.0;
    double sumPInfTerm = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double a = (ph < nPhases - 1) ? alphas[ph] : alphaN;
        double gm1 = mp.phases[ph].gamma - 1.0;
        sumInvGm1 += a / gm1;
        sumPInfTerm += a * mp.phases[ph].gamma * mp.phases[ph].pInf / gm1;
    }

    return (rhoE_internal - sumPInfTerm) / sumInvGm1;
}

double mixtureSoundSpeed(double rho, double p,
                         const std::vector<double>& alphas,
                         const std::vector<double>& alphaRhos,
                         const MultiPhaseParams& mp) {
    int nPhases = mp.nPhases;

    double alphaSum = 0.0;
    for (int ph = 0; ph < nPhases - 1; ++ph)
        alphaSum += alphas[ph];
    double alphaN = 1.0 - alphaSum;

    double sumInvRhoc2 = 0.0;
    for (int ph = 0; ph < nPhases; ++ph) {
        double a = (ph < nPhases - 1) ? alphas[ph] : alphaN;
        double rho_k = std::max(alphaRhos[ph], 1e-14) / std::max(a, 1e-14);
        double gk = mp.phases[ph].gamma;
        double pInfk = mp.phases[ph].pInf;
        double ck2 = gk * (p + pInfk) / rho_k;
        sumInvRhoc2 += a / (rho_k * std::max(ck2, 1e-14));
    }

    double c2 = 1.0 / (rho * std::max(sumInvRhoc2, 1e-30));
    return std::sqrt(std::max(c2, 0.0));
}

double mixtureTotalEnergy(double /*rho*/, double p,
                          const std::vector<double>& alphas,
                          double ke,
                          const MultiPhaseParams& mp) {
    int nPhases = mp.nPhases;

    double alphaSum = 0.0;
    for (int ph = 0; ph < nPhases - 1; ++ph)
        alphaSum += alphas[ph];
    double alphaN = 1.0 - alphaSum;

    double result = ke;
    for (int ph = 0; ph < nPhases; ++ph) {
        double a = (ph < nPhases - 1) ? alphas[ph] : alphaN;
        double gm1 = mp.phases[ph].gamma - 1.0;
        result += a * (p + mp.phases[ph].gamma * mp.phases[ph].pInf) / gm1;
    }
    return result;
}

void convertConservativeToPrimitive(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp) {
    int dim = state.dim();
    int nPhases = mp.nPhases;

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
                std::vector<double> alphas(nPhases - 1);
                for (int ph = 0; ph < nPhases - 1; ++ph)
                    alphas[ph] = state.alpha[ph][idx];

                // Internal energy
                double rhoE_internal = state.rhoE[idx] - ke;

                // Pressure
                double p = mixturePressure(rhoE_internal, alphas, mp);
                state.pres[idx] = p;
                state.temp[idx] = 0.0; // Temperature not well-defined for mixture
            }
        }
    }
}

void convertPrimitiveToConservative(const RectilinearMesh& mesh,
                                    SolutionState& state,
                                    const MultiPhaseParams& mp) {
    int dim = state.dim();
    int nPhases = mp.nPhases;

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

                std::vector<double> alphas(nPhases - 1);
                for (int ph = 0; ph < nPhases - 1; ++ph)
                    alphas[ph] = state.alpha[ph][idx];

                state.rhoE[idx] = mixtureTotalEnergy(rho, state.pres[idx],
                                                      alphas, ke, mp);
            }
        }
    }
}

} // namespace MixtureEOS
} // namespace SemiImplicitFV
