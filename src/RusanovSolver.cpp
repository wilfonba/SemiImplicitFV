#include "RusanovSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux RusanovSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    RiemannFlux flux;
    const int dim_ = config_.dim;
    const bool includePressure_ = !config_.semiImplicit;

    double uL = normalVelocity(left, normal, dim_);
    double uR = normalVelocity(right, normal, dim_);

    // Maximum wave speed
    double sMax = maxWaveSpeed(left, right, normal);

    // Conservative total energies (per unit volume)
    double rhoEL, rhoER;
    if (left.gammaEff > 0.0) {
        double keL = 0.5 * left.rho * (left.u[0]*left.u[0] + left.u[1]*left.u[1] + left.u[2]*left.u[2]);
        rhoEL = (left.p + left.gammaEff * left.piInfEff) / (left.gammaEff - 1.0) + keL;
    } else {
        rhoEL = left.rho * eos_->totalEnergy(left);
    }
    if (right.gammaEff > 0.0) {
        double keR = 0.5 * right.rho * (right.u[0]*right.u[0] + right.u[1]*right.u[1] + right.u[2]*right.u[2]);
        rhoER = (right.p + right.gammaEff * right.piInfEff) / (right.gammaEff - 1.0) + keR;
    } else {
        rhoER = right.rho * eos_->totalEnergy(right);
    }

    // Left flux
    double massFluxL = left.rho * uL;
    std::array<double, 3> momFluxL{};
    for (int i = 0; i < dim_; ++i) {
        momFluxL[i] = left.rho * left.u[i] * uL;
    }
    double energyFluxL = rhoEL * uL;

    // Right flux
    double massFluxR = right.rho * uR;
    std::array<double, 3> momFluxR{};
    for (int i = 0; i < dim_; ++i) {
        momFluxR[i] = right.rho * right.u[i] * uR;
    }
    double energyFluxR = rhoER * uR;

    if (includePressure_) {
        for (int i = 0; i < dim_; ++i) {
            momFluxL[i] += left.p * normal[i];
            momFluxR[i] += right.p * normal[i];
        }
        energyFluxL += left.p * uL;
        energyFluxR += right.p * uR;
    }

    // Rusanov flux = 0.5*(F_L + F_R) - 0.5*sMax*(U_R - U_L)
    flux.massFlux = 0.5 * (massFluxL + massFluxR)
                  - 0.5 * sMax * (right.rho - left.rho);

    for (int i = 0; i < dim_; ++i) {
        flux.momentumFlux[i] = 0.5 * (momFluxL[i] + momFluxR[i])
            - 0.5 * sMax * (right.rho * right.u[i] - left.rho * left.u[i]);
    }

    flux.energyFlux = 0.5 * (energyFluxL + energyFluxR)
                    - 0.5 * sMax * (rhoER - rhoEL);

    // Alpha and pressure fluxes: central + Rusanov dissipation
    flux.faceVelocity = 0.5 * (uL + uR);
    flux.pressureFlux = 0.5 * (left.p * uL + right.p * uR)
                      - 0.5 * sMax * (right.p - left.p);
    int nAlphas = config_.isMultiPhase() ? config_.multiPhaseParams.nPhases : 0;
    for (int ph = 0; ph < nAlphas; ++ph) {
        flux.alphaFlux[ph] = 0.5 * (left.alpha[ph] * uL + right.alpha[ph] * uR)
                           - 0.5 * sMax * (right.alpha[ph] - left.alpha[ph]);
    }

    return flux;
}

double RusanovSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    const int dim_ = config_.dim;
    const bool includePressure_ = !config_.semiImplicit;
    double uL = std::abs(normalVelocity(left, normal, dim_));
    double uR = std::abs(normalVelocity(right, normal, dim_));

    if (includePressure_) {
        double cL = soundSpeedFromState(left, *eos_);
        double cR = soundSpeedFromState(right, *eos_);
        return std::max(uL + cL, uR + cR);
    }

    return std::max(uL, uR);
}

} // namespace SemiImplicitFV
