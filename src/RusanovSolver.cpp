#include "RusanovSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux computeRusanovFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal,
    const FluxConfig& fc
) {
    RiemannFlux flux;
    const int dim = fc.dim;
    const bool includePressure = fc.includePressure;

    double uL = normalVelocity(left, normal, dim);
    double uR = normalVelocity(right, normal, dim);

    // Maximum wave speed (Rusanov uses normal velocity + sound speed)
    double absUL = std::abs(normalVelocity(left, normal, dim));
    double absUR = std::abs(normalVelocity(right, normal, dim));
    double sMax;
    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        sMax = std::max(absUL + cL, absUR + cR);
    } else {
        sMax = std::max(absUL, absUR);
    }

    // Conservative total energies
    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);

    // Left flux
    double massFluxL = left.rho * uL;
    std::array<double, 3> momFluxL{};
    for (int i = 0; i < dim; ++i) {
        momFluxL[i] = left.rho * left.u[i] * uL;
    }
    double energyFluxL = rhoEL * uL;

    // Right flux
    double massFluxR = right.rho * uR;
    std::array<double, 3> momFluxR{};
    for (int i = 0; i < dim; ++i) {
        momFluxR[i] = right.rho * right.u[i] * uR;
    }
    double energyFluxR = rhoER * uR;

    if (includePressure) {
        for (int i = 0; i < dim; ++i) {
            momFluxL[i] += left.p * normal[i];
            momFluxR[i] += right.p * normal[i];
        }
        energyFluxL += left.p * uL;
        energyFluxR += right.p * uR;
    }

    // Rusanov flux = 0.5*(F_L + F_R) - 0.5*sMax*(U_R - U_L)
    flux.massFlux = 0.5 * (massFluxL + massFluxR)
                  - 0.5 * sMax * (right.rho - left.rho);

    for (int i = 0; i < dim; ++i) {
        flux.momentumFlux[i] = 0.5 * (momFluxL[i] + momFluxR[i])
            - 0.5 * sMax * (right.rho * right.u[i] - left.rho * left.u[i]);
    }

    flux.energyFlux = 0.5 * (energyFluxL + energyFluxR)
                    - 0.5 * sMax * (rhoER - rhoEL);

    // Alpha and pressure fluxes: central + Rusanov dissipation
    flux.faceVelocity = 0.5 * (uL + uR);
    flux.pressureFlux = 0.5 * (left.p * uL + right.p * uR)
                      - 0.5 * sMax * (right.p - left.p);
    for (int ph = 0; ph < fc.nPhases; ++ph) {
        flux.alphaFlux[ph] = 0.5 * (left.alpha[ph] * uL + right.alpha[ph] * uR)
                           - 0.5 * sMax * (right.alpha[ph] - left.alpha[ph]);
    }

    return flux;
}

// Virtual method wrapper for backward compatibility
RiemannFlux RusanovSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    FluxConfig fc;
    fc.dim = config_.dim;
    fc.includePressure = !config_.semiImplicit;
    fc.useIGR = config_.useIGR;
    fc.nPhases = config_.isMultiPhase() ? config_.multiPhaseParams.nPhases : 0;
    return computeRusanovFlux(left, right, normal, fc);
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
