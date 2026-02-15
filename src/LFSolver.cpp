#include "LFSolver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux computeLFFlux(
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

    // Max wave speed (Lax-Friedrichs uses full velocity magnitude + sound speed)
    double uLS = 0.0, uRS = 0.0;
    for (int i = 0; i < dim; ++i) {
        uLS += left.u[i] * left.u[i];
        uRS += right.u[i] * right.u[i];
    }
    uLS = std::sqrt(uLS);
    uRS = std::sqrt(uRS);

    double C;
    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        C = std::max(uLS, uRS) + std::max(cL, cR);
    } else {
        C = std::max(uLS, uRS);
    }

    flux.massFlux = 0.5 * (left.rho * uL + right.rho * uR) -
                    0.5 * C * (right.rho - left.rho);

    for (int i = 0; i < dim; ++ i) {
        flux.momentumFlux[i] = 0.5 * (left.rho * left.u[i] * uL + right.rho * right.u[i] * uR) -
                               0.5 * C * (right.rho * right.u[i] - left.rho * left.u[i]);
    }

    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);
    flux.energyFlux = 0.5 * (rhoEL * uL + rhoER * uR) - 0.5 * C * (rhoER - rhoEL);

    if (includePressure) {
        for (int i = 0; i < dim; ++i) {
            flux.momentumFlux[i] += 0.5 * (left.p + right.p) * normal[i];
        }
        flux.energyFlux += 0.5 * (left.p * uL + right.p * uR);
    }

    if (fc.useIGR) {
        for (int i = 0; i < dim; ++i) {
            flux.momentumFlux[i] += 0.5 * (left.sigma + right.sigma) * normal[i];
        }
        flux.energyFlux += 0.5 * (left.sigma * uL + right.sigma * uR);
    }

    // Alpha and pressure fluxes: central + Lax-Friedrichs dissipation
    flux.faceVelocity = 0.5 * (uL + uR);
    flux.pressureFlux = 0.5 * (left.p * uL + right.p * uR)
                      - 0.5 * C * (right.p - left.p);
    for (int ph = 0; ph < fc.nPhases; ++ph) {
        flux.alphaFlux[ph] = 0.5 * (left.alpha[ph] * uL + right.alpha[ph] * uR)
                           - 0.5 * C * (right.alpha[ph] - left.alpha[ph]);
    }

    return flux;
}

// Virtual method wrapper for backward compatibility
RiemannFlux LFSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    FluxConfig fc;
    fc.dim = config_.dim;
    fc.includePressure = !config_.semiImplicit;
    fc.useIGR = config_.useIGR;
    fc.nPhases = config_.isMultiPhase() ? config_.multiPhaseParams.nPhases : 0;
    return computeLFFlux(left, right, normal, fc);
}

double LFSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    [[maybe_unused]] const std::array<double, 3>& normal
) const {
    const int dim_ = config_.dim;
    const bool includePressure_ = !config_.semiImplicit;
    double uLS{0.0};
    double uRS{0.0};

    for (int i = 0; i < dim_; ++i) {
        uLS += left.u[i] * left.u[i];
        uRS += right.u[i] * right.u[i];
    }

    uLS = std::sqrt(uLS);
    uRS = std::sqrt(uRS);

    if (includePressure_) {
        double cL = soundSpeedFromState(left, *eos_);
        double cR = soundSpeedFromState(right, *eos_);
        return std::max(uLS, uRS) + std::max(cL, cR);
    }

    return std::max(uLS, uRS);
}

} // namespace SemiImplicitFV
