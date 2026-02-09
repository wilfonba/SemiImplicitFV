#include "LFSolver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux LFSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    RiemannFlux flux;
    const int dim_ = config_.dim;
    const bool includePressure_ = !config_.semiImplicit;

    double uL = normalVelocity(left, normal, dim_);
    double uR = normalVelocity(right, normal, dim_);
    double C = maxWaveSpeed(left, right, normal);

    flux.massFlux = 0.5 * (left.rho * uL + right.rho * uR) -
                    0.5 * C * (right.rho - left.rho);

    for (int i = 0; i < dim_; ++ i) {
        flux.momentumFlux[i] = 0.5 * (left.rho * left.u[i] * uL + right.rho * right.u[i] * uR) -
                               0.5 * C * (right.rho * right.u[i] - left.rho * left.u[i]);
    }

    double rhoEL = left.rho * eos_->totalEnergy(left);
    double rhoER = right.rho * eos_->totalEnergy(right);
    flux.energyFlux = 0.5 * (rhoEL * uL + rhoER * uR) - 0.5 * C * (rhoER - rhoEL);

    if (includePressure_) {
        for (int i = 0; i < dim_; ++i) {
            flux.momentumFlux[i] += 0.5 * (left.p + right.p) * normal[i];
        }
        flux.energyFlux += 0.5 * (left.p * uL + right.p * uR);
    }

    if (config_.useIGR) {
        for (int i = 0; i < dim_; ++i) {
            flux.momentumFlux[i] += 0.5 * (left.sigma + right.sigma) * normal[i];
        }
        flux.energyFlux += 0.5 * (left.sigma * uL + right.sigma * uR);
    }

    return flux;
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
        uLS += left.u[i] * left.u[i] * normal[i];
        uRS += right.u[i] * right.u[i] * normal[i];
    }

    uLS = std::sqrt(uLS);
    uRS = std::sqrt(uRS);

    if (includePressure_) {
        double cL = eos_->soundSpeed(left);
        double cR = eos_->soundSpeed(right);
        return std::max(uLS, uRS) + std::max(cL, cR);
    }

    return std::max(uLS, uRS);
}

} // namespace SemiImplicitFV
