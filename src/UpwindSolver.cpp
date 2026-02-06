#include "UpwindSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux UpwindSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    RiemannFlux flux;

    double uL = normalVelocity(left, normal);
    double uR = normalVelocity(right, normal);

    // Interface velocity for upwinding
    double uInterface = 0.5 * (uL + uR);

    if (uInterface >= 0) {
        // Use left state
        flux.massFlux = left.rho * uL;

        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = left.rho * left.u[i] * uL;
        }

        double E = eos_->totalEnergy(left);
        flux.energyFlux = E * uL;

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                flux.momentumFlux[i] += left.p * normal[i];
            }
            flux.energyFlux += left.p * uL;
        }
    } else {
        // Use right state
        flux.massFlux = right.rho * uR;

        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = right.rho * right.u[i] * uR;
        }

        double E = eos_->totalEnergy(right);
        flux.energyFlux = E * uR;

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                flux.momentumFlux[i] += right.p * normal[i];
            }
            flux.energyFlux += right.p * uR;
        }
    }

    return flux;
}

double UpwindSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    double uL = std::abs(normalVelocity(left, normal));
    double uR = std::abs(normalVelocity(right, normal));

    if (includePressure_) {
        double cL = eos_->soundSpeed(left);
        double cR = eos_->soundSpeed(right);
        return std::max(uL + cL, uR + cR);
    }

    return std::max(uL, uR);
}

} // namespace SemiImplicitFV
