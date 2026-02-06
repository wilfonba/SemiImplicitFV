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

    double uL = normalVelocity(left, normal);
    double uR = normalVelocity(right, normal);

    // Maximum wave speed
    double sMax = maxWaveSpeed(left, right, normal);

    // Total energies
    double EL = eos_->totalEnergy(left);
    double ER = eos_->totalEnergy(right);

    // Left flux
    double massFluxL = left.rho * uL;
    std::array<double, 3> momFluxL;
    for (int i = 0; i < 3; ++i) {
        momFluxL[i] = left.rho * left.u[i] * uL;
    }
    double energyFluxL = EL * uL;

    // Right flux
    double massFluxR = right.rho * uR;
    std::array<double, 3> momFluxR;
    for (int i = 0; i < 3; ++i) {
        momFluxR[i] = right.rho * right.u[i] * uR;
    }
    double energyFluxR = ER * uR;

    if (includePressure_) {
        for (int i = 0; i < 3; ++i) {
            momFluxL[i] += left.p * normal[i];
            momFluxR[i] += right.p * normal[i];
        }
        energyFluxL += left.p * uL;
        energyFluxR += right.p * uR;
    }

    // Rusanov flux = 0.5*(F_L + F_R) - 0.5*sMax*(U_R - U_L)
    flux.massFlux = 0.5 * (massFluxL + massFluxR)
                  - 0.5 * sMax * (right.rho - left.rho);

    for (int i = 0; i < 3; ++i) {
        flux.momentumFlux[i] = 0.5 * (momFluxL[i] + momFluxR[i])
            - 0.5 * sMax * (right.rho * right.u[i] - left.rho * left.u[i]);
    }

    flux.energyFlux = 0.5 * (energyFluxL + energyFluxR)
                    - 0.5 * sMax * (ER - EL);

    return flux;
}

double RusanovSolver::maxWaveSpeed(
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

