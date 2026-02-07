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

    double uL = normalVelocity(left, normal, dim_);
    double uR = normalVelocity(right, normal, dim_);

    // Maximum wave speed
    double sMax = maxWaveSpeed(left, right, normal);

    // Conservative total energies (per unit volume)
    double rhoEL = left.rho * eos_->totalEnergy(left);
    double rhoER = right.rho * eos_->totalEnergy(right);

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

    return flux;
}

double RusanovSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    const int dim_ = config_.dim;
    double uL = std::abs(normalVelocity(left, normal, dim_));
    double uR = std::abs(normalVelocity(right, normal, dim_));

    if (includePressure_) {
        double cL = eos_->soundSpeed(left);
        double cR = eos_->soundSpeed(right);
        return std::max(uL + cL, uR + cR);
    }

    return std::max(uL, uR);
}

} // namespace SemiImplicitFV
