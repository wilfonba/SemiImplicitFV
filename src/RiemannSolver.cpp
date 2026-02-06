#include "RiemannSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

namespace {

// Helper: compute normal velocity
double normalVelocity(const PrimitiveState& W, const std::array<double, 3>& n) {
    return W.u[0] * n[0] + W.u[1] * n[1] + W.u[2] * n[2];
}

} // anonymous namespace

// =============================================================================
// Upwind Solver
// =============================================================================

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

// =============================================================================
// Rusanov Solver
// =============================================================================

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

// =============================================================================
// HLLC Solver
// =============================================================================

RiemannFlux HLLCSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    RiemannFlux flux;

    double uL = normalVelocity(left, normal);
    double uR = normalVelocity(right, normal);

    // Total energies
    double EL = eos_->totalEnergy(left);
    double ER = eos_->totalEnergy(right);

    // Wave speed estimates
    double sL, sR, sStar;

    if (includePressure_) {
        // Full Euler: use Davis estimates with sound speed
        double cL = eos_->soundSpeed(left);
        double cR = eos_->soundSpeed(right);
        sL = std::min(uL - cL, uR - cR);
        sR = std::max(uL + cL, uR + cR);

        // HLLC contact wave speed
        sStar = (right.p - left.p
                 + left.rho * uL * (sL - uL)
                 - right.rho * uR * (sR - uR))
              / (left.rho * (sL - uL) - right.rho * (sR - uR));
    } else {
        // Advective system: all waves travel at material velocity
        sL = std::min(uL, uR);
        sR = std::max(uL, uR);
        sStar = 0.5 * (uL + uR);
    }

    if (sL >= 0) {
        // Left state flux
        flux.massFlux = left.rho * uL;
        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = left.rho * left.u[i] * uL;
        }
        flux.energyFlux = EL * uL;

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                flux.momentumFlux[i] += left.p * normal[i];
            }
            flux.energyFlux += left.p * uL;
        }
    }
    else if (sR <= 0) {
        // Right state flux
        flux.massFlux = right.rho * uR;
        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = right.rho * right.u[i] * uR;
        }
        flux.energyFlux = ER * uR;

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                flux.momentumFlux[i] += right.p * normal[i];
            }
            flux.energyFlux += right.p * uR;
        }
    }
    else if (sStar >= 0) {
        // Left star region
        double rhoStarL = left.rho * (sL - uL) / (sL - sStar);

        flux.massFlux = left.rho * uL + sL * (rhoStarL - left.rho);

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                double rhoUStarL = rhoStarL * (left.u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left.rho * left.u[i] * uL + left.p * normal[i]
                    + sL * (rhoUStarL - left.rho * left.u[i]);
            }

            double eL = EL / left.rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * (sStar + left.p / (left.rho * (sL - uL))));
            flux.energyFlux = (EL + left.p) * uL + sL * (EStarL - EL);
        } else {
            for (int i = 0; i < 3; ++i) {
                double rhoUStarL = rhoStarL * (left.u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left.rho * left.u[i] * uL
                    + sL * (rhoUStarL - left.rho * left.u[i]);
            }

            double eL = EL / left.rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * sStar);
            flux.energyFlux = EL * uL + sL * (EStarL - EL);
        }
    }
    else {
        // Right star region
        double rhoStarR = right.rho * (sR - uR) / (sR - sStar);

        flux.massFlux = right.rho * uR + sR * (rhoStarR - right.rho);

        if (includePressure_) {
            for (int i = 0; i < 3; ++i) {
                double rhoUStarR = rhoStarR * (right.u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right.rho * right.u[i] * uR + right.p * normal[i]
                    + sR * (rhoUStarR - right.rho * right.u[i]);
            }

            double eR = ER / right.rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * (sStar + right.p / (right.rho * (sR - uR))));
            flux.energyFlux = (ER + right.p) * uR + sR * (EStarR - ER);
        } else {
            for (int i = 0; i < 3; ++i) {
                double rhoUStarR = rhoStarR * (right.u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right.rho * right.u[i] * uR
                    + sR * (rhoUStarR - right.rho * right.u[i]);
            }

            double eR = ER / right.rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * sStar);
            flux.energyFlux = ER * uR + sR * (EStarR - ER);
        }
    }

    return flux;
}

double HLLCSolver::maxWaveSpeed(
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
