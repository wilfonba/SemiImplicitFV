#include "RiemannSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

namespace {

// Helper: compute normal velocity
double normalVelocity(const PrimitiveState& W, const std::array<double, 3>& n) {
    return W.u[0] * n[0] + W.u[1] * n[1] + W.u[2] * n[2];
}

// Helper: compute kinetic energy per unit mass
double kineticEnergy(const PrimitiveState& W) {
    return 0.5 * (W.u[0] * W.u[0] + W.u[1] * W.u[1] + W.u[2] * W.u[2]);
}

} // anonymous namespace

// =============================================================================
// Upwind Advective Solver
// =============================================================================

AdvectiveFlux UpwindAdvectiveSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    AdvectiveFlux flux;

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

        // E = ρe + 0.5ρu² (total energy density)
        // For advective flux: Eu (no pu term)
        double E = left.p / (1.4 - 1.0) + left.rho * kineticEnergy(left);  // TODO: use EOS
        flux.energyFlux = E * uL;
    } else {
        // Use right state
        flux.massFlux = right.rho * uR;

        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = right.rho * right.u[i] * uR;
        }

        double E = right.p / (1.4 - 1.0) + right.rho * kineticEnergy(right);
        flux.energyFlux = E * uR;
    }

    return flux;
}

double UpwindAdvectiveSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    // For advective system, wave speed is just material velocity (no sound speed!)
    double uL = std::abs(normalVelocity(left, normal));
    double uR = std::abs(normalVelocity(right, normal));
    return std::max(uL, uR);
}

// =============================================================================
// Rusanov Advective Solver
// =============================================================================

AdvectiveFlux RusanovAdvectiveSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    AdvectiveFlux flux;

    double uL = normalVelocity(left, normal);
    double uR = normalVelocity(right, normal);

    // Maximum wave speed (material velocity only)
    double sMax = maxWaveSpeed(left, right, normal);

    // Total energies
    double EL = left.p / (1.4 - 1.0) + left.rho * kineticEnergy(left);
    double ER = right.p / (1.4 - 1.0) + right.rho * kineticEnergy(right);

    // Left flux (advective, no pressure)
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

double RusanovAdvectiveSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    double uL = std::abs(normalVelocity(left, normal));
    double uR = std::abs(normalVelocity(right, normal));
    return std::max(uL, uR);
}

// =============================================================================
// HLLC Advective Solver
// =============================================================================

AdvectiveFlux HLLCAdvectiveSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    AdvectiveFlux flux;

    double uL = normalVelocity(left, normal);
    double uR = normalVelocity(right, normal);

    // For the advective system, all waves travel at the material velocity
    // Wave speed estimates
    double sL = std::min(uL, uR);
    double sR = std::max(uL, uR);

    // Contact wave speed (average)
    double sStar = 0.5 * (uL + uR);

    // Total energies
    double EL = left.p / (1.4 - 1.0) + left.rho * kineticEnergy(left);
    double ER = right.p / (1.4 - 1.0) + right.rho * kineticEnergy(right);

    if (sL >= 0) {
        // Left state flux
        flux.massFlux = left.rho * uL;
        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = left.rho * left.u[i] * uL;
        }
        flux.energyFlux = EL * uL;
    }
    else if (sR <= 0) {
        // Right state flux
        flux.massFlux = right.rho * uR;
        for (int i = 0; i < 3; ++i) {
            flux.momentumFlux[i] = right.rho * right.u[i] * uR;
        }
        flux.energyFlux = ER * uR;
    }
    else if (sStar >= 0) {
        // Left star region
        double rhoStarL = left.rho * (sL - uL) / (sL - sStar);

        flux.massFlux = left.rho * uL + sL * (rhoStarL - left.rho);

        for (int i = 0; i < 3; ++i) {
            double rhoUStarL = rhoStarL * (left.u[i] + (sStar - uL) * normal[i]);
            flux.momentumFlux[i] = left.rho * left.u[i] * uL
                + sL * (rhoUStarL - left.rho * left.u[i]);
        }

        // Energy in star region
        double eL = EL / left.rho;  // specific total energy
        double EStarL = rhoStarL * (eL + (sStar - uL) * sStar);
        flux.energyFlux = EL * uL + sL * (EStarL - EL);
    }
    else {
        // Right star region
        double rhoStarR = right.rho * (sR - uR) / (sR - sStar);

        flux.massFlux = right.rho * uR + sR * (rhoStarR - right.rho);

        for (int i = 0; i < 3; ++i) {
            double rhoUStarR = rhoStarR * (right.u[i] + (sStar - uR) * normal[i]);
            flux.momentumFlux[i] = right.rho * right.u[i] * uR
                + sR * (rhoUStarR - right.rho * right.u[i]);
        }

        double eR = ER / right.rho;
        double EStarR = rhoStarR * (eR + (sStar - uR) * sStar);
        flux.energyFlux = ER * uR + sR * (EStarR - ER);
    }

    return flux;
}

double HLLCAdvectiveSolver::maxWaveSpeed(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    double uL = std::abs(normalVelocity(left, normal));
    double uR = std::abs(normalVelocity(right, normal));
    return std::max(uL, uR);
}

} // namespace SemiImplicitFV
