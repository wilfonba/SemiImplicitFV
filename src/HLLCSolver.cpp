#include "HLLCSolver.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

RiemannFlux HLLCSolver::computeFlux(
    const PrimitiveState& left,
    const PrimitiveState& right,
    const std::array<double, 3>& normal
) const {
    RiemannFlux flux;
    const int dim_ = config_.dim;

    double uL = normalVelocity(left, normal, dim_);
    double uR = normalVelocity(right, normal, dim_);

    // Conservative total energies (per unit volume)
    double rhoEL = left.rho * eos_->totalEnergy(left);
    double rhoER = right.rho * eos_->totalEnergy(right);

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
        for (int i = 0; i < dim_; ++i) {
            flux.momentumFlux[i] = left.rho * left.u[i] * uL;
        }
        flux.energyFlux = rhoEL * uL;

        if (includePressure_) {
            for (int i = 0; i < dim_; ++i) {
                flux.momentumFlux[i] += left.p * normal[i];
            }
            flux.energyFlux += left.p * uL;
        }
    }
    else if (sR <= 0) {
        // Right state flux
        flux.massFlux = right.rho * uR;
        for (int i = 0; i < dim_; ++i) {
            flux.momentumFlux[i] = right.rho * right.u[i] * uR;
        }
        flux.energyFlux = rhoER * uR;

        if (includePressure_) {
            for (int i = 0; i < dim_; ++i) {
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
            for (int i = 0; i < dim_; ++i) {
                double rhoUStarL = rhoStarL * (left.u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left.rho * left.u[i] * uL + left.p * normal[i]
                    + sL * (rhoUStarL - left.rho * left.u[i]);
            }

            double eL = rhoEL / left.rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * (sStar + left.p / (left.rho * (sL - uL))));
            flux.energyFlux = (rhoEL + left.p) * uL + sL * (EStarL - rhoEL);
        } else {
            for (int i = 0; i < dim_; ++i) {
                double rhoUStarL = rhoStarL * (left.u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left.rho * left.u[i] * uL
                    + sL * (rhoUStarL - left.rho * left.u[i]);
            }

            double eL = rhoEL / left.rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * sStar);
            flux.energyFlux = rhoEL * uL + sL * (EStarL - rhoEL);
        }
    }
    else {
        // Right star region
        double rhoStarR = right.rho * (sR - uR) / (sR - sStar);

        flux.massFlux = right.rho * uR + sR * (rhoStarR - right.rho);

        if (includePressure_) {
            for (int i = 0; i < dim_; ++i) {
                double rhoUStarR = rhoStarR * (right.u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right.rho * right.u[i] * uR + right.p * normal[i]
                    + sR * (rhoUStarR - right.rho * right.u[i]);
            }

            double eR = rhoER / right.rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * (sStar + right.p / (right.rho * (sR - uR))));
            flux.energyFlux = (rhoER + right.p) * uR + sR * (EStarR - rhoER);
        } else {
            for (int i = 0; i < dim_; ++i) {
                double rhoUStarR = rhoStarR * (right.u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right.rho * right.u[i] * uR
                    + sR * (rhoUStarR - right.rho * right.u[i]);
            }

            double eR = rhoER / right.rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * sStar);
            flux.energyFlux = rhoER * uR + sR * (EStarR - rhoER);
        }
    }

    return flux;
}

double HLLCSolver::maxWaveSpeed(
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
