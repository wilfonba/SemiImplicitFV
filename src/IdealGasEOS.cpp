#include "IdealGasEOS.hpp"

namespace SemiImplicitFV {

IdealGasEOS::IdealGasEOS(double gamma, double R, const SimulationConfig& config)
    : EquationOfState(config)
    , gamma_(gamma)
    , R_(R)
{}

double IdealGasEOS::pressure(const ConservativeState& U) const {
    double rho = std::max(U.rho, 1e-14);
    double ke = 0.0;
    for (int d = 0; d < config_.dim; ++d)
        ke += U.rhoU[d] * U.rhoU[d];
    ke *= 0.5 / rho;
    double e = U.rhoE - ke;

    return (gamma_ - 1.0) * e;
}

double IdealGasEOS::temperature(const PrimitiveState& W) const {
    return W.p / (W.rho * R_);
}

double IdealGasEOS::soundSpeed(const PrimitiveState& W) const {
    double rho = std::max(W.rho, 1e-14);
    double p = std::max(W.p, 1e-14);
    return std::sqrt(gamma_ * p / rho);
}

double IdealGasEOS::internalEnergy(const PrimitiveState& W) const {
    // e = p / (rho * (gamma - 1)) = cv * T
    double rho = std::max(W.rho, 1e-14);
    return W.p / (rho * (gamma_ - 1.0));
}

double IdealGasEOS::totalEnergy(const PrimitiveState& W) const {
    double ke = 0.0;
    for (int d = 0; d < config_.dim; ++d)
        ke += W.u[d] * W.u[d];
    ke *= 0.5;
    return internalEnergy(W) + ke;
}

PrimitiveState IdealGasEOS::toPrimitive(const ConservativeState& U) const {
    PrimitiveState W;

    W.rho = std::max(U.rho, 1e-14);

    for (int d = 0; d < config_.dim; ++d)
        W.u[d] = U.rhoU[d] / W.rho;

    double ke = 0.0;
    for (int d = 0; d < config_.dim; ++d)
        ke += W.u[d] * W.u[d];
    ke *= 0.5 * W.rho;
    double e = U.rhoE - ke;

    W.p = std::max((gamma_ - 1.0) * e, 1e-14);
    W.T = W.p / (W.rho * R_);

    return W;
}

ConservativeState IdealGasEOS::toConservative(const PrimitiveState& W) const {
    ConservativeState U;

    U.rho = W.rho;
    for (int d = 0; d < config_.dim; ++d)
        U.rhoU[d] = W.rho * W.u[d];

    double ke = 0.0;
    for (int d = 0; d < config_.dim; ++d)
        ke += W.u[d] * W.u[d];
    ke *= 0.5 * W.rho;
    double e = W.p / (gamma_ - 1.0);

    U.rhoE = e + ke;

    return U;
}

} // namespace SemiImplicitFV

