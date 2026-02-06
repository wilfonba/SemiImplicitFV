#include "StiffenedGasEOS.hpp"

namespace SemiImplicitFV {

StiffenedGasEOS::StiffenedGasEOS(double gamma, double pInf, double R)
    : gamma_(gamma)
    , pInf_(pInf)
    , R_(R)
{}

double StiffenedGasEOS::pressure(const ConservativeState& U) const {
    double rho = std::max(U.rho, 1e-14);
    double ke = 0.5 * (U.rhoU[0]*U.rhoU[0] + U.rhoU[1]*U.rhoU[1] + U.rhoU[2]*U.rhoU[2]) / rho;
    double e = U.rhoE - ke;

    return (gamma_ - 1.0) * e - gamma_ * pInf_;
}

double StiffenedGasEOS::temperature(const PrimitiveState& W) const {
    return (W.p + pInf_) / (W.rho * R_);
}

double StiffenedGasEOS::soundSpeed(const PrimitiveState& W) const {
    double rho = std::max(W.rho, 1e-14);
    double pEff = W.p + pInf_;
    return std::sqrt(gamma_ * pEff / rho);
}

double StiffenedGasEOS::internalEnergy(const PrimitiveState& W) const {
    double rho = std::max(W.rho, 1e-14);
    return (W.p + gamma_ * pInf_) / (rho * (gamma_ - 1.0));
}

double StiffenedGasEOS::totalEnergy(const PrimitiveState& W) const {
    double ke = 0.5 * (W.u[0]*W.u[0] + W.u[1]*W.u[1] + W.u[2]*W.u[2]);
    return internalEnergy(W) + ke;
}

PrimitiveState StiffenedGasEOS::toPrimitive(const ConservativeState& U) const {
    PrimitiveState W;

    W.rho = std::max(U.rho, 1e-14);

    W.u[0] = U.rhoU[0] / W.rho;
    W.u[1] = U.rhoU[1] / W.rho;
    W.u[2] = U.rhoU[2] / W.rho;

    double ke = 0.5 * W.rho * (W.u[0]*W.u[0] + W.u[1]*W.u[1] + W.u[2]*W.u[2]);
    double e = U.rhoE - ke;

    W.p = (gamma_ - 1.0) * e - gamma_ * pInf_;
    W.T = (W.p + pInf_) / (W.rho * R_);

    return W;
}

ConservativeState StiffenedGasEOS::toConservative(const PrimitiveState& W) const {
    ConservativeState U;

    U.rho = W.rho;
    U.rhoU[0] = W.rho * W.u[0];
    U.rhoU[1] = W.rho * W.u[1];
    U.rhoU[2] = W.rho * W.u[2];

    double ke = 0.5 * W.rho * (W.u[0]*W.u[0] + W.u[1]*W.u[1] + W.u[2]*W.u[2]);
    double e = (W.p + gamma_ * pInf_) / (gamma_ - 1.0);

    U.rhoE = e + ke;

    return U;
}

} // SemiImplicitFV
