#include "EquationOfState.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

EOSData eos_create_ideal_gas(double gamma, double R, int dim) {
    EOSData eos;
    memset(&eos, 0, sizeof(eos));
    eos.type = EOS_IDEAL_GAS;
    eos.gamma = gamma;
    eos.pInf = 0.0;
    eos.R = R;
    eos.dim = dim;
    return eos;
}

EOSData eos_create_stiffened_gas(double gamma, double pInf, double R, int dim) {
    EOSData eos;
    memset(&eos, 0, sizeof(eos));
    eos.type = EOS_STIFFENED_GAS;
    eos.gamma = gamma;
    eos.pInf = pInf;
    eos.R = R;
    eos.dim = dim;
    return eos;
}

#pragma omp declare target
double eos_pressure(const EOSData* eos, const ConservativeState* U) {
    double rho = std::max(U->rho, 1e-14);
    double ke = 0.0;
    for (int d = 0; d < eos->dim; ++d)
        ke += U->rhoU[d] * U->rhoU[d];
    ke *= 0.5 / rho;
    double e = U->rhoE - ke;
    return (eos->gamma - 1.0) * e - eos->gamma * eos->pInf;
}

double eos_temperature(const EOSData* eos, const PrimitiveState* W) {
    return (W->p + eos->pInf) / (W->rho * eos->R);
}

double eos_sound_speed(const EOSData* eos, const PrimitiveState* W) {
    double rho = std::max(W->rho, 1e-14);
    double pEff = W->p + eos->pInf;
    return std::sqrt(eos->gamma * pEff / rho);
}

double eos_internal_energy(const EOSData* eos, const PrimitiveState* W) {
    double rho = std::max(W->rho, 1e-14);
    return (W->p + eos->gamma * eos->pInf) / (rho * (eos->gamma - 1.0));
}

double eos_total_energy(const EOSData* eos, const PrimitiveState* W) {
    double ke = 0.0;
    for (int d = 0; d < eos->dim; ++d)
        ke += W->u[d] * W->u[d];
    ke *= 0.5;
    return eos_internal_energy(eos, W) + ke;
}

PrimitiveState eos_to_primitive(const EOSData* eos, const ConservativeState* U) {
    PrimitiveState W;
    memset(&W, 0, sizeof(W));

    W.rho = std::max(U->rho, 1e-14);

    for (int d = 0; d < eos->dim; ++d)
        W.u[d] = U->rhoU[d] / W.rho;

    double ke = 0.0;
    for (int d = 0; d < eos->dim; ++d)
        ke += W.u[d] * W.u[d];
    ke *= 0.5 * W.rho;
    double e = U->rhoE - ke;

    W.p = (eos->gamma - 1.0) * e - eos->gamma * eos->pInf;

    return W;
}

ConservativeState eos_to_conservative(const EOSData* eos, const PrimitiveState* W) {
    ConservativeState U;
    memset(&U, 0, sizeof(U));

    U.rho = W->rho;
    for (int d = 0; d < eos->dim; ++d)
        U.rhoU[d] = W->rho * W->u[d];

    double ke = 0.0;
    for (int d = 0; d < eos->dim; ++d)
        ke += W->u[d] * W->u[d];
    ke *= 0.5 * W->rho;
    double e = (W->p + eos->gamma * eos->pInf) / (eos->gamma - 1.0);

    U.rhoE = e + ke;

    return U;
}
#pragma omp end declare target
