#include "SolutionState.hpp"

namespace SemiImplicitFV {

void SolutionState::allocate(std::size_t totalCells, int dim) {
    dim_ = dim;

    rho.assign(totalCells, 0.0);
    rhoU.assign(totalCells, 0.0);
    rhoE.assign(totalCells, 0.0);

    velU.assign(totalCells, 0.0);
    pres.assign(totalCells, 0.0);
    temp.assign(totalCells, 0.0);
    sigma.assign(totalCells, 0.0);

    if (dim_ >= 2) {
        rhoV.assign(totalCells, 0.0);
        velV.assign(totalCells, 0.0);
    } else {
        rhoV.clear();
        velV.clear();
    }

    if (dim_ >= 3) {
        rhoW.assign(totalCells, 0.0);
        velW.assign(totalCells, 0.0);
    } else {
        rhoW.clear();
        velW.clear();
    }

    aux.assign(totalCells, 0.0);
}

void SolutionState::copyCell(std::size_t dst, std::size_t src,
                              double sU, double sV, double sW)
{
    rho[dst]  = rho[src];

    rhoU[dst] = sU * rhoU[src];
    velU[dst] = sU * velU[src];

    if (dim_ >= 2) {
        rhoV[dst] = sV * rhoV[src];
        velV[dst] = sV * velV[src];
    }

    if (dim_ >= 3) {
        rhoW[dst] = sW * rhoW[src];
        velW[dst] = sW * velW[src];
    }

    rhoE[dst] = rhoE[src];
    pres[dst] = pres[src];

    temp[dst] = temp[src];
    sigma[dst] = sigma[src];

    aux[dst] = aux[src];
}

ConservativeState SolutionState::getConservativeState(std::size_t idx) const {
    ConservativeState U;
    U.rho = rho[idx];
    U.rhoU[0] = rhoU[idx];
    U.rhoU[1] = (dim_ >= 2) ? rhoV[idx] : 0.0;
    U.rhoU[2] = (dim_ >= 3) ? rhoW[idx] : 0.0;
    U.rhoE = rhoE[idx];
    return U;
}

void SolutionState::setConservativeState(std::size_t idx, const ConservativeState& U) {
    rho[idx]  = U.rho;
    rhoU[idx] = U.rhoU[0];
    if (dim_ >= 2) rhoV[idx] = U.rhoU[1];
    if (dim_ >= 3) rhoW[idx] = U.rhoU[2];
    rhoE[idx] = U.rhoE;
}

PrimitiveState SolutionState::getPrimitiveState(std::size_t idx) const {
    PrimitiveState W;
    W.rho = rho[idx];
    W.u[0] = velU[idx];
    W.u[1] = (dim_ >= 2) ? velV[idx] : 0.0;
    W.u[2] = (dim_ >= 3) ? velW[idx] : 0.0;
    W.p = pres[idx];
    W.T = temp[idx];
    W.sigma = sigma[idx];
    return W;
}

void SolutionState::setPrimitiveState(std::size_t idx, const PrimitiveState& W) {
    rho[idx]   = W.rho;
    velU[idx]  = W.u[0];
    if (dim_ >= 2) velV[idx] = W.u[1];
    if (dim_ >= 3) velW[idx] = W.u[2];
    pres[idx]  = W.p;
    temp[idx]  = W.T;
    sigma[idx] = W.sigma;
}

} // namespace SemiImplicitFV
