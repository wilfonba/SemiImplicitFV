#include "SolutionState.hpp"
#include "RectilinearMesh.hpp"
#include "EquationOfState.hpp"

namespace SemiImplicitFV {

void SolutionState::allocate(std::size_t totalCells, const SimulationConfig& config) {
    dim_ = config.dim;

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

    // Allocate star states for semi-implicit time stepping
    if (config.semiImplicit) {
        rhoUStar.assign(totalCells, 0.0);
        if (dim_ >= 2) rhoVStar.assign(totalCells, 0.0); else rhoVStar.clear();
        if (dim_ >= 3) rhoWStar.assign(totalCells, 0.0); else rhoWStar.clear();
        rhoEstar.assign(totalCells, 0.0);
        pAdvected.assign(totalCells, 0.0);
        rhoc2.assign(totalCells, 0.0);
        divUStar.assign(totalCells, 0.0);
    }

    aux.assign(totalCells, 0.0);

    // Allocate backup arrays for multi-stage time stepping
    if (config.RKOrder > 1) {
        rho0.assign(totalCells, 0.0);
        rhoU0.assign(totalCells, 0.0);
        rhoE0.assign(totalCells, 0.0);
        if (dim_ >= 2) rhoV0.assign(totalCells, 0.0); else rhoV0.clear();
        if (dim_ >= 3) rhoW0.assign(totalCells, 0.0); else rhoW0.clear();
    }
}

void SolutionState::copyCell_P(std::size_t dst, std::size_t src,
                                       double sU, double sV, double sW)
{
    rho[dst]  = rho[src];
    velU[dst] = sU * velU[src];

    if (dim_ >= 2) {
        velV[dst] = sV * velV[src];
    }

    if (dim_ >= 3) {
        velW[dst] = sW * velW[src];
    }

    pres[dst] = pres[src];
    temp[dst] = temp[src];
    sigma[dst] = sigma[src];
    aux[dst] = aux[src];
}

void SolutionState::copyCell_C(std::size_t dst, std::size_t src,
                                          double sU, double sV, double sW)
{
    rho[dst]  = rho[src];
    rhoU[dst] = sU * rhoU[src];

    if (dim_ >= 2) {
        rhoV[dst] = sV * rhoV[src];
    }

    if (dim_ >= 3) {
        rhoW[dst] = sW * rhoW[src];
    }

    rhoE[dst] = rhoE[src];
    sigma[dst] = sigma[src];
    aux[dst] = aux[src];
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

void SolutionState::convertConservativeToPrimitiveVariables(
        const RectilinearMesh& mesh,
        const std::shared_ptr<EquationOfState>& eos
        )
{
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U = getConservativeState(idx);
                PrimitiveState W = eos->toPrimitive(U);
                velU[idx] = W.u[0];
                if (dim_ >= 2) velV[idx] = W.u[1];
                if (dim_ >= 3) velW[idx] = W.u[2];
                pres[idx] = W.p;
                temp[idx] = W.T;
            }
        }
    }
}

void SolutionState::convertPrimitiveToConservativeVariables(
        const RectilinearMesh& mesh,
        const std::shared_ptr<EquationOfState>& eos
        )
{
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                PrimitiveState W = getPrimitiveState(idx);
                ConservativeState U = eos->toConservative(W);
                rho[idx] = U.rho;
                rhoU[idx] = U.rhoU[0];
                if (dim_ >= 2) rhoV[idx] = U.rhoU[1];
                if (dim_ >= 3) rhoW[idx] = U.rhoU[2];
                rhoE[idx] = U.rhoE;
            }
        }
    }
}

void SolutionState::saveConservativeCell(std::size_t idx) {
    rho0[idx]  = rho[idx];
    rhoU0[idx] = rhoU[idx];
    if (dim_ >= 2) rhoV0[idx] = rhoV[idx];
    if (dim_ >= 3) rhoW0[idx] = rhoW[idx];
    rhoE0[idx] = rhoE[idx];
}

void SolutionState::smoothFields(const RectilinearMesh& mesh, int nIterations) {
    // Smooth a single scalar field using explicit heat equation iterations.
    // Uses diffusion number nu = 1/(2*dim) for stability.
    // Ghost cells are filled with outflow (zero-gradient) via fillScalarGhosts.
    int dim = dim_;
    double nu = 1.0 / (4.0 * dim);

    auto smoothField = [&](std::vector<double>& field) {
        for (int iter = 0; iter < nIterations; ++iter) {
            mesh.fillScalarGhosts(field);
            for (int k = 0; k < mesh.nz(); ++k) {
                for (int j = 0; j < mesh.ny(); ++j) {
                    for (int i = 0; i < mesh.nx(); ++i) {
                        std::size_t idx = mesh.index(i, j, k);
                        double lap = 0.0;

                        std::size_t xm = mesh.index(i - 1, j, k);
                        std::size_t xp = mesh.index(i + 1, j, k);
                        lap += field[xm] + 2.0 * field[idx] + field[xp];

                        if (dim >= 2) {
                            std::size_t ym = mesh.index(i, j - 1, k);
                            std::size_t yp = mesh.index(i, j + 1, k);
                            lap += field[ym] + 2.0 * field[idx] + field[yp];
                        }
                        if (dim >= 3) {
                            std::size_t zm = mesh.index(i, j, k - 1);
                            std::size_t zp = mesh.index(i, j, k + 1);
                            lap += field[zm] + 2.0 * field[idx] + field[zp];
                        }

                        aux[idx] = nu * lap;
                    }
                }
            }
            // Copy result back
            for (int k = 0; k < mesh.nz(); ++k) {
                for (int j = 0; j < mesh.ny(); ++j) {
                    for (int i = 0; i < mesh.nx(); ++i) {
                        std::size_t idx = mesh.index(i, j, k);
                        field[idx] = aux[idx];
                    }
                }
            }
        }
    };

    // Smooth conservative variables
    smoothField(rho);
    smoothField(rhoU);
    if (dim >= 2) smoothField(rhoV);
    if (dim >= 3) smoothField(rhoW);
    smoothField(rhoE);

    // Smooth primitive variables to keep them consistent
    smoothField(velU);
    if (dim >= 2) smoothField(velV);
    if (dim >= 3) smoothField(velW);
    smoothField(pres);
    smoothField(temp);
}

#ifdef ENABLE_MPI
#include "HaloExchange.hpp"

void SolutionState::smoothFields(const RectilinearMesh& mesh, int nIterations,
                                  HaloExchange& halo) {
    int dim = dim_;
    double nu = 1.0 / (4.0 * dim);

    auto smoothField = [&](std::vector<double>& field) {
        for (int iter = 0; iter < nIterations; ++iter) {
            mesh.fillScalarGhosts(field, halo);
            for (int k = 0; k < mesh.nz(); ++k) {
                for (int j = 0; j < mesh.ny(); ++j) {
                    for (int i = 0; i < mesh.nx(); ++i) {
                        std::size_t idx = mesh.index(i, j, k);
                        double lap = 0.0;

                        std::size_t xm = mesh.index(i - 1, j, k);
                        std::size_t xp = mesh.index(i + 1, j, k);
                        lap += field[xm] + 2.0 * field[idx] + field[xp];

                        if (dim >= 2) {
                            std::size_t ym = mesh.index(i, j - 1, k);
                            std::size_t yp = mesh.index(i, j + 1, k);
                            lap += field[ym] + 2.0 * field[idx] + field[yp];
                        }
                        if (dim >= 3) {
                            std::size_t zm = mesh.index(i, j, k - 1);
                            std::size_t zp = mesh.index(i, j, k + 1);
                            lap += field[zm] + 2.0 * field[idx] + field[zp];
                        }

                        aux[idx] = nu * lap;
                    }
                }
            }
            for (int k = 0; k < mesh.nz(); ++k) {
                for (int j = 0; j < mesh.ny(); ++j) {
                    for (int i = 0; i < mesh.nx(); ++i) {
                        std::size_t idx = mesh.index(i, j, k);
                        field[idx] = aux[idx];
                    }
                }
            }
        }
    };

    smoothField(rho);
    smoothField(rhoU);
    if (dim >= 2) smoothField(rhoV);
    if (dim >= 3) smoothField(rhoW);
    smoothField(rhoE);

    smoothField(velU);
    if (dim >= 2) smoothField(velV);
    if (dim >= 3) smoothField(velW);
    smoothField(pres);
    smoothField(temp);
}

#endif // ENABLE_MPI

} // namespace SemiImplicitFV
