#include "SemiImplicitSolver.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace SemiImplicitFV {

SemiImplicitSolver::SemiImplicitSolver(
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<PressureSolver> pressureSolver,
    std::shared_ptr<EquationOfState> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const SemiImplicitParams& params
)
    : riemannSolver_(std::move(riemannSolver))
    , pressureSolver_(std::move(pressureSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(params)
    , lastPressureIters_(0)
{}

void SemiImplicitSolver::ensureStorage(const RectilinearMesh& mesh) {
    std::size_t n = mesh.totalCells();
    if (rhoStar_.size() == n) return;

    int dim = mesh.dim();

    rhoStar_.resize(n);
    rhoUStar_.resize(n);
    if (dim >= 2) {
        rhoVStar_.resize(n);
    } else {
        rhoVStar_.clear();
    }
    if (dim >= 3) {
        rhoWStar_.resize(n);
    } else {
        rhoWStar_.clear();
    }
    rhoEStar_.resize(n);
    pAdvected_.resize(n);
    rhoc2_.resize(n);
    pressureRhs_.resize(n);
    pressure_.resize(n);
    divUstar_.resize(n);
    if (igrSolver_) {
        gradU_.resize(n);
    }
}

void SemiImplicitSolver::writeStarToState(const RectilinearMesh& mesh, SolutionState& state) {
    int dim = mesh.dim();
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                state.rho[idx]  = rhoStar_[idx];
                state.rhoU[idx] = rhoUStar_[idx];
                if (dim >= 2) state.rhoV[idx] = rhoVStar_[idx];
                if (dim >= 3) state.rhoW[idx] = rhoWStar_[idx];
                state.rhoE[idx] = rhoEStar_[idx];

                ConservativeState U;
                U.rho = rhoStar_[idx];
                U.rhoU = {rhoUStar_[idx],
                          (dim >= 2) ? rhoVStar_[idx] : 0.0,
                          (dim >= 3) ? rhoWStar_[idx] : 0.0};
                U.rhoE = rhoEStar_[idx];
                PrimitiveState W = eos_->toPrimitive(U);
                state.velU[idx] = W.u[0];
                if (dim >= 2) state.velV[idx] = W.u[1];
                if (dim >= 3) state.velW[idx] = W.u[2];
                state.pres[idx] = W.p;
                state.temp[idx] = W.T;
            }
        }
    }
    mesh.applyBoundaryConditions(state);
}

double SemiImplicitSolver::step(const RectilinearMesh& mesh, SolutionState& state, double targetDt) {
    double dt = computeAdvectiveTimeStep(mesh, state);
    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }
    dt = std::clamp(dt, params_.minDt, params_.maxDt);

    ensureStorage(mesh);

    mesh.applyBoundaryConditions(state);

    advectionStep(mesh, state, dt);
    advectPressure(mesh, state, dt);
    writeStarToState(mesh, state);

    if (params_.useIGR && igrSolver_) {
        solveIGR(mesh, state);
    }

    solvePressure(mesh, state, dt);
    correctionStep(mesh, state, dt);

    // Final: convert corrected conservatives to primitives
    int dim = mesh.dim();
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U;
                U.rho = state.rho[idx];
                U.rhoU = {state.rhoU[idx],
                          (dim >= 2) ? state.rhoV[idx] : 0.0,
                          (dim >= 3) ? state.rhoW[idx] : 0.0};
                U.rhoE = state.rhoE[idx];
                PrimitiveState W = eos_->toPrimitive(U);
                state.velU[idx] = W.u[0];
                if (dim >= 2) state.velV[idx] = W.u[1];
                if (dim >= 3) state.velW[idx] = W.u[2];
                state.pres[idx] = W.p;
                state.temp[idx] = W.T;
            }
        }
    }

    return dt;
}

double SemiImplicitSolver::computeAdvectiveTimeStep(const RectilinearMesh& mesh, const SolutionState& state) const {
    double maxSpeed = 0.0;
    double minDx = std::numeric_limits<double>::max();
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                double dxMin = mesh.dx(i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh.dz(k));
                minDx = std::min(minDx, dxMin);

                std::size_t idx = mesh.index(i, j, k);
                double speed2 = state.velU[idx] * state.velU[idx];
                if (dim >= 2) speed2 += state.velV[idx] * state.velV[idx];
                if (dim >= 3) speed2 += state.velW[idx] * state.velW[idx];
                double u = std::sqrt(speed2);
                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return params_.maxDt;
    return params_.cfl * minDx / maxSpeed;
}

void SemiImplicitSolver::advectionStep(const RectilinearMesh& mesh, const SolutionState& state, double dt) {
    int dim = mesh.dim();

    std::copy(state.rho.begin(),  state.rho.end(),  rhoStar_.begin());
    std::copy(state.rhoU.begin(), state.rhoU.end(), rhoUStar_.begin());
    if (dim >= 2) std::copy(state.rhoV.begin(), state.rhoV.end(), rhoVStar_.begin());
    if (dim >= 3) std::copy(state.rhoW.begin(), state.rhoW.end(), rhoWStar_.begin());
    std::copy(state.rhoE.begin(), state.rhoE.end(), rhoEStar_.begin());

    // --- X-direction fluxes ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                std::size_t idxL = mesh.index(i - 1, j, k);
                std::size_t idxR = mesh.index(i, j, k);

                PrimitiveState left  = state.getPrimitiveState(idxL);
                PrimitiveState right = state.getPrimitiveState(idxR);

                RiemannFlux flux = riemannSolver_->computeFlux(
                    left, right, {1.0, 0.0, 0.0});

                double area = mesh.faceAreaX(j, k);

                if (i >= 1) {
                    double coeff = dt * area / mesh.cellVolume(i - 1, j, k);
                    rhoStar_[idxL]  -= coeff * flux.massFlux;
                    rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                    rhoEStar_[idxL] -= coeff * flux.energyFlux;
                }

                if (i < mesh.nx()) {
                    double coeff = dt * area / mesh.cellVolume(i, j, k);
                    rhoStar_[idxR]  += coeff * flux.massFlux;
                    rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                    rhoEStar_[idxR] += coeff * flux.energyFlux;
                }
            }
        }
    }

    // --- Y-direction fluxes ---
    if (dim >= 2) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j <= mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j - 1, k);
                    std::size_t idxR = mesh.index(i, j, k);

                    PrimitiveState left  = state.getPrimitiveState(idxL);
                    PrimitiveState right = state.getPrimitiveState(idxR);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 1.0, 0.0});

                    double area = mesh.faceAreaY(i, k);

                    if (j >= 1) {
                        double coeff = dt * area / mesh.cellVolume(i, j - 1, k);
                        rhoStar_[idxL]  -= coeff * flux.massFlux;
                        rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                        rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                        rhoEStar_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (j < mesh.ny()) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k);
                        rhoStar_[idxR]  += coeff * flux.massFlux;
                        rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                        rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                        rhoEStar_[idxR] += coeff * flux.energyFlux;
                    }
                }
            }
        }
    }

    // --- Z-direction fluxes ---
    if (dim >= 3) {
        for (int k = 0; k <= mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j, k - 1);
                    std::size_t idxR = mesh.index(i, j, k);

                    PrimitiveState left  = state.getPrimitiveState(idxL);
                    PrimitiveState right = state.getPrimitiveState(idxR);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 0.0, 1.0});

                    double area = mesh.faceAreaZ(i, j);

                    if (k >= 1) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k - 1);
                        rhoStar_[idxL]  -= coeff * flux.massFlux;
                        rhoUStar_[idxL] -= coeff * flux.momentumFlux[0];
                        rhoVStar_[idxL] -= coeff * flux.momentumFlux[1];
                        rhoWStar_[idxL] -= coeff * flux.momentumFlux[2];
                        rhoEStar_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (k < mesh.nz()) {
                        double coeff = dt * area / mesh.cellVolume(i, j, k);
                        rhoStar_[idxR]  += coeff * flux.massFlux;
                        rhoUStar_[idxR] += coeff * flux.momentumFlux[0];
                        rhoVStar_[idxR] += coeff * flux.momentumFlux[1];
                        rhoWStar_[idxR] += coeff * flux.momentumFlux[2];
                        rhoEStar_[idxR] += coeff * flux.energyFlux;
                    }
                }
            }
        }
    }
}

void SemiImplicitSolver::advectPressure(const RectilinearMesh& mesh, const SolutionState& state, double dt) {
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double p = state.pres[idx];
                double advection = 0.0;

                {
                    double u = state.velU[idx];
                    double pxm = state.pres[mesh.index(i - 1, j, k)];
                    double pxp = state.pres[mesh.index(i + 1, j, k)];
                    if (u > 0) {
                        advection += u * (p - pxm) / mesh.dx(i);
                    } else {
                        advection += u * (pxp - p) / mesh.dx(i);
                    }
                }

                if (mesh.dim() >= 2) {
                    double v = state.velV[idx];
                    double pym = state.pres[mesh.index(i, j - 1, k)];
                    double pyp = state.pres[mesh.index(i, j + 1, k)];
                    if (v > 0) {
                        advection += v * (p - pym) / mesh.dy(j);
                    } else {
                        advection += v * (pyp - p) / mesh.dy(j);
                    }
                }

                if (mesh.dim() >= 3) {
                    double w = state.velW[idx];
                    double pzm = state.pres[mesh.index(i, j, k - 1)];
                    double pzp = state.pres[mesh.index(i, j, k + 1)];
                    if (w > 0) {
                        advection += w * (p - pzm) / mesh.dz(k);
                    } else {
                        advection += w * (pzp - p) / mesh.dz(k);
                    }
                }

                pAdvected_[idx] = p - dt * advection;
            }
        }
    }
}

void SemiImplicitSolver::solveIGR(const RectilinearMesh& mesh, SolutionState& state) {
    if (!igrSolver_) return;

    computeVelocityGradients(mesh, state);

    int nNeighbors = 2 * mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double dxi = mesh.dx(i);

                double alpha = igrSolver_->computeAlpha(dxi);
                double rhs = igrSolver_->computeIGRRhs(gradU_[idx], alpha);

                double rhoCell = state.rho[idx];

                std::array<double, 6> neighborSigmaRho = {0, 0, 0, 0, 0, 0};

                std::size_t xm = mesh.index(i - 1, j, k);
                std::size_t xp = mesh.index(i + 1, j, k);
                neighborSigmaRho[0] = state.sigma[xm] / std::max(state.rho[xm], 1e-14);
                neighborSigmaRho[1] = state.sigma[xp] / std::max(state.rho[xp], 1e-14);

                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    neighborSigmaRho[2] = state.sigma[ym] / std::max(state.rho[ym], 1e-14);
                    neighborSigmaRho[3] = state.sigma[yp] / std::max(state.rho[yp], 1e-14);
                }

                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    neighborSigmaRho[4] = state.sigma[zm] / std::max(state.rho[zm], 1e-14);
                    neighborSigmaRho[5] = state.sigma[zp] / std::max(state.rho[zp], 1e-14);
                }

                state.sigma[idx] = igrSolver_->solveEntropicPressure(
                    rhs, rhoCell, alpha, dxi,
                    state.sigma[idx],
                    neighborSigmaRho,
                    nNeighbors);
            }
        }
    }
}

void SemiImplicitSolver::solvePressure(const RectilinearMesh& mesh, SolutionState& state, double dt) {
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                PrimitiveState W = state.getPrimitiveState(idx);
                double c = eos_->soundSpeed(W);
                rhoc2_[idx] = state.rho[idx] * c * c;
            }
        }
    }

    computeDivergence(mesh, state, divUstar_);

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                pressureRhs_[idx] = pAdvected_[idx] - rhoc2_[idx] * dt * divUstar_[idx];
            }
        }
    }

    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                pressure_[mesh.index(i, j, k)] = state.pres[mesh.index(i, j, k)];

    lastPressureIters_ = pressureSolver_->solve(
        mesh, state.rho, rhoc2_, pressureRhs_, pressure_,
        dt, params_.pressureTol, params_.maxPressureIters);

    mesh.fillScalarGhosts(pressure_);
    mesh.fillScalarGhosts(state.sigma);

    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                state.pres[mesh.index(i, j, k)] = pressure_[mesh.index(i, j, k)];
}

void SemiImplicitSolver::correctionStep(const RectilinearMesh& mesh, SolutionState& state, double dt) {
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                {
                    std::size_t xm = mesh.index(i - 1, j, k);
                    std::size_t xp = mesh.index(i + 1, j, k);
                    double pTotL = 0.5 * ((pressure_[xm] + state.sigma[xm]) +
                                           (pressure_[idx] + state.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + state.sigma[idx]) +
                                           (pressure_[xp] + state.sigma[xp]));
                    state.rhoU[idx] = rhoUStar_[idx] - dt * (pTotR - pTotL) / mesh.dx(i);
                }

                if (dim >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double pTotL = 0.5 * ((pressure_[ym] + state.sigma[ym]) +
                                           (pressure_[idx] + state.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + state.sigma[idx]) +
                                           (pressure_[yp] + state.sigma[yp]));
                    state.rhoV[idx] = rhoVStar_[idx] - dt * (pTotR - pTotL) / mesh.dy(j);
                }

                if (dim >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double pTotL = 0.5 * ((pressure_[zm] + state.sigma[zm]) +
                                           (pressure_[idx] + state.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + state.sigma[idx]) +
                                           (pressure_[zp] + state.sigma[zp]));
                    state.rhoW[idx] = rhoWStar_[idx] - dt * (pTotR - pTotL) / mesh.dz(k);
                }
            }
        }
    }

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                ConservativeState U;
                U.rho = state.rho[idx];
                U.rhoU = {state.rhoU[idx],
                          (dim >= 2) ? state.rhoV[idx] : 0.0,
                          (dim >= 3) ? state.rhoW[idx] : 0.0};
                U.rhoE = state.rhoE[idx];
                PrimitiveState W = eos_->toPrimitive(U);
                state.velU[idx] = W.u[0];
                if (dim >= 2) state.velV[idx] = W.u[1];
                if (dim >= 3) state.velW[idx] = W.u[2];
            }
        }
    }
    mesh.applyBoundaryConditions(state);

    // Reconstruct total energy from solved pressure and corrected velocity.
    // This avoids the unstable div(p*u) finite difference which can produce
    // negative internal energy at strong discontinuities.
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                PrimitiveState W;
                W.rho   = state.rho[idx];
                W.u[0]  = state.velU[idx];
                if (dim >= 2) W.u[1] = state.velV[idx];
                if (dim >= 3) W.u[2] = state.velW[idx];
                W.p     = pressure_[idx];
                W.sigma = state.sigma[idx];

                ConservativeState U = eos_->toConservative(W);
                state.rhoE[idx] = U.rhoE;
            }
        }
    }
}

void SemiImplicitSolver::computeDivergence(
    const RectilinearMesh& mesh, const SolutionState& state, std::vector<double>& divU)
{
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double div = 0.0;

                {
                    std::size_t xm = mesh.index(i - 1, j, k);
                    std::size_t xp = mesh.index(i + 1, j, k);
                    double uFaceL = 0.5 * (state.velU[xm] + state.velU[idx]);
                    double uFaceR = 0.5 * (state.velU[idx] + state.velU[xp]);
                    div += (uFaceR - uFaceL) / mesh.dx(i);
                }

                if (mesh.dim() >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double vFaceL = 0.5 * (state.velV[ym] + state.velV[idx]);
                    double vFaceR = 0.5 * (state.velV[idx] + state.velV[yp]);
                    div += (vFaceR - vFaceL) / mesh.dy(j);
                }

                if (mesh.dim() >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double wFaceL = 0.5 * (state.velW[zm] + state.velW[idx]);
                    double wFaceR = 0.5 * (state.velW[idx] + state.velW[zp]);
                    div += (wFaceR - wFaceL) / mesh.dz(k);
                }

                divU[idx] = div;
            }
        }
    }
}

void SemiImplicitSolver::computeVelocityGradients(const RectilinearMesh& mesh, const SolutionState& state) {
    int dim = mesh.dim();
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                std::size_t xm = mesh.index(i - 1, j, k);
                std::size_t xp = mesh.index(i + 1, j, k);
                std::array<double, 3> u_xm = {state.velU[xm],
                                               (dim >= 2) ? state.velV[xm] : 0.0,
                                               (dim >= 3) ? state.velW[xm] : 0.0};
                std::array<double, 3> u_xp = {state.velU[xp],
                                               (dim >= 2) ? state.velV[xp] : 0.0,
                                               (dim >= 3) ? state.velW[xp] : 0.0};

                std::array<double, 3> u_ym, u_yp;
                double dyj;
                if (dim >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    u_ym = {state.velU[ym], state.velV[ym],
                            (dim >= 3) ? state.velW[ym] : 0.0};
                    u_yp = {state.velU[yp], state.velV[yp],
                            (dim >= 3) ? state.velW[yp] : 0.0};
                    dyj = mesh.dy(j);
                } else {
                    u_ym = u_yp = {state.velU[idx], 0.0, 0.0};
                    dyj = 1.0;
                }

                std::array<double, 3> u_zm, u_zp;
                double dzk;
                if (dim >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    u_zm = {state.velU[zm], state.velV[zm], state.velW[zm]};
                    u_zp = {state.velU[zp], state.velV[zp], state.velW[zp]};
                    dzk = mesh.dz(k);
                } else {
                    u_zm = u_zp = {state.velU[idx],
                                   (dim >= 2) ? state.velV[idx] : 0.0,
                                   0.0};
                    dzk = 1.0;
                }

                gradU_[idx] = IGRSolver::computeVelocityGradient(
                    u_xm, u_xp, u_ym, u_yp, u_zm, u_zp,
                    mesh.dx(i), dyj, dzk);
            }
        }
    }
}

} // namespace SemiImplicitFV
