#include "ExplicitSolver.hpp"
#include "SimulationConfig.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace SemiImplicitFV {

ExplicitSolver::ExplicitSolver(
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<EquationOfState> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const ExplicitParams& params
)
    : riemannSolver_(std::move(riemannSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(params)
    , reconstructor_(params.reconOrder)
{}

void ExplicitSolver::ensureStorage(const RectilinearMesh& mesh) {
    std::size_t n = mesh.totalCells();
    if (rhsRho_.size() == n) return;

    int dim = mesh.dim();
    rhsRho_.resize(n);
    rhsRhoU_.resize(n);
    if (dim >= 2) rhsRhoV_.resize(n); else rhsRhoV_.clear();
    if (dim >= 3) rhsRhoW_.resize(n); else rhsRhoW_.clear();
    rhsRhoE_.resize(n);
    if (igrSolver_) {
        gradU_.resize(n);
    }
}

double ExplicitSolver::step(const SimulationConfig& config,
                            const RectilinearMesh& mesh,
                            SolutionState& state,
                            double targetDt) {
    double dt;
    if (params_.constDt > 0) {
        dt = params_.constDt;
    } else {
        dt = computeAcousticTimeStep(mesh, state);
    }

    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }

    ensureStorage(mesh);

    for (int s = 1; s <= config.RKOrder; ++s) {

        state.convertConservativeToPrimitiveVariables(mesh, eos_);
        mesh.applyBoundaryConditions(state, VarSet::PRIM);

        if (config.useIGR && igrSolver_) solveIGR(config, mesh, state);

        computeRHS(config, mesh, state);

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);

                    // Save U^n before first stage modifies it
                    if (s == 1 && config.RKOrder > 1) {
                        state.saveConservativeCell(idx);
                    }

                    // Forward Euler update
                    state.rho[idx]  += dt * rhsRho_[idx];
                    state.rhoU[idx] += dt * rhsRhoU_[idx];
                    if (config.dim >= 2) state.rhoV[idx] += dt * rhsRhoV_[idx];
                    if (config.dim >= 3) state.rhoW[idx] += dt * rhsRhoW_[idx];
                    state.rhoE[idx] += dt * rhsRhoE_[idx];

                    // SSP-RK blending: U = alpha * U^n + (1-alpha) * U
                    if (config.RKOrder > 1 && s >= 2) {
                        double alpha = sspRKBlendCoeff(config, s);
                        state.blendConservativeCell(idx, alpha);
                    }
                }
            }
        }
    }

    return dt;
}

double ExplicitSolver::sspRKBlendCoeff(const SimulationConfig& config, int stage) const {
    // SSP-RK2: U^(n+1) = 1/2 * U^n + 1/2 * [U^(1) + dt*L(U^(1))]
    if (config.RKOrder == 2 && stage == 2) return 0.5;
    // SSP-RK3: stage 2: U^(2) = 3/4 * U^n + 1/4 * [U^(1) + dt*L(U^(1))]
    if (config.RKOrder == 3 && stage == 2) return 3.0 / 4.0;
    // SSP-RK3: stage 3: U^(n+1) = 1/3 * U^n + 2/3 * [U^(2) + dt*L(U^(2))]
    if (config.RKOrder == 3 && stage == 3) return 1.0 / 3.0;
    return 0.0;
}

void ExplicitSolver::solveIGR(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state) {
    if (!igrSolver_) return;

    computeVelocityGradients(config, mesh, state);

    igrSolver_->solveEntropicPressure(config, mesh, state, gradU_);
}

void ExplicitSolver::computeVelocityGradients(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state) {
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
                    mesh.dx(i), dyj, dzk, config.dim);
            }
        }
    }
}


void ExplicitSolver::computeRHS(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state) {
    reconstructor_.reconstruct(config, mesh, state);

    int dim = mesh.dim();

    // Zero RHS arrays
    std::fill(rhsRho_.begin(),  rhsRho_.end(),  0.0);
    std::fill(rhsRhoU_.begin(), rhsRhoU_.end(), 0.0);
    if (dim >= 2) std::fill(rhsRhoV_.begin(), rhsRhoV_.end(), 0.0);
    if (dim >= 3) std::fill(rhsRhoW_.begin(), rhsRhoW_.end(), 0.0);
    std::fill(rhsRhoE_.begin(), rhsRhoE_.end(), 0.0);

    // --- X-direction fluxes ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                std::size_t f = reconstructor_.xFaceIndex(i, j, k);
                const PrimitiveState& left  = reconstructor_.xFaceLeft(f);
                const PrimitiveState& right = reconstructor_.xFaceRight(f);

                RiemannFlux flux = riemannSolver_->computeFlux(
                    left, right, {1.0, 0.0, 0.0});

                double area = mesh.faceAreaX(j, k);

                if (i >= 1) {
                    std::size_t idxL = mesh.index(i - 1, j, k);
                    double coeff = area / mesh.cellVolume(i - 1, j, k);
                    rhsRho_[idxL]  -= coeff * flux.massFlux;
                    rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                    rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                }

                if (i < mesh.nx()) {
                    std::size_t idxR = mesh.index(i, j, k);
                    double coeff = area / mesh.cellVolume(i, j, k);
                    rhsRho_[idxR]  += coeff * flux.massFlux;
                    rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                    rhsRhoE_[idxR] += coeff * flux.energyFlux;
                }
            }
        }
    }

    // --- Y-direction fluxes ---
    if (dim >= 2) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j <= mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t f = reconstructor_.yFaceIndex(i, j, k);
                    const PrimitiveState& left  = reconstructor_.yFaceLeft(f);
                    const PrimitiveState& right = reconstructor_.yFaceRight(f);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 1.0, 0.0});

                    double area = mesh.faceAreaY(i, k);

                    if (j >= 1) {
                        std::size_t idxL = mesh.index(i, j - 1, k);
                        double coeff = area / mesh.cellVolume(i, j - 1, k);
                        rhsRho_[idxL]  -= coeff * flux.massFlux;
                        rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (j < mesh.ny()) {
                        std::size_t idxR = mesh.index(i, j, k);
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRho_[idxR]  += coeff * flux.massFlux;
                        rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxR] += coeff * flux.energyFlux;
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
                    std::size_t f = reconstructor_.zFaceIndex(i, j, k);
                    const PrimitiveState& left  = reconstructor_.zFaceLeft(f);
                    const PrimitiveState& right = reconstructor_.zFaceRight(f);

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 0.0, 1.0});

                    double area = mesh.faceAreaZ(i, j);

                    if (k >= 1) {
                        std::size_t idxL = mesh.index(i, j, k - 1);
                        double coeff = area / mesh.cellVolume(i, j, k - 1);
                        rhsRho_[idxL]  -= coeff * flux.massFlux;
                        rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                        rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                    }

                    if (k < mesh.nz()) {
                        std::size_t idxR = mesh.index(i, j, k);
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRho_[idxR]  += coeff * flux.massFlux;
                        rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                        rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxR] += coeff * flux.energyFlux;
                    }
                }
            }
        }
    }
}

double ExplicitSolver::computeAcousticTimeStep(const RectilinearMesh& mesh, SolutionState& state) const {
    double maxSpeed = 0.0;
    double minDx = std::numeric_limits<double>::max();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                double dxMin = mesh.dx(i);
                if (mesh.dim() >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (mesh.dim() >= 3) dxMin = std::min(dxMin, mesh.dz(k));
                minDx = std::min(minDx, dxMin);

                std::size_t idx = mesh.index(i, j, k);
                double speed2 = state.velU[idx] * state.velU[idx];
                if (mesh.dim() >= 2) speed2 += state.velV[idx] * state.velV[idx];
                if (mesh.dim() >= 3) speed2 += state.velW[idx] * state.velW[idx];
                double u = std::sqrt(speed2);
                double c = eos_->soundSpeed(state.getPrimitiveState(idx));
                u += c;

                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return params_.maxDt;
    return params_.cfl * minDx / maxSpeed;
}

} // namespace SemiImplicitFV
