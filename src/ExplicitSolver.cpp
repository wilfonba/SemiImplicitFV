#include "ExplicitSolver.hpp"
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
}

double ExplicitSolver::step(const SimulationConfig& config,
        const RectilinearMesh& mesh, SolutionState& state, double targetDt) {
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
    int dim = mesh.dim();

    // Compute RHS = -div(F)
    computeRHS(config, mesh, state);

    // Forward Euler: U^{n+1} = U^n + dt * RHS
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                state.rho[idx]  += dt * rhsRho_[idx];
                state.rhoU[idx] += dt * rhsRhoU_[idx];
                if (dim >= 2) state.rhoV[idx] += dt * rhsRhoV_[idx];
                if (dim >= 3) state.rhoW[idx] += dt * rhsRhoW_[idx];
                state.rhoE[idx] += dt * rhsRhoE_[idx];
            }
        }
    }

    return dt;
}

void ExplicitSolver::computeRHS(const SimulationConfig& config,
        const RectilinearMesh& mesh, SolutionState& state) {
    state.convertConservativeToPrimitiveVariables(mesh, eos_);
    mesh.applyBoundaryConditions(state, VarSet::PRIM);
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

double ExplicitSolver::computeAcousticTimeStep(const RectilinearMesh& mesh, const SolutionState& state) const {
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
