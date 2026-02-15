#include "SemiImplicitSolver.hpp"
#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "ViscousFlux.hpp"
#include "SurfaceTension.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace SemiImplicitFV {

SemiImplicitSolver::SemiImplicitSolver(
    const RectilinearMesh& mesh,
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<PressureSolver> pressureSolver,
    std::shared_ptr<EquationOfState> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const SimulationConfig& config
)
    : riemannSolver_(std::move(riemannSolver))
    , pressureSolver_(std::move(pressureSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(config.semiImplicitParams)
    , lastPressureIters_(0)
    , reconstructor_(config.reconOrder, config.wenoEps,
                     eos_->gamma(), eos_->pInf())
{
    // Determine solver type for non-virtual dispatch
    const std::string& sname = riemannSolver_->name();
    if (sname == "HLLC") solverType_ = RiemannSolverType::HLLC;
    else if (sname == "Rusanov") solverType_ = RiemannSolverType::Rusanov;
    else solverType_ = RiemannSolverType::LF;

    fluxConfig_.dim = config.dim;
    fluxConfig_.includePressure = !config.semiImplicit;
    fluxConfig_.useIGR = config.useIGR;
    fluxConfig_.nPhases = config.isMultiPhase() ? config.multiPhaseParams.nPhases : 0;

    gamma_ = eos_->gamma();
    pInf_ = eos_->pInf();

    std::size_t n = mesh.totalCells();
    int dim = mesh.dim();

    pressureRhs_.resize(n);
    pressure_.resize(n);
    divUstar_.resize(n);

    rhsRho_.resize(n);
    rhsRhoU_.resize(n);
    if (dim >= 2) rhsRhoV_.resize(n);
    if (dim >= 3) rhsRhoW_.resize(n);
    rhsRhoE_.resize(n);
    rhsPadvected_.resize(n);

    reconstructor_.allocate(mesh);

    divU_.resize(n);

    if (config.isMultiPhase()) {
        int nPhases = config.multiPhaseParams.nPhases;
        rhsAlphaRho_.resize(nPhases);
        for (int ph = 0; ph < nPhases; ++ph)
            rhsAlphaRho_[ph].resize(n);
        rhsAlpha_.resize(nPhases);
        for (int ph = 0; ph < nPhases; ++ph)
            rhsAlpha_[ph].resize(n);
        scratchAlphas_.resize(nPhases);
        scratchAlphaRhos_.resize(nPhases);
    }

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
                // density already updated in-place
                state.rhoU[idx] = state.rhoUStar[idx];
                if (dim >= 2) state.rhoV[idx] = state.rhoVStar[idx];
                if (dim >= 3) state.rhoW[idx] = state.rhoWStar[idx];
                state.rhoE[idx] = state.rhoEstar[idx];

                double rhoSafe = std::max(state.rho[idx], 1e-14);
                state.velU[idx] = state.rhoUStar[idx] / rhoSafe;
                if (dim >= 2) state.velV[idx] = state.rhoVStar[idx] / rhoSafe;
                if (dim >= 3) state.velW[idx] = state.rhoWStar[idx] / rhoSafe;
                double ke = 0.5 * rhoSafe * state.velU[idx] * state.velU[idx];
                if (dim >= 2) ke += 0.5 * rhoSafe * state.velV[idx] * state.velV[idx];
                if (dim >= 3) ke += 0.5 * rhoSafe * state.velW[idx] * state.velW[idx];
                state.pres[idx] = (gamma_ - 1.0) * (state.rhoEstar[idx] - ke) - gamma_ * pInf_;
            }
        }
    }
    mesh.applyBoundaryConditions(state, VarSet::PRIM, *halo_);
}

double SemiImplicitSolver::step(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state,
        double targetDt) {
    double dt = SemiImplicitFV::computeAdvectiveTimeStep(
        mesh, state, params_.cfl, params_.maxDt, halo_->mpi().comm());
    if (config.hasViscosity()) {
        dt = std::min(dt, computeViscousDt(mesh, state,
            config.viscousParams.mu, params_.cfl, params_.maxDt, halo_->mpi().comm()));
    }
    if (config.hasSurfaceTension()) {
        dt = std::min(dt, computeCapillaryDt(mesh, state,
            config.surfaceTensionParams.sigma, params_.cfl, params_.maxDt, halo_->mpi().comm()));
    }
    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }
    dt = std::clamp(dt, params_.minDt, params_.maxDt);

    // TVD RK coefficients: U = (c1*U_current + c2*U_saved + c3*dt*RHS) / c4
    std::array<std::array<double, 4>, 3> rk_coef;
    if (config.RKOrder == 1) {
        rk_coef[0] = {1.0, 0.0, 1.0, 1.0};
    } else if (config.RKOrder == 2) {
        rk_coef[0] = {1.0, 0.0, 1.0, 1.0};
        rk_coef[1] = {1.0, 1.0, 1.0, 2.0};
    } else {
        rk_coef[0] = {1.0, 0.0, 1.0, 1.0};
        rk_coef[1] = {1.0, 3.0, 1.0, 4.0};
        rk_coef[2] = {2.0, 1.0, 2.0, 3.0};
    }

    const bool multiPhase = config.isMultiPhase();
    const int nPhases = multiPhase ? config.multiPhaseParams.nPhases : 0;
    const double alphaMin = multiPhase ? config.multiPhaseParams.alphaMin : 0.0;

    for (int s = 0; s < config.RKOrder; ++s) {

        if (multiPhase)
            MixtureEOS::convertConservativeToPrimitive(mesh, state, config.multiPhaseParams);
        else
            state.convertConservativeToPrimitiveVariables(mesh, eos_);
        mesh.applyBoundaryConditions(state, VarSet::PRIM, *halo_);

        if (config.useIGR && igrSolver_) solveIGR(config, mesh, state);

        computeRHS(config, mesh, state);

        double c1 = rk_coef[s][0];
        double c2 = rk_coef[s][1];
        double c3 = rk_coef[s][2];
        double c4 = rk_coef[s][3];

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);

                    if (s == 0 && config.RKOrder > 1) {
                        state.saveConservativeCell(idx);
                    }

                    state.pAdvected[idx] = state.pres[idx];

                    double rho0  = (config.RKOrder > 1) ? state.rho0[idx]  : 0.0;
                    double rhoU0 = (config.RKOrder > 1) ? state.rhoU0[idx] : 0.0;
                    double rhoE0 = (config.RKOrder > 1) ? state.rhoE0[idx] : 0.0;

                    state.rho[idx]      = (c1 * state.rho[idx]  + c2 * rho0  + c3 * dt * rhsRho_[idx])   / c4;
                    state.rhoUStar[idx] = (c1 * state.rhoU[idx] + c2 * rhoU0 + c3 * dt * rhsRhoU_[idx])  / c4;
                    if (config.dim >= 2) {
                        double rhoV0 = (config.RKOrder > 1) ? state.rhoV0[idx] : 0.0;
                        state.rhoVStar[idx] = (c1 * state.rhoV[idx] + c2 * rhoV0 + c3 * dt * rhsRhoV_[idx]) / c4;
                    }
                    if (config.dim >= 3) {
                        double rhoW0 = (config.RKOrder > 1) ? state.rhoW0[idx] : 0.0;
                        state.rhoWStar[idx] = (c1 * state.rhoW[idx] + c2 * rhoW0 + c3 * dt * rhsRhoW_[idx]) / c4;
                    }
                    state.rhoEstar[idx] = (c1 * state.rhoE[idx] + c2 * rhoE0 + c3 * dt * rhsRhoE_[idx])  / c4;
                    double p0 = (config.RKOrder > 1) ? state.pres0[idx] : 0.0;
                    state.pAdvected[idx] = (c1 * state.pAdvected[idx] + c2 * p0 + c3 * dt * rhsPadvected_[idx]) / c4;

                    // Multi-phase RK update
                    if (multiPhase) {
                        if (config.RKOrder == 1) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alphaRho[ph][idx] += dt * rhsAlphaRho_[ph][idx];
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alpha[ph][idx] += dt * rhsAlpha_[ph][idx];
                        } else {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alphaRho[ph][idx] = (c1 * state.alphaRho[ph][idx] + c2 * state.alphaRho0[ph][idx] + c3 * dt * rhsAlphaRho_[ph][idx]) / c4;
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alpha[ph][idx] = (c1 * state.alpha[ph][idx] + c2 * state.alpha0[ph][idx] + c3 * dt * rhsAlpha_[ph][idx]) / c4;
                        }

                        // Recompute rho, clamp and normalize alphas
                        double rhoSum = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            state.alphaRho[ph][idx] = std::max(state.alphaRho[ph][idx], 1e-14);
                            rhoSum += state.alphaRho[ph][idx];
                        }
                        state.rho[idx] = rhoSum;

                        for (int ph = 0; ph < nPhases; ++ph)
                            state.alpha[ph][idx] = std::max(state.alpha[ph][idx], alphaMin);
                        double alphaSum = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph)
                            alphaSum += state.alpha[ph][idx];
                        for (int ph = 0; ph < nPhases; ++ph)
                            state.alpha[ph][idx] /= alphaSum;
                    }

                    // Compute star velocities for divergence computation
                    double rhoSafe = std::max(state.rho[idx], 1e-14);
                    state.velU[idx] = state.rhoUStar[idx] / rhoSafe;
                    if (config.dim >= 2) state.velV[idx] = state.rhoVStar[idx] / rhoSafe;
                    if (config.dim >= 3) state.velW[idx] = state.rhoWStar[idx] / rhoSafe;
                }
            }
        }

        // Re-fill ghost cells (density + star velocity) before pressure solve
        mesh.applyBoundaryConditions(state, VarSet::PRIM, *halo_);

        solvePressure(config, mesh, state, dt);
        correctionStep(config, mesh, state, dt);
    }

    return dt;
}

void SemiImplicitSolver::computeRHS(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state) {
    reconstructor_.reconstruct(config, mesh, state);

    int dim = mesh.dim();
    const bool multiPhase = config.isMultiPhase();
    const int nPhases = multiPhase ? config.multiPhaseParams.nPhases : 0;

    // Zero RHS arrays
    std::fill(rhsRho_.begin(),  rhsRho_.end(),  0.0);
    std::fill(rhsRhoU_.begin(), rhsRhoU_.end(), 0.0);
    if (dim >= 2) std::fill(rhsRhoV_.begin(), rhsRhoV_.end(), 0.0);
    if (dim >= 3) std::fill(rhsRhoW_.begin(), rhsRhoW_.end(), 0.0);
    std::fill(rhsRhoE_.begin(), rhsRhoE_.end(), 0.0);
    std::fill(rhsPadvected_.begin(), rhsPadvected_.end(), 0.0);
    std::fill(divU_.begin(), divU_.end(), 0.0);

    if (multiPhase) {
        for (int ph = 0; ph < nPhases; ++ph)
            std::fill(rhsAlphaRho_[ph].begin(), rhsAlphaRho_[ph].end(), 0.0);
        for (int ph = 0; ph < nPhases; ++ph)
            std::fill(rhsAlpha_[ph].begin(), rhsAlpha_[ph].end(), 0.0);
    }

    // --- X-direction fluxes ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                std::size_t f = reconstructor_.xFaceIndex(i, j, k);
                const PrimitiveState& left  = reconstructor_.xFaceLeft(f);
                const PrimitiveState& right = reconstructor_.xFaceRight(f);

                RiemannFlux flux = computeFluxDirect(solverType_,
                    left, right, {1.0, 0.0, 0.0}, fluxConfig_);

                double area = mesh.faceAreaX(j, k);

                std::size_t upwindIdx = 0;
                if (multiPhase)
                    upwindIdx = (flux.massFlux >= 0) ? mesh.index(i - 1, j, k) : mesh.index(i, j, k);

                if (i >= 1) {
                    std::size_t idxL = mesh.index(i - 1, j, k);
                    double coeff = area / mesh.cellVolume(i - 1, j, k);
                    rhsRho_[idxL]  -= coeff * flux.massFlux;
                    rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                    rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                    rhsPadvected_[idxL] -= coeff * flux.pressureFlux;
                    divU_[idxL] += coeff * flux.faceVelocity;

                    if (multiPhase) {
                        double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                            rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                    }
                }

                if (i < mesh.nx()) {
                    std::size_t idxR = mesh.index(i, j, k);
                    double coeff = area / mesh.cellVolume(i, j, k);
                    rhsRho_[idxR]  += coeff * flux.massFlux;
                    rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                    if (dim >= 2) rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                    if (dim >= 3) rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                    rhsRhoE_[idxR] += coeff * flux.energyFlux;
                    rhsPadvected_[idxR] += coeff * flux.pressureFlux;
                    divU_[idxR] -= coeff * flux.faceVelocity;

                    if (multiPhase) {
                        double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                            rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                    }
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

                    RiemannFlux flux = computeFluxDirect(solverType_,
                        left, right, {0.0, 1.0, 0.0}, fluxConfig_);

                    double area = mesh.faceAreaY(i, k);

                    std::size_t upwindIdx = 0;
                    if (multiPhase)
                        upwindIdx = (flux.massFlux >= 0) ? mesh.index(i, j - 1, k) : mesh.index(i, j, k);

                    if (j >= 1) {
                        std::size_t idxL = mesh.index(i, j - 1, k);
                        double coeff = area / mesh.cellVolume(i, j - 1, k);
                        rhsRho_[idxL]  -= coeff * flux.massFlux;
                        rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                        rhsPadvected_[idxL] -= coeff * flux.pressureFlux;
                        divU_[idxL] += coeff * flux.faceVelocity;

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                        }
                    }

                    if (j < mesh.ny()) {
                        std::size_t idxR = mesh.index(i, j, k);
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRho_[idxR]  += coeff * flux.massFlux;
                        rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                        if (dim >= 3) rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxR] += coeff * flux.energyFlux;
                        rhsPadvected_[idxR] += coeff * flux.pressureFlux;
                        divU_[idxR] -= coeff * flux.faceVelocity;

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                        }
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

                    RiemannFlux flux = computeFluxDirect(solverType_,
                        left, right, {0.0, 0.0, 1.0}, fluxConfig_);

                    double area = mesh.faceAreaZ(i, j);

                    std::size_t upwindIdx = 0;
                    if (multiPhase)
                        upwindIdx = (flux.massFlux >= 0) ? mesh.index(i, j, k - 1) : mesh.index(i, j, k);

                    if (k >= 1) {
                        std::size_t idxL = mesh.index(i, j, k - 1);
                        double coeff = area / mesh.cellVolume(i, j, k - 1);
                        rhsRho_[idxL]  -= coeff * flux.massFlux;
                        rhsRhoU_[idxL] -= coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxL] -= coeff * flux.momentumFlux[1];
                        rhsRhoW_[idxL] -= coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxL] -= coeff * flux.energyFlux;
                        rhsPadvected_[idxL] -= coeff * flux.pressureFlux;
                        divU_[idxL] += coeff * flux.faceVelocity;

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                        }
                    }

                    if (k < mesh.nz()) {
                        std::size_t idxR = mesh.index(i, j, k);
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRho_[idxR]  += coeff * flux.massFlux;
                        rhsRhoU_[idxR] += coeff * flux.momentumFlux[0];
                        rhsRhoV_[idxR] += coeff * flux.momentumFlux[1];
                        rhsRhoW_[idxR] += coeff * flux.momentumFlux[2];
                        rhsRhoE_[idxR] += coeff * flux.energyFlux;
                        rhsPadvected_[idxR] += coeff * flux.pressureFlux;
                        divU_[idxR] -= coeff * flux.faceVelocity;

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                        }
                    }
                }
            }
        }
    }

    // --- Alpha source term: α ∇·u ---
    if (multiPhase) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    for (int ph = 0; ph < nPhases; ++ph)
                        rhsAlpha_[ph][idx] += state.alpha[ph][idx] * divU_[idx];
                }
            }
        }
    }

    // --- Pressure advection source term: dp/dt = -∇·(pu) + p∇·u ---
    // The -∇·(pu) part was accumulated from flux.pressureFlux in the flux loops above.
    // Now add the p∇·u source term.
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                rhsPadvected_[idx] += state.pres[idx] * divU_[idx];
            }
        }
    }

    // --- Body force source terms ---
    if (config.hasBodyForce()) {
        const auto& bf = config.bodyForceParams;
        double accel_x = bf.a[0] + bf.b[0] * std::cos(bf.c[0] * config.time + bf.d[0]);
        double accel_y = bf.a[1] + bf.b[1] * std::cos(bf.c[1] * config.time + bf.d[1]);
        double accel_z = bf.a[2] + bf.b[2] * std::cos(bf.c[2] * config.time + bf.d[2]);

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);

                    rhsRhoU_[idx] += state.rho[idx] * accel_x;
                    if (dim >= 2) rhsRhoV_[idx] += state.rho[idx] * accel_y;
                    if (dim >= 3) rhsRhoW_[idx] += state.rho[idx] * accel_z;

                    double work = state.velU[idx] * accel_x;
                    if (dim >= 2) work += state.velV[idx] * accel_y;
                    if (dim >= 3) work += state.velW[idx] * accel_z;
                    rhsRhoE_[idx] += state.rho[idx] * work;
                }
            }
        }
    }

    // --- Viscous stress ---
    if (config.hasViscosity()) {
        addViscousFluxes(config, mesh, state,
                         rhsRhoU_, rhsRhoV_, rhsRhoW_, rhsRhoE_);
    }

    // --- Surface tension (capillary stress) ---
    if (config.hasSurfaceTension()) {
        addSurfaceTensionFluxes(config, mesh, state,
            config.surfaceTensionParams.sigma,
            rhsRhoU_, rhsRhoV_, rhsRhoW_, rhsRhoE_);
    }
}

void SemiImplicitSolver::solveIGR(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state) {
    if (!igrSolver_) return;

    computeVelocityGradients(config, mesh, state);

    igrSolver_->solveEntropicPressure(config, mesh, state, gradU_, *halo_);
}

void SemiImplicitSolver::solvePressure(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state, double dt) {
    const bool multiPhase = config.isMultiPhase();
    const auto& mp = config.multiPhaseParams;
    int nPhases = multiPhase ? mp.nPhases : 0;

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                if (multiPhase) {
                    for (int ph = 0; ph < nPhases; ++ph)
                        scratchAlphas_[ph] = state.alpha[ph][idx];
                    for (int ph = 0; ph < nPhases; ++ph)
                        scratchAlphaRhos_[ph] = state.alphaRho[ph][idx];
                    double c = MixtureEOS::mixtureSoundSpeed(
                        state.rho[idx], state.pres[idx],
                        scratchAlphas_.data(), scratchAlphaRhos_.data(),
                        nPhases, mp.phases.data());
                    state.rhoc2[idx] = state.rho[idx] * c * c;
                } else {
                    double c = std::sqrt(gamma_ * std::max(state.pres[idx] + pInf_, 1e-14)
                                         / std::max(state.rho[idx], 1e-14));
                    state.rhoc2[idx] = state.rho[idx] * c * c;
                }
            }
        }
    }

    computeDivergence(mesh, state, divUstar_);

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                pressureRhs_[idx] = state.pAdvected[idx] - state.rhoc2[idx] * dt * divUstar_[idx];
            }
        }
    }

    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                pressure_[mesh.index(i, j, k)] = state.pres[mesh.index(i, j, k)];

    lastPressureIters_ = pressureSolver_->solve(
        mesh, state.rho, state.rhoc2, pressureRhs_, pressure_,
        dt, params_.pressureTol, params_.maxPressureIters, *halo_);

    mesh.fillScalarGhosts(pressure_, *halo_);
    mesh.fillScalarGhosts(state.sigma, *halo_);

    for (int k = 0; k < mesh.nz(); ++k)
        for (int j = 0; j < mesh.ny(); ++j)
            for (int i = 0; i < mesh.nx(); ++i)
                state.pres[mesh.index(i, j, k)] = pressure_[mesh.index(i, j, k)];
}

void SemiImplicitSolver::correctionStep(const SimulationConfig& config, const RectilinearMesh& mesh, SolutionState& state, double dt) {
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
                    state.rhoU[idx] = state.rhoUStar[idx] - dt * (pTotR - pTotL) / mesh.dx(i);
                }

                if (dim >= 2) {
                    std::size_t ym = mesh.index(i, j - 1, k);
                    std::size_t yp = mesh.index(i, j + 1, k);
                    double pTotL = 0.5 * ((pressure_[ym] + state.sigma[ym]) +
                                           (pressure_[idx] + state.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + state.sigma[idx]) +
                                           (pressure_[yp] + state.sigma[yp]));
                    state.rhoV[idx] = state.rhoVStar[idx] - dt * (pTotR - pTotL) / mesh.dy(j);
                }

                if (dim >= 3) {
                    std::size_t zm = mesh.index(i, j, k - 1);
                    std::size_t zp = mesh.index(i, j, k + 1);
                    double pTotL = 0.5 * ((pressure_[zm] + state.sigma[zm]) +
                                           (pressure_[idx] + state.sigma[idx]));
                    double pTotR = 0.5 * ((pressure_[idx] + state.sigma[idx]) +
                                           (pressure_[zp] + state.sigma[zp]));
                    state.rhoW[idx] = state.rhoWStar[idx] - dt * (pTotR - pTotL) / mesh.dz(k);
                }
            }
        }
    }

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double rhoSafe = std::max(state.rho[idx], 1e-14);
                state.velU[idx] = state.rhoU[idx] / rhoSafe;
                if (dim >= 2) state.velV[idx] = state.rhoV[idx] / rhoSafe;
                if (dim >= 3) state.velW[idx] = state.rhoW[idx] / rhoSafe;
            }
        }
    }
    mesh.applyBoundaryConditions(state, VarSet::PRIM, *halo_);

    // Reconstruct total energy from solved pressure and corrected velocity.
    // This avoids the unstable div(p*u) finite difference which can produce
    // negative internal energy at strong discontinuities.
    const bool multiPhase = config.isMultiPhase();
    const auto& mp = config.multiPhaseParams;
    int nPhases = multiPhase ? mp.nPhases : 0;

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (multiPhase) {
                    double rho = state.rho[idx];
                    double ke = 0.5 * rho * state.velU[idx] * state.velU[idx];
                    if (dim >= 2) ke += 0.5 * rho * state.velV[idx] * state.velV[idx];
                    if (dim >= 3) ke += 0.5 * rho * state.velW[idx] * state.velW[idx];

                    for (int ph = 0; ph < nPhases; ++ph)
                        scratchAlphas_[ph] = state.alpha[ph][idx];

                    state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(
                        rho, pressure_[idx], scratchAlphas_.data(),
                        nPhases, ke, mp.phases.data());
                } else {
                    double rho = state.rho[idx];
                    double ke = 0.5 * rho * state.velU[idx] * state.velU[idx];
                    if (dim >= 2) ke += 0.5 * rho * state.velV[idx] * state.velV[idx];
                    if (dim >= 3) ke += 0.5 * rho * state.velW[idx] * state.velW[idx];
                    state.rhoE[idx] = (pressure_[idx] + gamma_ * pInf_) / (gamma_ - 1.0) + ke;
                }
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

void SemiImplicitSolver::computeVelocityGradients(const SimulationConfig& config, const RectilinearMesh& mesh, const SolutionState& state) {
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

} // namespace SemiImplicitFV
