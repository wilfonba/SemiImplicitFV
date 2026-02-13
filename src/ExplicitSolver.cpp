#include "ExplicitSolver.hpp"
#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "ViscousFlux.hpp"
#include "SurfaceTension.hpp"
#include "SimulationConfig.hpp"
#include "ImmersedBoundary.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace SemiImplicitFV {

ExplicitSolver::ExplicitSolver(
    const RectilinearMesh& mesh,
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<EquationOfState> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const SimulationConfig& config
)
    : riemannSolver_(std::move(riemannSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(config.explicitParams)
    , reconstructor_(config.reconOrder, config.wenoEps)
{
    std::size_t n = mesh.totalCells();
    int dim = mesh.dim();

    rhsRho_.resize(n);
    rhsRhoU_.resize(n);
    if (dim >= 2) rhsRhoV_.resize(n);
    if (dim >= 3) rhsRhoW_.resize(n);
    rhsRhoE_.resize(n);

    reconstructor_.allocate(mesh);

    if (config.isMultiPhase()) {
        int nPhases = config.multiPhaseParams.nPhases;
        rhsAlphaRho_.resize(nPhases);
        for (int ph = 0; ph < nPhases; ++ph)
            rhsAlphaRho_[ph].resize(n);
        rhsAlpha_.resize(nPhases - 1);
        for (int ph = 0; ph < nPhases - 1; ++ph)
            rhsAlpha_[ph].resize(n);
        divU_.resize(n);
    }

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
        dt = SemiImplicitFV::computeAcousticTimeStep(
            mesh, state, *eos_, config, params_.cfl, params_.maxDt, halo_->mpi().comm(), ibm_);
        if (config.hasViscosity()) {
            dt = std::min(dt, computeViscousDt(mesh, state,
                config.viscousParams.mu, params_.cfl, params_.maxDt, halo_->mpi().comm(), ibm_));
        }
        if (config.hasSurfaceTension()) {
            dt = std::min(dt, computeCapillaryDt(mesh, state,
                config.surfaceTensionParams.sigma, params_.cfl, params_.maxDt, halo_->mpi().comm(), ibm_));
        }
    }

    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }

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
        if (ibm_) ibm_->applyGhostCells(mesh, state, config.dim);

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
                    if (ibm_ && ibm_->isSolid(idx)) continue;

                    if (s == 0 && config.RKOrder > 1) {
                        state.saveConservativeCell(idx);
                    }

                    if (config.RKOrder == 1) {
                        state.rho[idx]  += dt * rhsRho_[idx];
                        state.rhoU[idx] += dt * rhsRhoU_[idx];
                        if (config.dim >= 2)
                            state.rhoV[idx] += dt * rhsRhoV_[idx];
                        if (config.dim >= 3)
                            state.rhoW[idx] += dt * rhsRhoW_[idx];
                        state.rhoE[idx] += dt * rhsRhoE_[idx];

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alphaRho[ph][idx] += dt * rhsAlphaRho_[ph][idx];
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                state.alpha[ph][idx] += dt * rhsAlpha_[ph][idx];
                        }
                    } else {
                        state.rho[idx]  = (c1 * state.rho[idx]  + c2 * state.rho0[idx]  + c3 * dt * rhsRho_[idx])  / c4;
                        state.rhoU[idx] = (c1 * state.rhoU[idx] + c2 * state.rhoU0[idx] + c3 * dt * rhsRhoU_[idx]) / c4;
                        if (config.dim >= 2)
                            state.rhoV[idx] = (c1 * state.rhoV[idx] + c2 * state.rhoV0[idx] + c3 * dt * rhsRhoV_[idx]) / c4;
                        if (config.dim >= 3)
                            state.rhoW[idx] = (c1 * state.rhoW[idx] + c2 * state.rhoW0[idx] + c3 * dt * rhsRhoW_[idx]) / c4;
                        state.rhoE[idx] = (c1 * state.rhoE[idx] + c2 * state.rhoE0[idx] + c3 * dt * rhsRhoE_[idx]) / c4;

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state.alphaRho[ph][idx] = (c1 * state.alphaRho[ph][idx] + c2 * state.alphaRho0[ph][idx] + c3 * dt * rhsAlphaRho_[ph][idx]) / c4;
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                state.alpha[ph][idx] = (c1 * state.alpha[ph][idx] + c2 * state.alpha0[ph][idx] + c3 * dt * rhsAlpha_[ph][idx]) / c4;
                        }
                    }

                    // Multi-phase post-update: recompute rho, clamp alpha and alphaRho
                    if (multiPhase) {
                        double rhoSum = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            state.alphaRho[ph][idx] = std::max(state.alphaRho[ph][idx], 1e-14);
                            rhoSum += state.alphaRho[ph][idx];
                        }
                        state.rho[idx] = rhoSum;

                        double alphaSum = 0.0;
                        for (int ph = 0; ph < nPhases - 1; ++ph) {
                            state.alpha[ph][idx] = std::clamp(state.alpha[ph][idx], alphaMin, 1.0 - alphaMin);
                            alphaSum += state.alpha[ph][idx];
                        }
                        if (alphaSum > 1.0 - alphaMin) {
                            double scale = (1.0 - alphaMin) / alphaSum;
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                state.alpha[ph][idx] *= scale;
                        }
                    }
                }
            }
        }

    }

    // Finalize: recompute primitives from the updated conservative state
    // and synchronize ghost cells so the state is consistent for output
    // and the next time step.
    if (multiPhase)
        MixtureEOS::convertConservativeToPrimitive(mesh, state, config.multiPhaseParams);
    else
        state.convertConservativeToPrimitiveVariables(mesh, eos_);
    mesh.applyBoundaryConditions(state, VarSet::PRIM, *halo_);
    if (ibm_) ibm_->applyGhostCells(mesh, state, config.dim);

    return dt;
}

void ExplicitSolver::solveIGR(const SimulationConfig& config,
        const RectilinearMesh& mesh,
        SolutionState& state) {
    if (!igrSolver_) return;

    computeVelocityGradients(config, mesh, state);

    igrSolver_->solveEntropicPressure(config, mesh, state, gradU_, *halo_);
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
    const bool multiPhase = config.isMultiPhase();
    const int nPhases = multiPhase ? config.multiPhaseParams.nPhases : 0;

    // Zero RHS arrays
    std::fill(rhsRho_.begin(),  rhsRho_.end(),  0.0);
    std::fill(rhsRhoU_.begin(), rhsRhoU_.end(), 0.0);
    if (dim >= 2) std::fill(rhsRhoV_.begin(), rhsRhoV_.end(), 0.0);
    if (dim >= 3) std::fill(rhsRhoW_.begin(), rhsRhoW_.end(), 0.0);
    std::fill(rhsRhoE_.begin(), rhsRhoE_.end(), 0.0);

    if (multiPhase) {
        for (int ph = 0; ph < nPhases; ++ph)
            std::fill(rhsAlphaRho_[ph].begin(), rhsAlphaRho_[ph].end(), 0.0);
        for (int ph = 0; ph < nPhases - 1; ++ph)
            std::fill(rhsAlpha_[ph].begin(), rhsAlpha_[ph].end(), 0.0);
        std::fill(divU_.begin(), divU_.end(), 0.0);
    }

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

                // Upwind partial density fluxes for multi-phase
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

                    if (multiPhase) {
                        double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                            rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases - 1; ++ph)
                            rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                        divU_[idxL] += coeff * flux.faceVelocity;
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

                    if (multiPhase) {
                        double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                            rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases - 1; ++ph)
                            rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                        divU_[idxR] -= coeff * flux.faceVelocity;
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

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 1.0, 0.0});

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

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                            divU_[idxL] += coeff * flux.faceVelocity;
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

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                            divU_[idxR] -= coeff * flux.faceVelocity;
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

                    RiemannFlux flux = riemannSolver_->computeFlux(
                        left, right, {0.0, 0.0, 1.0});

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

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                rhsAlpha_[ph][idxL] -= coeff * flux.alphaFlux[ph];
                            divU_[idxL] += coeff * flux.faceVelocity;
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

                        if (multiPhase) {
                            double rhoUpw = std::max(state.rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state.alphaRho[ph][upwindIdx] / rhoUpw) * flux.massFlux;
                                rhsAlphaRho_[ph][idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases - 1; ++ph)
                                rhsAlpha_[ph][idxR] += coeff * flux.alphaFlux[ph];
                            divU_[idxR] -= coeff * flux.faceVelocity;
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
                    for (int ph = 0; ph < nPhases - 1; ++ph)
                        rhsAlpha_[ph][idx] += state.alpha[ph][idx] * divU_[idx];
                }
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

} // namespace SemiImplicitFV
