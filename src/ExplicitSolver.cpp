#include "ExplicitSolver.hpp"
#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "ViscousFlux.hpp"
#include "SurfaceTension.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "HaloExchange.hpp"
#include "IGR.hpp"
#include "NvtxRange.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>

void explicit_solver_init(ExplicitSolverWork* w,
                          const RectilinearMesh* mesh,
                          const EOSData* eos,
                          RiemannSolverType solverType,
                          IGRSolver* igrSolver,
                          const SimulationConfig* config)
{
    std::memset(w, 0, sizeof(ExplicitSolverWork));
    w->eos = *eos;
    w->params = config->explicitParams;
    w->solverType = solverType;
    w->igrSolver = igrSolver;
    w->halo = NULL;

    reconstructor_init(&w->reconstructor, config->reconOrder, config->wenoEps,
                       eos->gamma, eos->pInf);

    w->fluxConfig.dim = config->dim;
    w->fluxConfig.includePressure = !config->semiImplicit;
    w->fluxConfig.useIGR = config->useIGR;
    w->fluxConfig.nPhases = config_is_multi_phase(config) ? config->multiPhaseParams.nPhases : 0;

    size_t n = mesh_total_cells(mesh);
    int dim = mesh->dim;
    w->totalCells = n;
    w->dim = dim;
    w->nPhases = config_is_multi_phase(config) ? config->multiPhaseParams.nPhases : 0;

    w->rhsRho  = (double*)std::calloc(n, sizeof(double));
    w->rhsRhoU = (double*)std::calloc(n, sizeof(double));
    if (dim >= 2) w->rhsRhoV = (double*)std::calloc(n, sizeof(double));
    if (dim >= 3) w->rhsRhoW = (double*)std::calloc(n, sizeof(double));
    w->rhsRhoE = (double*)std::calloc(n, sizeof(double));

    reconstructor_allocate(&w->reconstructor, mesh);

    if (w->nPhases > 0) {
        w->rhsAlphaRho = (double*)std::calloc((size_t)w->nPhases * n, sizeof(double));
        w->rhsAlpha    = (double*)std::calloc((size_t)w->nPhases * n, sizeof(double));
        w->divU        = (double*)std::calloc(n, sizeof(double));
    }

    if (igrSolver) {
        w->gradU = (GradientTensor*)std::calloc(n, sizeof(GradientTensor));
    }
}

void explicit_solver_free(ExplicitSolverWork* w)
{
    std::free(w->rhsRho);  w->rhsRho = NULL;
    std::free(w->rhsRhoU); w->rhsRhoU = NULL;
    std::free(w->rhsRhoV); w->rhsRhoV = NULL;
    std::free(w->rhsRhoW); w->rhsRhoW = NULL;
    std::free(w->rhsRhoE); w->rhsRhoE = NULL;
    std::free(w->rhsAlphaRho); w->rhsAlphaRho = NULL;
    std::free(w->rhsAlpha);    w->rhsAlpha = NULL;
    std::free(w->divU);        w->divU = NULL;
    std::free(w->gradU);       w->gradU = NULL;
    reconstructor_free(&w->reconstructor);
}

/* Internal: compute velocity gradients for IGR */
static void explicit_compute_velocity_gradients(ExplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, const SolutionState* state)
{
    int dim = mesh->dim;
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                size_t xm = mesh_index(mesh, i - 1, j, k);
                size_t xp = mesh_index(mesh, i + 1, j, k);
                double u_xm[3] = {state->velU[xm],
                                   (dim >= 2) ? state->velV[xm] : 0.0,
                                   (dim >= 3) ? state->velW[xm] : 0.0};
                double u_xp[3] = {state->velU[xp],
                                   (dim >= 2) ? state->velV[xp] : 0.0,
                                   (dim >= 3) ? state->velW[xp] : 0.0};

                double u_ym[3], u_yp[3];
                double dyj;
                if (dim >= 2) {
                    size_t ym = mesh_index(mesh, i, j - 1, k);
                    size_t yp = mesh_index(mesh, i, j + 1, k);
                    u_ym[0] = state->velU[ym]; u_ym[1] = state->velV[ym];
                    u_ym[2] = (dim >= 3) ? state->velW[ym] : 0.0;
                    u_yp[0] = state->velU[yp]; u_yp[1] = state->velV[yp];
                    u_yp[2] = (dim >= 3) ? state->velW[yp] : 0.0;
                    dyj = mesh_dy(mesh, j);
                } else {
                    u_ym[0] = state->velU[idx]; u_ym[1] = 0.0; u_ym[2] = 0.0;
                    u_yp[0] = state->velU[idx]; u_yp[1] = 0.0; u_yp[2] = 0.0;
                    dyj = 1.0;
                }

                double u_zm[3], u_zp[3];
                double dzk;
                if (dim >= 3) {
                    size_t zm = mesh_index(mesh, i, j, k - 1);
                    size_t zp = mesh_index(mesh, i, j, k + 1);
                    u_zm[0] = state->velU[zm]; u_zm[1] = state->velV[zm]; u_zm[2] = state->velW[zm];
                    u_zp[0] = state->velU[zp]; u_zp[1] = state->velV[zp]; u_zp[2] = state->velW[zp];
                    dzk = mesh_dz(mesh, k);
                } else {
                    u_zm[0] = state->velU[idx];
                    u_zm[1] = (dim >= 2) ? state->velV[idx] : 0.0;
                    u_zm[2] = 0.0;
                    u_zp[0] = u_zm[0]; u_zp[1] = u_zm[1]; u_zp[2] = 0.0;
                    dzk = 1.0;
                }

                /* Compute gradient directly into the flat GradientTensor */
                double dxi = mesh_dx(mesh, i);
                for (int c = 0; c < 3; ++c) {
                    w->gradU[idx][0][c] = (u_xp[c] - u_xm[c]) / (2.0 * dxi);
                    w->gradU[idx][1][c] = (dim >= 2) ? (u_yp[c] - u_ym[c]) / (2.0 * dyj) : 0.0;
                    w->gradU[idx][2][c] = (dim >= 3) ? (u_zp[c] - u_zm[c]) / (2.0 * dzk) : 0.0;
                }
            }
        }
    }
}

/* Internal: solve IGR */
static void explicit_solve_igr(ExplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, SolutionState* state)
{
    if (!w->igrSolver) return;
    NVTX_PUSH("Explicit::solveIGR");
    explicit_compute_velocity_gradients(w, config, mesh, state);
    igr_solve_entropic_pressure_mpi(config, &config->igrParams, mesh, state,
                                    w->gradU, w->halo);
    NVTX_POP();
}

/* Internal: compute RHS */
static void explicit_compute_rhs(ExplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, SolutionState* state)
{
    NVTX_PUSH("Explicit::computeRHS");
    reconstruct(&w->reconstructor, config, mesh, state);

    int dim = mesh->dim;
    int multiPhase = config_is_multi_phase(config);
    int nPhases = multiPhase ? config->multiPhaseParams.nPhases : 0;
    size_t n = w->totalCells;
    size_t tc = state->totalCells;

    /* Zero RHS */
    std::memset(w->rhsRho,  0, n * sizeof(double));
    std::memset(w->rhsRhoU, 0, n * sizeof(double));
    if (dim >= 2) std::memset(w->rhsRhoV, 0, n * sizeof(double));
    if (dim >= 3) std::memset(w->rhsRhoW, 0, n * sizeof(double));
    std::memset(w->rhsRhoE, 0, n * sizeof(double));

    if (multiPhase) {
        std::memset(w->rhsAlphaRho, 0, (size_t)nPhases * n * sizeof(double));
        std::memset(w->rhsAlpha,    0, (size_t)nPhases * n * sizeof(double));
        std::memset(w->divU, 0, n * sizeof(double));
    }

    ReconstructorData* rec = &w->reconstructor;
    FluxConfig fc = w->fluxConfig;

    /* --- X-direction fluxes --- */
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i <= mesh->nx; ++i) {
                size_t f = x_face_index(rec, i, j, k);
                const PrimitiveState* left  = x_face_left(rec, f);
                const PrimitiveState* right = x_face_right(rec, f);

                double normal[3] = {1.0, 0.0, 0.0};
                RiemannFlux flux = computeFluxDirect(w->solverType, left, right, normal, &fc);

                double area = mesh_faceAreaX(mesh, j, k);

                size_t upwindIdx = 0;
                if (multiPhase)
                    upwindIdx = (flux.massFlux >= 0) ? mesh_index(mesh, i - 1, j, k) : mesh_index(mesh, i, j, k);

                if (i >= 1) {
                    size_t idxL = mesh_index(mesh, i - 1, j, k);
                    double coeff = area / mesh_cell_volume(mesh, i - 1, j, k);
                    w->rhsRho[idxL]  -= coeff * flux.massFlux;
                    w->rhsRhoU[idxL] -= coeff * flux.momentumFlux[0];
                    if (dim >= 2) w->rhsRhoV[idxL] -= coeff * flux.momentumFlux[1];
                    if (dim >= 3) w->rhsRhoW[idxL] -= coeff * flux.momentumFlux[2];
                    w->rhsRhoE[idxL] -= coeff * flux.energyFlux;

                    if (multiPhase) {
                        double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                            w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
                        w->divU[idxL] += coeff * flux.faceVelocity;
                    }
                }

                if (i < mesh->nx) {
                    size_t idxR = mesh_index(mesh, i, j, k);
                    double coeff = area / mesh_cell_volume(mesh, i, j, k);
                    w->rhsRho[idxR]  += coeff * flux.massFlux;
                    w->rhsRhoU[idxR] += coeff * flux.momentumFlux[0];
                    if (dim >= 2) w->rhsRhoV[idxR] += coeff * flux.momentumFlux[1];
                    if (dim >= 3) w->rhsRhoW[idxR] += coeff * flux.momentumFlux[2];
                    w->rhsRhoE[idxR] += coeff * flux.energyFlux;

                    if (multiPhase) {
                        double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                            w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
                        w->divU[idxR] -= coeff * flux.faceVelocity;
                    }
                }
            }
        }
    }

    /* --- Y-direction fluxes --- */
    if (dim >= 2) {
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j <= mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t f = y_face_index(rec, i, j, k);
                    const PrimitiveState* left  = y_face_left(rec, f);
                    const PrimitiveState* right = y_face_right(rec, f);

                    double normal[3] = {0.0, 1.0, 0.0};
                    RiemannFlux flux = computeFluxDirect(w->solverType, left, right, normal, &fc);

                    double area = mesh_faceAreaY(mesh, i, k);

                    size_t upwindIdx = 0;
                    if (multiPhase)
                        upwindIdx = (flux.massFlux >= 0) ? mesh_index(mesh, i, j - 1, k) : mesh_index(mesh, i, j, k);

                    if (j >= 1) {
                        size_t idxL = mesh_index(mesh, i, j - 1, k);
                        double coeff = area / mesh_cell_volume(mesh, i, j - 1, k);
                        w->rhsRho[idxL]  -= coeff * flux.massFlux;
                        w->rhsRhoU[idxL] -= coeff * flux.momentumFlux[0];
                        w->rhsRhoV[idxL] -= coeff * flux.momentumFlux[1];
                        if (dim >= 3) w->rhsRhoW[idxL] -= coeff * flux.momentumFlux[2];
                        w->rhsRhoE[idxL] -= coeff * flux.energyFlux;

                        if (multiPhase) {
                            double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                                w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
                            w->divU[idxL] += coeff * flux.faceVelocity;
                        }
                    }

                    if (j < mesh->ny) {
                        size_t idxR = mesh_index(mesh, i, j, k);
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
                        w->rhsRho[idxR]  += coeff * flux.massFlux;
                        w->rhsRhoU[idxR] += coeff * flux.momentumFlux[0];
                        w->rhsRhoV[idxR] += coeff * flux.momentumFlux[1];
                        if (dim >= 3) w->rhsRhoW[idxR] += coeff * flux.momentumFlux[2];
                        w->rhsRhoE[idxR] += coeff * flux.energyFlux;

                        if (multiPhase) {
                            double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                                w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
                            w->divU[idxR] -= coeff * flux.faceVelocity;
                        }
                    }
                }
            }
        }
    }

    /* --- Z-direction fluxes --- */
    if (dim >= 3) {
        for (int k = 0; k <= mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t f = z_face_index(rec, i, j, k);
                    const PrimitiveState* left  = z_face_left(rec, f);
                    const PrimitiveState* right = z_face_right(rec, f);

                    double normal[3] = {0.0, 0.0, 1.0};
                    RiemannFlux flux = computeFluxDirect(w->solverType, left, right, normal, &fc);

                    double area = mesh_faceAreaZ(mesh, i, j);

                    size_t upwindIdx = 0;
                    if (multiPhase)
                        upwindIdx = (flux.massFlux >= 0) ? mesh_index(mesh, i, j, k - 1) : mesh_index(mesh, i, j, k);

                    if (k >= 1) {
                        size_t idxL = mesh_index(mesh, i, j, k - 1);
                        double coeff = area / mesh_cell_volume(mesh, i, j, k - 1);
                        w->rhsRho[idxL]  -= coeff * flux.massFlux;
                        w->rhsRhoU[idxL] -= coeff * flux.momentumFlux[0];
                        w->rhsRhoV[idxL] -= coeff * flux.momentumFlux[1];
                        w->rhsRhoW[idxL] -= coeff * flux.momentumFlux[2];
                        w->rhsRhoE[idxL] -= coeff * flux.energyFlux;

                        if (multiPhase) {
                            double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                                w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
                            w->divU[idxL] += coeff * flux.faceVelocity;
                        }
                    }

                    if (k < mesh->nz) {
                        size_t idxR = mesh_index(mesh, i, j, k);
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
                        w->rhsRho[idxR]  += coeff * flux.massFlux;
                        w->rhsRhoU[idxR] += coeff * flux.momentumFlux[0];
                        w->rhsRhoV[idxR] += coeff * flux.momentumFlux[1];
                        w->rhsRhoW[idxR] += coeff * flux.momentumFlux[2];
                        w->rhsRhoE[idxR] += coeff * flux.energyFlux;

                        if (multiPhase) {
                            double rhoUpw = std::max(state->rho[upwindIdx], 1e-14);
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / rhoUpw) * flux.massFlux;
                                w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
                            w->divU[idxR] -= coeff * flux.faceVelocity;
                        }
                    }
                }
            }
        }
    }

    /* --- Alpha source term --- */
    if (multiPhase) {
        for (int k = 0; k < mesh->nz; ++k)
            for (int j = 0; j < mesh->ny; ++j)
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    for (int ph = 0; ph < nPhases; ++ph)
                        w->rhsAlpha[ph * n + idx] += state->alpha[ph * tc + idx] * w->divU[idx];
                }
    }

    /* --- Body force --- */
    if (config_has_body_force(config)) {
        const BodyForceParams* bf = &config->bodyForceParams;
        double accel_x = bf->a[0] + bf->b[0] * std::cos(bf->c[0] * config->time + bf->d[0]);
        double accel_y = bf->a[1] + bf->b[1] * std::cos(bf->c[1] * config->time + bf->d[1]);
        double accel_z = bf->a[2] + bf->b[2] * std::cos(bf->c[2] * config->time + bf->d[2]);

        for (int k = 0; k < mesh->nz; ++k)
            for (int j = 0; j < mesh->ny; ++j)
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    w->rhsRhoU[idx] += state->rho[idx] * accel_x;
                    if (dim >= 2) w->rhsRhoV[idx] += state->rho[idx] * accel_y;
                    if (dim >= 3) w->rhsRhoW[idx] += state->rho[idx] * accel_z;
                    double work = state->velU[idx] * accel_x;
                    if (dim >= 2) work += state->velV[idx] * accel_y;
                    if (dim >= 3) work += state->velW[idx] * accel_z;
                    w->rhsRhoE[idx] += state->rho[idx] * work;
                }
    }

    /* --- Viscous stress --- */
    if (config_has_viscosity(config)) {
        add_viscous_fluxes(config, mesh, state, w->rhsRhoU, w->rhsRhoV, w->rhsRhoW, w->rhsRhoE);
    }

    /* --- Surface tension --- */
    if (config_has_surface_tension(config)) {
        add_surface_tension_fluxes(config, mesh, state,
            config->surfaceTensionParams.sigma,
            w->rhsRhoU, w->rhsRhoV, w->rhsRhoW, w->rhsRhoE);
    }

    NVTX_POP();
}

double explicit_step(ExplicitSolverWork* w,
                     const SimulationConfig* config,
                     const RectilinearMesh* mesh,
                     SolutionState* state,
                     double targetDt)
{
    NVTX_PUSH("Explicit::step");

    double dt;
    if (w->params.constDt > 0) {
        dt = w->params.constDt;
    } else {
        dt = computeAcousticTimeStep_config_mpi(
            mesh, state, &w->eos, config, w->params.cfl, w->params.maxDt,
            w->halo->mpi->cartComm);
        if (config_has_viscosity(config)) {
            dt = std::min(dt, computeViscousDt_config_mpi(mesh, state,
                config, w->params.cfl, w->params.maxDt, w->halo->mpi->cartComm));
        }
        if (config_has_surface_tension(config)) {
            dt = std::min(dt, computeCapillaryDt_mpi(mesh, state,
                config->surfaceTensionParams.sigma, w->params.cfl, w->params.maxDt,
                w->halo->mpi->cartComm));
        }
    }

    if (targetDt > 0) dt = std::min(dt, targetDt);

    /* TVD RK coefficients */
    double rk_coef[3][4];
    if (config->RKOrder == 1) {
        rk_coef[0][0] = 1.0; rk_coef[0][1] = 0.0; rk_coef[0][2] = 1.0; rk_coef[0][3] = 1.0;
    } else if (config->RKOrder == 2) {
        rk_coef[0][0] = 1.0; rk_coef[0][1] = 0.0; rk_coef[0][2] = 1.0; rk_coef[0][3] = 1.0;
        rk_coef[1][0] = 1.0; rk_coef[1][1] = 1.0; rk_coef[1][2] = 1.0; rk_coef[1][3] = 2.0;
    } else {
        rk_coef[0][0] = 1.0; rk_coef[0][1] = 0.0; rk_coef[0][2] = 1.0; rk_coef[0][3] = 1.0;
        rk_coef[1][0] = 1.0; rk_coef[1][1] = 3.0; rk_coef[1][2] = 1.0; rk_coef[1][3] = 4.0;
        rk_coef[2][0] = 2.0; rk_coef[2][1] = 1.0; rk_coef[2][2] = 2.0; rk_coef[2][3] = 3.0;
    }

    int multiPhase = config_is_multi_phase(config);
    int nPhases = multiPhase ? config->multiPhaseParams.nPhases : 0;
    double alphaMin = multiPhase ? config->multiPhaseParams.alphaMin : 0.0;
    size_t n = w->totalCells;
    size_t tc = state->totalCells;

    for (int s = 0; s < config->RKOrder; ++s) {
        if (multiPhase)
            mixture_cons_to_prim(mesh, state, &config->multiPhaseParams);
        else
            state_cons_to_prim(state, mesh, &w->eos);
        mesh_apply_bcs_mpi(mesh, state, VARSET_PRIM, w->halo);

        if (config->useIGR && w->igrSolver) explicit_solve_igr(w, config, mesh, state);

        explicit_compute_rhs(w, config, mesh, state);

        double c1 = rk_coef[s][0], c2 = rk_coef[s][1];
        double c3 = rk_coef[s][2], c4 = rk_coef[s][3];

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);

                    if (s == 0 && config->RKOrder > 1) {
                        state_save_conservative_cell(state, idx);
                    }

                    if (config->RKOrder == 1) {
                        state->rho[idx]  += dt * w->rhsRho[idx];
                        state->rhoU[idx] += dt * w->rhsRhoU[idx];
                        if (config->dim >= 2) state->rhoV[idx] += dt * w->rhsRhoV[idx];
                        if (config->dim >= 3) state->rhoW[idx] += dt * w->rhsRhoW[idx];
                        state->rhoE[idx] += dt * w->rhsRhoE[idx];
                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alphaRho[ph * tc + idx] += dt * w->rhsAlphaRho[ph * n + idx];
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alpha[ph * tc + idx] += dt * w->rhsAlpha[ph * n + idx];
                        }
                    } else {
                        state->rho[idx]  = (c1 * state->rho[idx]  + c2 * state->rho0[idx]  + c3 * dt * w->rhsRho[idx])  / c4;
                        state->rhoU[idx] = (c1 * state->rhoU[idx] + c2 * state->rhoU0[idx] + c3 * dt * w->rhsRhoU[idx]) / c4;
                        if (config->dim >= 2)
                            state->rhoV[idx] = (c1 * state->rhoV[idx] + c2 * state->rhoV0[idx] + c3 * dt * w->rhsRhoV[idx]) / c4;
                        if (config->dim >= 3)
                            state->rhoW[idx] = (c1 * state->rhoW[idx] + c2 * state->rhoW0[idx] + c3 * dt * w->rhsRhoW[idx]) / c4;
                        state->rhoE[idx] = (c1 * state->rhoE[idx] + c2 * state->rhoE0[idx] + c3 * dt * w->rhsRhoE[idx]) / c4;
                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alphaRho[ph * tc + idx] = (c1 * state->alphaRho[ph * tc + idx] + c2 * state->alphaRho0[ph * tc + idx] + c3 * dt * w->rhsAlphaRho[ph * n + idx]) / c4;
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alpha[ph * tc + idx] = (c1 * state->alpha[ph * tc + idx] + c2 * state->alpha0[ph * tc + idx] + c3 * dt * w->rhsAlpha[ph * n + idx]) / c4;
                        }
                    }

                    if (multiPhase) {
                        double rhoSum = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph) {
                            state->alphaRho[ph * tc + idx] = std::max(state->alphaRho[ph * tc + idx], 1e-14);
                            rhoSum += state->alphaRho[ph * tc + idx];
                        }
                        state->rho[idx] = rhoSum;
                        for (int ph = 0; ph < nPhases; ++ph)
                            state->alpha[ph * tc + idx] = std::max(state->alpha[ph * tc + idx], alphaMin);
                        double alphaSum = 0.0;
                        for (int ph = 0; ph < nPhases; ++ph)
                            alphaSum += state->alpha[ph * tc + idx];
                        for (int ph = 0; ph < nPhases; ++ph)
                            state->alpha[ph * tc + idx] /= alphaSum;
                    }
                }
            }
        }
    }

    /* Finalize */
    if (multiPhase)
        mixture_cons_to_prim(mesh, state, &config->multiPhaseParams);
    else
        state_cons_to_prim(state, mesh, &w->eos);
    mesh_apply_bcs_mpi(mesh, state, VARSET_PRIM, w->halo);

    NVTX_POP();
    return dt;
}
