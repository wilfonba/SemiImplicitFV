#include "SemiImplicitSolver.hpp"
#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "ViscousFlux.hpp"
#include "SurfaceTension.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "HaloExchange.hpp"
#include "PressureSolver.hpp"
#include "IGR.hpp"
#include "NvtxRange.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <vector>

void semi_implicit_solver_init(SemiImplicitSolverWork* w,
                               const RectilinearMesh* mesh,
                               const EOSData* eos,
                               RiemannSolverType solverType,
                               PressureSolverData* pressureSolver,
                               IGRSolver* igrSolver,
                               const SimulationConfig* config)
{
    std::memset(w, 0, sizeof(SemiImplicitSolverWork));
    w->eos = *eos;
    w->params = config->semiImplicitParams;
    w->solverType = solverType;
    w->pressureSolver = pressureSolver;
    w->igrSolver = igrSolver;
    w->halo = NULL;
    w->lastPressureIters = 0;

    w->gamma = eos->gamma;
    w->pInf = eos->pInf;

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

    w->pressureRhs = (double*)std::calloc(n, sizeof(double));
    w->pressure    = (double*)std::calloc(n, sizeof(double));
    w->divUstar    = (double*)std::calloc(n, sizeof(double));

    w->rhsRho  = (double*)std::calloc(n, sizeof(double));
    w->rhsRhoU = (double*)std::calloc(n, sizeof(double));
    if (dim >= 2) w->rhsRhoV = (double*)std::calloc(n, sizeof(double));
    if (dim >= 3) w->rhsRhoW = (double*)std::calloc(n, sizeof(double));
    w->rhsRhoE      = (double*)std::calloc(n, sizeof(double));
    w->rhsPadvected  = (double*)std::calloc(n, sizeof(double));

    reconstructor_allocate(&w->reconstructor, mesh);

    w->divU = (double*)std::calloc(n, sizeof(double));

    if (w->nPhases > 0) {
        w->rhsAlphaRho = (double*)std::calloc((size_t)w->nPhases * n, sizeof(double));
        w->rhsAlpha    = (double*)std::calloc((size_t)w->nPhases * n, sizeof(double));
    }

    if (igrSolver) {
        w->gradU = (GradientTensor*)std::calloc(n, sizeof(GradientTensor));
    }
}

void semi_implicit_solver_free(SemiImplicitSolverWork* w)
{
    std::free(w->pressureRhs); w->pressureRhs = NULL;
    std::free(w->pressure);    w->pressure = NULL;
    std::free(w->divUstar);    w->divUstar = NULL;
    std::free(w->rhsRho);     w->rhsRho = NULL;
    std::free(w->rhsRhoU);    w->rhsRhoU = NULL;
    std::free(w->rhsRhoV);    w->rhsRhoV = NULL;
    std::free(w->rhsRhoW);    w->rhsRhoW = NULL;
    std::free(w->rhsRhoE);    w->rhsRhoE = NULL;
    std::free(w->rhsPadvected); w->rhsPadvected = NULL;
    std::free(w->rhsAlphaRho); w->rhsAlphaRho = NULL;
    std::free(w->rhsAlpha);    w->rhsAlpha = NULL;
    std::free(w->divU);        w->divU = NULL;
    std::free(w->gradU);       w->gradU = NULL;
    reconstructor_free(&w->reconstructor);
}

/* Internal: compute velocity gradients for IGR */
static void semi_implicit_compute_velocity_gradients(SemiImplicitSolverWork* w,
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
static void semi_implicit_solve_igr(SemiImplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, SolutionState* state)
{
    if (!w->igrSolver) return;
    NVTX_PUSH("SemiImplicit::solveIGR");
    semi_implicit_compute_velocity_gradients(w, config, mesh, state);
    igr_solve_entropic_pressure_mpi(config, &config->igrParams, mesh, state,
                                    w->gradU, w->halo);
    NVTX_POP();
}

/* Internal: compute divergence of velocity field */
static void semi_implicit_compute_divergence(const RectilinearMesh* mesh,
    const SolutionState* state, double* divU)
{
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                double div = 0.0;

                {
                    size_t xm = mesh_index(mesh, i - 1, j, k);
                    size_t xp = mesh_index(mesh, i + 1, j, k);
                    double uFaceL = 0.5 * (state->velU[xm] + state->velU[idx]);
                    double uFaceR = 0.5 * (state->velU[idx] + state->velU[xp]);
                    div += (uFaceR - uFaceL) / mesh_dx(mesh, i);
                }

                if (mesh->dim >= 2) {
                    size_t ym = mesh_index(mesh, i, j - 1, k);
                    size_t yp = mesh_index(mesh, i, j + 1, k);
                    double vFaceL = 0.5 * (state->velV[ym] + state->velV[idx]);
                    double vFaceR = 0.5 * (state->velV[idx] + state->velV[yp]);
                    div += (vFaceR - vFaceL) / mesh_dy(mesh, j);
                }

                if (mesh->dim >= 3) {
                    size_t zm = mesh_index(mesh, i, j, k - 1);
                    size_t zp = mesh_index(mesh, i, j, k + 1);
                    double wFaceL = 0.5 * (state->velW[zm] + state->velW[idx]);
                    double wFaceR = 0.5 * (state->velW[idx] + state->velW[zp]);
                    div += (wFaceR - wFaceL) / mesh_dz(mesh, k);
                }

                divU[idx] = div;
            }
        }
    }
}

/* Internal: compute RHS */
static void semi_implicit_compute_rhs(SemiImplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, SolutionState* state)
{
    NVTX_PUSH("SemiImplicit::computeRHS");
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
    std::memset(w->rhsPadvected, 0, n * sizeof(double));
    std::memset(w->divU, 0, n * sizeof(double));

    if (multiPhase) {
        std::memset(w->rhsAlphaRho, 0, (size_t)nPhases * n * sizeof(double));
        std::memset(w->rhsAlpha,    0, (size_t)nPhases * n * sizeof(double));
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
                    w->rhsPadvected[idxL] -= coeff * flux.pressureFlux;
                    w->divU[idxL] += coeff * flux.faceVelocity;

                    if (multiPhase) {
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                            double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                            w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
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
                    w->rhsPadvected[idxR] += coeff * flux.pressureFlux;
                    w->divU[idxR] -= coeff * flux.faceVelocity;

                    if (multiPhase) {
                        for (int ph = 0; ph < nPhases; ++ph) {
                            double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                            double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                            w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                        }
                        for (int ph = 0; ph < nPhases; ++ph)
                            w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
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
                        w->rhsPadvected[idxL] -= coeff * flux.pressureFlux;
                        w->divU[idxL] += coeff * flux.faceVelocity;

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                                w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
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
                        w->rhsPadvected[idxR] += coeff * flux.pressureFlux;
                        w->divU[idxR] -= coeff * flux.faceVelocity;

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                                w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
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
                        w->rhsPadvected[idxL] -= coeff * flux.pressureFlux;
                        w->divU[idxL] += coeff * flux.faceVelocity;

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                                w->rhsAlphaRho[ph * n + idxL] -= coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxL] -= coeff * flux.alphaFlux[ph];
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
                        w->rhsPadvected[idxR] += coeff * flux.pressureFlux;
                        w->divU[idxR] -= coeff * flux.faceVelocity;

                        if (multiPhase) {
                            for (int ph = 0; ph < nPhases; ++ph) {
                                double aUpw = std::max(state->alpha[ph * tc + upwindIdx], 1e-14);
                                double alphaRhoFlux = (state->alphaRho[ph * tc + upwindIdx] / aUpw) * flux.alphaFlux[ph];
                                w->rhsAlphaRho[ph * n + idxR] += coeff * alphaRhoFlux;
                            }
                            for (int ph = 0; ph < nPhases; ++ph)
                                w->rhsAlpha[ph * n + idxR] += coeff * flux.alphaFlux[ph];
                        }
                    }
                }
            }
        }
    }

    /* --- Alpha source term: alpha * div(u) --- */
    if (multiPhase) {
        for (int k = 0; k < mesh->nz; ++k)
            for (int j = 0; j < mesh->ny; ++j)
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    for (int ph = 0; ph < nPhases; ++ph)
                        w->rhsAlpha[ph * n + idx] += state->alpha[ph * tc + idx] * w->divU[idx];
                }
    }

    /* --- Pressure advection source term: dp/dt = -div(pu) + p*div(u) --- */
    for (int k = 0; k < mesh->nz; ++k)
        for (int j = 0; j < mesh->ny; ++j)
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                w->rhsPadvected[idx] += state->pres[idx] * w->divU[idx];
            }

    /* --- Body force source terms --- */
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

    /* --- Surface tension (capillary stress) --- */
    if (config_has_surface_tension(config)) {
        add_surface_tension_fluxes(config, mesh, state,
            config->surfaceTensionParams.sigma,
            w->rhsRhoU, w->rhsRhoV, w->rhsRhoW, w->rhsRhoE);
    }

    NVTX_POP();
}

/* Internal: solve pressure Poisson equation */
static void semi_implicit_solve_pressure(SemiImplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh,
    SolutionState* state, double dt)
{
    NVTX_PUSH("SemiImplicit::solvePressure");
    int multiPhase = config_is_multi_phase(config);
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    int nPhases = multiPhase ? mp->nPhases : 0;
    size_t tc = state->totalCells;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                if (multiPhase) {
                    for (int ph = 0; ph < nPhases; ++ph)
                        w->scratchAlphas[ph] = state->alpha[ph * tc + idx];
                    for (int ph = 0; ph < nPhases; ++ph)
                        w->scratchAlphaRhos[ph] = state->alphaRho[ph * tc + idx];
                    double c = mixture_sound_speed(
                        state->rho[idx], state->pres[idx],
                        w->scratchAlphas, w->scratchAlphaRhos,
                        nPhases, mp->phases);
                    state->rhoc2[idx] = state->rho[idx] * c * c;
                } else {
                    double c = std::sqrt(w->gamma * std::max(state->pres[idx] + w->pInf, 1e-14)
                                         / std::max(state->rho[idx], 1e-14));
                    state->rhoc2[idx] = state->rho[idx] * c * c;
                }
            }
        }
    }

    semi_implicit_compute_divergence(mesh, state, w->divUstar);

    for (int k = 0; k < mesh->nz; ++k)
        for (int j = 0; j < mesh->ny; ++j)
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                w->pressureRhs[idx] = state->pAdvected[idx] - state->rhoc2[idx] * dt * w->divUstar[idx];
            }

    for (int k = 0; k < mesh->nz; ++k)
        for (int j = 0; j < mesh->ny; ++j)
            for (int i = 0; i < mesh->nx; ++i)
                w->pressure[mesh_index(mesh, i, j, k)] = state->pres[mesh_index(mesh, i, j, k)];

    w->lastPressureIters = pressure_solve_mpi(
        w->pressureSolver,
        mesh, state->rho, state->rhoc2, w->pressureRhs, w->pressure,
        w->totalCells, dt, w->params.pressureTol, w->params.maxPressureIters,
        w->halo);

    mesh_fill_scalar_ghosts_mpi(mesh, w->pressure, w->halo);
    mesh_fill_scalar_ghosts_mpi(mesh, state->sigma, w->halo);

    for (int k = 0; k < mesh->nz; ++k)
        for (int j = 0; j < mesh->ny; ++j)
            for (int i = 0; i < mesh->nx; ++i)
                state->pres[mesh_index(mesh, i, j, k)] = w->pressure[mesh_index(mesh, i, j, k)];

    NVTX_POP();
}

/* Internal: correction step - apply pressure gradient to momentum */
static void semi_implicit_correction_step(SemiImplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh,
    SolutionState* state, double dt)
{
    NVTX_PUSH("SemiImplicit::correctionStep");
    int dim = mesh->dim;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                {
                    size_t xm = mesh_index(mesh, i - 1, j, k);
                    size_t xp = mesh_index(mesh, i + 1, j, k);
                    double pTotL = 0.5 * ((w->pressure[xm] + state->sigma[xm]) +
                                           (w->pressure[idx] + state->sigma[idx]));
                    double pTotR = 0.5 * ((w->pressure[idx] + state->sigma[idx]) +
                                           (w->pressure[xp] + state->sigma[xp]));
                    state->rhoU[idx] = state->rhoUStar[idx] - dt * (pTotR - pTotL) / mesh_dx(mesh, i);
                }

                if (dim >= 2) {
                    size_t ym = mesh_index(mesh, i, j - 1, k);
                    size_t yp = mesh_index(mesh, i, j + 1, k);
                    double pTotL = 0.5 * ((w->pressure[ym] + state->sigma[ym]) +
                                           (w->pressure[idx] + state->sigma[idx]));
                    double pTotR = 0.5 * ((w->pressure[idx] + state->sigma[idx]) +
                                           (w->pressure[yp] + state->sigma[yp]));
                    state->rhoV[idx] = state->rhoVStar[idx] - dt * (pTotR - pTotL) / mesh_dy(mesh, j);
                }

                if (dim >= 3) {
                    size_t zm = mesh_index(mesh, i, j, k - 1);
                    size_t zp = mesh_index(mesh, i, j, k + 1);
                    double pTotL = 0.5 * ((w->pressure[zm] + state->sigma[zm]) +
                                           (w->pressure[idx] + state->sigma[idx]));
                    double pTotR = 0.5 * ((w->pressure[idx] + state->sigma[idx]) +
                                           (w->pressure[zp] + state->sigma[zp]));
                    state->rhoW[idx] = state->rhoWStar[idx] - dt * (pTotR - pTotL) / mesh_dz(mesh, k);
                }
            }
        }
    }

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                double rhoSafe = std::max(state->rho[idx], 1e-14);
                state->velU[idx] = state->rhoU[idx] / rhoSafe;
                if (dim >= 2) state->velV[idx] = state->rhoV[idx] / rhoSafe;
                if (dim >= 3) state->velW[idx] = state->rhoW[idx] / rhoSafe;
            }
        }
    }
    mesh_apply_bcs_mpi(mesh, state, VARSET_PRIM, w->halo);

    /* Reconstruct total energy from solved pressure and corrected velocity */
    int multiPhase = config_is_multi_phase(config);
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    int nPhases = multiPhase ? mp->nPhases : 0;
    size_t tc = state->totalCells;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                if (multiPhase) {
                    double rho = state->rho[idx];
                    double ke = 0.5 * rho * state->velU[idx] * state->velU[idx];
                    if (dim >= 2) ke += 0.5 * rho * state->velV[idx] * state->velV[idx];
                    if (dim >= 3) ke += 0.5 * rho * state->velW[idx] * state->velW[idx];

                    for (int ph = 0; ph < nPhases; ++ph)
                        w->scratchAlphas[ph] = state->alpha[ph * tc + idx];

                    state->rhoE[idx] = mixture_total_energy(
                        rho, w->pressure[idx], w->scratchAlphas,
                        nPhases, ke, mp->phases);
                } else {
                    double rho = state->rho[idx];
                    double ke = 0.5 * rho * state->velU[idx] * state->velU[idx];
                    if (dim >= 2) ke += 0.5 * rho * state->velV[idx] * state->velV[idx];
                    if (dim >= 3) ke += 0.5 * rho * state->velW[idx] * state->velW[idx];
                    state->rhoE[idx] = (w->pressure[idx] + w->gamma * w->pInf) / (w->gamma - 1.0) + ke;
                }
            }
        }
    }

    NVTX_POP();
}

double semi_implicit_step(SemiImplicitSolverWork* w,
                          const SimulationConfig* config,
                          const RectilinearMesh* mesh,
                          SolutionState* state,
                          double targetDt)
{
    NVTX_PUSH("SemiImplicit::step");

    double dt;
    if (w->params.constDt > 0) {
        dt = w->params.constDt;
    } else {
        dt = computeAdvectiveTimeStep_mpi(
            mesh, state, w->params.cfl, w->params.maxDt,
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
    if (w->params.maxAcousticCFL > 0) {
        double acousticDt = computeAcousticTimeStep_config_mpi(
            mesh, state, &w->eos, config, w->params.maxAcousticCFL, w->params.maxDt,
            w->halo->mpi->cartComm);
        dt = std::min(dt, acousticDt);
    }
    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }
    dt = std::clamp(dt, w->params.minDt, w->params.maxDt);

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

        if (config->useIGR && w->igrSolver) semi_implicit_solve_igr(w, config, mesh, state);

        semi_implicit_compute_rhs(w, config, mesh, state);

        double c1 = rk_coef[s][0], c2 = rk_coef[s][1];
        double c3 = rk_coef[s][2], c4 = rk_coef[s][3];

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);

                    if (s == 0 && config->RKOrder > 1) {
                        state_save_conservative_cell(state, idx);
                    }

                    state->pAdvected[idx] = state->pres[idx];

                    double rho0  = (config->RKOrder > 1) ? state->rho0[idx]  : 0.0;
                    double rhoU0 = (config->RKOrder > 1) ? state->rhoU0[idx] : 0.0;
                    double rhoE0 = (config->RKOrder > 1) ? state->rhoE0[idx] : 0.0;

                    state->rho[idx]      = (c1 * state->rho[idx]  + c2 * rho0  + c3 * dt * w->rhsRho[idx])   / c4;
                    state->rhoUStar[idx] = (c1 * state->rhoU[idx] + c2 * rhoU0 + c3 * dt * w->rhsRhoU[idx])  / c4;
                    if (config->dim >= 2) {
                        double rhoV0 = (config->RKOrder > 1) ? state->rhoV0[idx] : 0.0;
                        state->rhoVStar[idx] = (c1 * state->rhoV[idx] + c2 * rhoV0 + c3 * dt * w->rhsRhoV[idx]) / c4;
                    }
                    if (config->dim >= 3) {
                        double rhoW0 = (config->RKOrder > 1) ? state->rhoW0[idx] : 0.0;
                        state->rhoWStar[idx] = (c1 * state->rhoW[idx] + c2 * rhoW0 + c3 * dt * w->rhsRhoW[idx]) / c4;
                    }
                    state->rhoEstar[idx] = (c1 * state->rhoE[idx] + c2 * rhoE0 + c3 * dt * w->rhsRhoE[idx])  / c4;
                    double p0 = (config->RKOrder > 1) ? state->pres0[idx] : 0.0;
                    state->pAdvected[idx] = (c1 * state->pAdvected[idx] + c2 * p0 + c3 * dt * w->rhsPadvected[idx]) / c4;

                    /* Multi-phase RK update */
                    if (multiPhase) {
                        if (config->RKOrder == 1) {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alphaRho[ph * tc + idx] += dt * w->rhsAlphaRho[ph * n + idx];
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alpha[ph * tc + idx] += dt * w->rhsAlpha[ph * n + idx];
                        } else {
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alphaRho[ph * tc + idx] = (c1 * state->alphaRho[ph * tc + idx] + c2 * state->alphaRho0[ph * tc + idx] + c3 * dt * w->rhsAlphaRho[ph * n + idx]) / c4;
                            for (int ph = 0; ph < nPhases; ++ph)
                                state->alpha[ph * tc + idx] = (c1 * state->alpha[ph * tc + idx] + c2 * state->alpha0[ph * tc + idx] + c3 * dt * w->rhsAlpha[ph * n + idx]) / c4;
                        }

                        /* Recompute rho, clamp and normalize alphas */
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

                    /* Compute star velocities for divergence computation */
                    double rhoSafe = std::max(state->rho[idx], 1e-14);
                    state->velU[idx] = state->rhoUStar[idx] / rhoSafe;
                    if (config->dim >= 2) state->velV[idx] = state->rhoVStar[idx] / rhoSafe;
                    if (config->dim >= 3) state->velW[idx] = state->rhoWStar[idx] / rhoSafe;
                }
            }
        }

        /* Re-fill ghost cells (density + star velocity) before pressure solve */
        mesh_apply_bcs_mpi(mesh, state, VARSET_PRIM, w->halo);

        if (w->params.singlePressureSolve && s < config->RKOrder - 1) {
            /* Intermediate stage: copy star states to conservative (no pressure correction) */
            for (int k = 0; k < mesh->nz; ++k)
                for (int j = 0; j < mesh->ny; ++j)
                    for (int i = 0; i < mesh->nx; ++i) {
                        size_t idx = mesh_index(mesh, i, j, k);
                        state->rhoU[idx] = state->rhoUStar[idx];
                        if (config->dim >= 2) state->rhoV[idx] = state->rhoVStar[idx];
                        if (config->dim >= 3) state->rhoW[idx] = state->rhoWStar[idx];
                        state->rhoE[idx] = state->rhoEstar[idx];
                    }
            mesh_apply_bcs_mpi(mesh, state, VARSET_CONS, w->halo);
        } else {
            semi_implicit_solve_pressure(w, config, mesh, state, dt);
            semi_implicit_correction_step(w, config, mesh, state, dt);
        }
    }

    NVTX_POP();
    return dt;
}
