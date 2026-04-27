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
#include <cstdio>
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
        double* g = (double*)w->gradU;
        size_t gn = n * 9;
        #pragma omp target enter data map(alloc: g[0:gn])
    }

    /* Device-side RHS scratch buffers — written by the flux kernel, consumed
     * by the RK update kernel.  No host copy needed. */
    double* rRho  = w->rhsRho;
    double* rRhoU = w->rhsRhoU;
    double* rRhoE = w->rhsRhoE;
    #pragma omp target enter data map(alloc: rRho[0:n], rRhoU[0:n], rRhoE[0:n])
    if (dim >= 2) {
        double* rRhoV = w->rhsRhoV;
        #pragma omp target enter data map(alloc: rRhoV[0:n])
    }
    if (dim >= 3) {
        double* rRhoW = w->rhsRhoW;
        #pragma omp target enter data map(alloc: rRhoW[0:n])
    }
    if (w->nPhases > 0) {
        double* rAR = w->rhsAlphaRho;
        double* rA  = w->rhsAlpha;
        double* dU  = w->divU;
        size_t mp = (size_t)w->nPhases * n;
        #pragma omp target enter data map(alloc: rAR[0:mp], rA[0:mp], dU[0:n])
    }

    /* GPU warmup: the first target launch and the first MPI_Allreduce-after-
     * device-sync each pay a multi-second JIT/context-init cost (showed up as
     * a 3.8 s outlier in the dt NVTX range).  Fire a trivial reduction kernel
     * here so the JIT and CUDA context are hot before the time loop starts. */
    {
        double warmup = 0.0;
        double* rRhoEW = w->rhsRhoE;
        #pragma omp target teams distribute parallel for reduction(+:warmup)
        for (size_t ii = 0; ii < n; ++ii) {
            warmup += rRhoEW[ii];
        }
        (void)warmup;
    }
}

void explicit_solver_free(ExplicitSolverWork* w)
{
    size_t n = w->totalCells;
    double* rRho  = w->rhsRho;
    double* rRhoU = w->rhsRhoU;
    double* rRhoE = w->rhsRhoE;
    if (rRho) {
        #pragma omp target exit data map(delete: rRho[0:n], rRhoU[0:n], rRhoE[0:n])
    }
    if (w->dim >= 2 && w->rhsRhoV) {
        double* rRhoV = w->rhsRhoV;
        #pragma omp target exit data map(delete: rRhoV[0:n])
    }
    if (w->dim >= 3 && w->rhsRhoW) {
        double* rRhoW = w->rhsRhoW;
        #pragma omp target exit data map(delete: rRhoW[0:n])
    }
    if (w->nPhases > 0 && w->rhsAlphaRho) {
        double* rAR = w->rhsAlphaRho;
        double* rA  = w->rhsAlpha;
        double* dU  = w->divU;
        size_t mp = (size_t)w->nPhases * n;
        #pragma omp target exit data map(delete: rAR[0:mp], rA[0:mp], dU[0:n])
    }
    if (w->gradU) {
        double* g = (double*)w->gradU;
        size_t gn = n * 9;
        #pragma omp target exit data map(delete: g[0:gn])
    }

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

/* Internal: solve IGR on device */
static void explicit_solve_igr_device(ExplicitSolverWork* w,
    const SimulationConfig* config, const RectilinearMesh* mesh, SolutionState* state)
{
    if (!w->igrSolver) return;
    NVTX_PUSH("Explicit::solveIGR");
    igr_compute_velocity_gradients_device(config, mesh, state, (double*)w->gradU);
    igr_solve_entropic_pressure_mpi_device(config, &config->igrParams, mesh, state,
                                           w->gradU, w->halo);
    NVTX_POP();
}

/* Internal: compute RHS
 *
 * GPU-offload layout.  Each direction's flux application is a cell-based
 * kernel: every cell computes the Riemann flux at its left and right faces
 * in parallel.  Interior faces are therefore computed twice (once per
 * neighbour cell), which is the standard GPU trade-off — it eliminates the
 * read-modify-write race that a face-parallel pattern would introduce and
 * avoids atomics.
 *
 * Only single-phase, no-body-force, no-viscous, no-surface-tension,
 * no-IGR configurations are ported in Phase 1.  Other features fall back
 * to the host path if ENABLE_GPU_OFFLOAD is off, and are explicitly
 * rejected when offload is enabled (see explicit_compute_rhs entry). */
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

    ReconstructorData* rec = &w->reconstructor;
    FluxConfig fc = w->fluxConfig;

    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int recNx = rec->nx, recNy = rec->ny;
    const RiemannSolverType solver = w->solverType;

    double* rhsRho  = w->rhsRho;
    double* rhsRhoU = w->rhsRhoU;
    double* rhsRhoV = w->rhsRhoV;
    double* rhsRhoW = w->rhsRhoW;
    double* rhsRhoE = w->rhsRhoE;
    double* rhsAlphaRho = w->rhsAlphaRho;
    double* rhsAlpha    = w->rhsAlpha;
    double* divU        = w->divU;
    double* alphaArr    = state->alpha;
    double* alphaRhoArr = state->alphaRho;
    const PrimitiveState* xL = rec->xLeft;
    const PrimitiveState* xR = rec->xRight;
    const PrimitiveState* yL = rec->yLeft;
    const PrimitiveState* yR = rec->yRight;
    const PrimitiveState* zL = rec->zLeft;
    const PrimitiveState* zR = rec->zRight;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;

    /* RHS init lives inside the X-direction flux kernel — Flux::X *assigns*
     * (=) into rhsRho/U/V/W/E (and, when multi-phase, rhsAlpha/rhsAlphaRho
     * /divU) for every interior cell, then Y/Z and the source terms
     * accumulate (+=).  This eliminates a full n-cell zero kernel pass per
     * RHS call. */

    /* Per-dimension cell widths from the extended node arrays.  Stored as
     * small device-resident arrays (already mapped via mesh_init).  */

    /* --- X-direction flux: cell-based, each cell computes both of its
     *      X-faces in parallel.  Interior faces are computed twice. --- */
    NVTX_PUSH("Flux::X");
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t fL = (size_t)(i     + (recNx + 1) * (j + recNy * k));
                size_t fR = (size_t)(i + 1 + (recNx + 1) * (j + recNy * k));

                double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                double area = dy * dz;
                double coeff = area / (dx * dy * dz);

                double normal[3] = {1.0, 0.0, 0.0};
                RiemannFlux fluxL = computeFluxDirect(solver, &xL[fL], &xR[fL], normal, &fc);
                RiemannFlux fluxR = computeFluxDirect(solver, &xL[fR], &xR[fR], normal, &fc);

                /* `=` not `+=`: this kernel initialises every interior cell's
                 * RHS slot for this RK stage; downstream Y/Z/source-term
                 * passes accumulate into it. */
                rhsRho[idx]  = coeff * (fluxL.massFlux        - fluxR.massFlux);
                rhsRhoU[idx] = coeff * (fluxL.momentumFlux[0] - fluxR.momentumFlux[0]);
                if (dim >= 2) rhsRhoV[idx] = coeff * (fluxL.momentumFlux[1] - fluxR.momentumFlux[1]);
                if (dim >= 3) rhsRhoW[idx] = coeff * (fluxL.momentumFlux[2] - fluxR.momentumFlux[2]);
                rhsRhoE[idx] = coeff * (fluxL.energyFlux      - fluxR.energyFlux);

                if (multiPhase) {
                    /* Upwind cell index for alphaRho flux:
                     *  - Left face  (i face):    i-1 if massFlux >= 0, else i.
                     *  - Right face (i+1 face):  i   if massFlux >= 0, else i+1.
                     * In linear index space (X-direction stride = 1):
                     *    upwindL = (fluxL.massFlux >= 0) ? idx - 1 : idx
                     *    upwindR = (fluxR.massFlux >= 0) ? idx     : idx + 1 */
                    size_t upL = (fluxL.massFlux >= 0.0) ? (idx - 1) : idx;
                    size_t upR = (fluxR.massFlux >= 0.0) ?  idx      : (idx + 1);
                    for (int ph = 0; ph < nPhases; ++ph) {
                        size_t off = (size_t)ph * tc;
                        double aL = alphaArr[off + upL]; if (aL < 1e-14) aL = 1e-14;
                        double aR = alphaArr[off + upR]; if (aR < 1e-14) aR = 1e-14;
                        double arFluxL = (alphaRhoArr[off + upL] / aL) * fluxL.alphaFlux[ph];
                        double arFluxR = (alphaRhoArr[off + upR] / aR) * fluxR.alphaFlux[ph];
                        rhsAlphaRho[(size_t)ph * n + idx] = coeff * (arFluxL - arFluxR);
                        rhsAlpha   [(size_t)ph * n + idx] = coeff * (fluxL.alphaFlux[ph] - fluxR.alphaFlux[ph]);
                    }
                    divU[idx] = coeff * (fluxR.faceVelocity - fluxL.faceVelocity);
                }
            }
        }
    }
    NVTX_POP();

    /* --- Y-direction flux --- */
    if (dim >= 2) {
        NVTX_PUSH("Flux::Y");
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    size_t fL = (size_t)(i + recNx * (j     + (recNy + 1) * k));
                    size_t fR = (size_t)(i + recNx * (j + 1 + (recNy + 1) * k));

                    double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                    double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                    double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                    double area = dx * dz;
                    double coeff = area / (dx * dy * dz);

                    double normal[3] = {0.0, 1.0, 0.0};
                    RiemannFlux fluxL = computeFluxDirect(solver, &yL[fL], &yR[fL], normal, &fc);
                    RiemannFlux fluxR = computeFluxDirect(solver, &yL[fR], &yR[fR], normal, &fc);

                    rhsRho[idx]  += coeff * (fluxL.massFlux        - fluxR.massFlux);
                    rhsRhoU[idx] += coeff * (fluxL.momentumFlux[0] - fluxR.momentumFlux[0]);
                    rhsRhoV[idx] += coeff * (fluxL.momentumFlux[1] - fluxR.momentumFlux[1]);
                    if (dim >= 3) rhsRhoW[idx] += coeff * (fluxL.momentumFlux[2] - fluxR.momentumFlux[2]);
                    rhsRhoE[idx] += coeff * (fluxL.energyFlux      - fluxR.energyFlux);

                    if (multiPhase) {
                        /* Y-direction stride in linear index = nxTot. */
                        size_t upL = (fluxL.massFlux >= 0.0) ? (idx - (size_t)nxTot) : idx;
                        size_t upR = (fluxR.massFlux >= 0.0) ?  idx                   : (idx + (size_t)nxTot);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            size_t off = (size_t)ph * tc;
                            double aL = alphaArr[off + upL]; if (aL < 1e-14) aL = 1e-14;
                            double aR = alphaArr[off + upR]; if (aR < 1e-14) aR = 1e-14;
                            double arFluxL = (alphaRhoArr[off + upL] / aL) * fluxL.alphaFlux[ph];
                            double arFluxR = (alphaRhoArr[off + upR] / aR) * fluxR.alphaFlux[ph];
                            rhsAlphaRho[(size_t)ph * n + idx] += coeff * (arFluxL - arFluxR);
                            rhsAlpha   [(size_t)ph * n + idx] += coeff * (fluxL.alphaFlux[ph] - fluxR.alphaFlux[ph]);
                        }
                        divU[idx] += coeff * (fluxR.faceVelocity - fluxL.faceVelocity);
                    }
                }
            }
        }
        NVTX_POP();
    }

    /* --- Z-direction flux --- */
    if (dim >= 3) {
        NVTX_PUSH("Flux::Z");
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    size_t fL = (size_t)(i + recNx * (j + recNy * k));
                    size_t fR = (size_t)(i + recNx * (j + recNy * (k + 1)));

                    double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                    double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                    double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                    double area = dx * dy;
                    double coeff = area / (dx * dy * dz);

                    double normal[3] = {0.0, 0.0, 1.0};
                    RiemannFlux fluxL = computeFluxDirect(solver, &zL[fL], &zR[fL], normal, &fc);
                    RiemannFlux fluxR = computeFluxDirect(solver, &zL[fR], &zR[fR], normal, &fc);

                    rhsRho[idx]  += coeff * (fluxL.massFlux        - fluxR.massFlux);
                    rhsRhoU[idx] += coeff * (fluxL.momentumFlux[0] - fluxR.momentumFlux[0]);
                    rhsRhoV[idx] += coeff * (fluxL.momentumFlux[1] - fluxR.momentumFlux[1]);
                    rhsRhoW[idx] += coeff * (fluxL.momentumFlux[2] - fluxR.momentumFlux[2]);
                    rhsRhoE[idx] += coeff * (fluxL.energyFlux      - fluxR.energyFlux);

                    if (multiPhase) {
                        size_t zStride = (size_t)nxTot * (size_t)nyTot;
                        size_t upL = (fluxL.massFlux >= 0.0) ? (idx - zStride) : idx;
                        size_t upR = (fluxR.massFlux >= 0.0) ?  idx             : (idx + zStride);
                        for (int ph = 0; ph < nPhases; ++ph) {
                            size_t off = (size_t)ph * tc;
                            double aL = alphaArr[off + upL]; if (aL < 1e-14) aL = 1e-14;
                            double aR = alphaArr[off + upR]; if (aR < 1e-14) aR = 1e-14;
                            double arFluxL = (alphaRhoArr[off + upL] / aL) * fluxL.alphaFlux[ph];
                            double arFluxR = (alphaRhoArr[off + upR] / aR) * fluxR.alphaFlux[ph];
                            rhsAlphaRho[(size_t)ph * n + idx] += coeff * (arFluxL - arFluxR);
                            rhsAlpha   [(size_t)ph * n + idx] += coeff * (fluxL.alphaFlux[ph] - fluxR.alphaFlux[ph]);
                        }
                        divU[idx] += coeff * (fluxR.faceVelocity - fluxL.faceVelocity);
                    }
                }
            }
        }
        NVTX_POP();
    }

    /* --- Alpha source term: rhsAlpha[ph] += alpha[ph] * divU --- */
    if (multiPhase) {
        NVTX_PUSH("AlphaSource");
        const int nP = nPhases;
        const size_t tcL = tc;
        const size_t nL = n;
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    double dU = divU[idx];
                    for (int ph = 0; ph < nP; ++ph) {
                        rhsAlpha[(size_t)ph * nL + idx] += alphaArr[(size_t)ph * tcL + idx] * dU;
                    }
                }
            }
        }
        NVTX_POP();
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
        NVTX_PUSH("Viscous");
        add_viscous_fluxes_device(config, mesh, state,
            w->rhsRhoU, w->rhsRhoV, w->rhsRhoW, w->rhsRhoE);
        NVTX_POP();
    }

    /* --- Surface tension --- */
    if (config_has_surface_tension(config)) {
        NVTX_PUSH("SurfaceTension");
        add_surface_tension_fluxes(config, mesh, state,
            config->surfaceTensionParams.sigma,
            w->rhsRhoU, w->rhsRhoV, w->rhsRhoW, w->rhsRhoE);
        NVTX_POP();
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

    /* One-shot warmup: the first MPI_Allreduce that consumes a value just
     * produced by a target reduction triggers CUDA-aware MPI handshake
     * (~1-3 s).  Drive a dummy Allreduce now so the first real dt call
     * doesn't pay it on the timing path. */
    if (!w->firstStepDone) {
        w->firstStepDone = 1;
        if (w->halo) {
            double dummy = 0.0, gDummy;
            MPI_Allreduce(&dummy, &gDummy, 1, MPI_DOUBLE, MPI_MIN,
                          w->halo->mpi->cartComm);
            (void)gDummy;
        }
    }

    double dt;
    if (w->params.constDt > 0) {
        dt = w->params.constDt;
    } else {
        NVTX_PUSH("Explicit::dt");
        dt = computeAcousticTimeStep_config_mpi_device(
            mesh, state, &w->eos, config, w->params.cfl, w->params.maxDt,
            w->halo->mpi->cartComm);
        if (config_has_viscosity(config)) {
            dt = std::min(dt, computeViscousDt_config_mpi_device(mesh, state,
                config, w->params.cfl, w->params.maxDt, w->halo->mpi->cartComm));
        }
        if (config_has_surface_tension(config)) {
            dt = std::min(dt, computeCapillaryDt_mpi(mesh, state,
                config->surfaceTensionParams.sigma, w->params.cfl, w->params.maxDt,
                w->halo->mpi->cartComm));
        }
        NVTX_POP();
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

    /* Surface tension / body force flux contributions are still host-only.
     * Everything else (multi-phase reconstruct + RHS, IGR, viscous) runs on
     * device. */
    if (config_has_surface_tension(config) || config_has_body_force(config))
    {
        fprintf(stderr,
            "explicit_step (GPU): surface-tension / body-force terms are\n"
            "                     not yet ported to GPU.\n");
        std::abort();
    }

    size_t n = w->totalCells;
    size_t tc = state->totalCells;
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dimLocal = config->dim;
    const int RKOrder = config->RKOrder;

    double* rho  = state->rho;
    double* rhoU = state->rhoU;
    double* rhoV = state->rhoV;
    double* rhoW = state->rhoW;
    double* rhoE = state->rhoE;
    double* rho0  = state->rho0;
    double* rhoU0 = state->rhoU0;
    double* rhoV0 = state->rhoV0;
    double* rhoW0 = state->rhoW0;
    double* rhoE0 = state->rhoE0;
    double* rhsRho  = w->rhsRho;
    double* rhsRhoU = w->rhsRhoU;
    double* rhsRhoV = w->rhsRhoV;
    double* rhsRhoW = w->rhsRhoW;
    double* rhsRhoE = w->rhsRhoE;
    double* alphaArr    = state->alpha;
    double* alphaRhoArr = state->alphaRho;
    double* alpha0Arr    = state->alpha0;
    double* alphaRho0Arr = state->alphaRho0;
    double* rhsAlphaArr    = w->rhsAlpha;
    double* rhsAlphaRhoArr = w->rhsAlphaRho;

    for (int s = 0; s < RKOrder; ++s) {
        NVTX_PUSH("Explicit::cons2prim");
        if (multiPhase)
            mixture_cons_to_prim_device(mesh, state, &config->multiPhaseParams);
        else
            state_cons_to_prim(state, mesh, &w->eos);
        NVTX_POP();
        NVTX_PUSH("Explicit::BCs");
        mesh_apply_bcs_mpi_device(mesh, state, VARSET_PRIM, w->halo);
        NVTX_POP();

        if (config->useIGR && w->igrSolver)
            explicit_solve_igr_device(w, config, mesh, state);

        explicit_compute_rhs(w, config, mesh, state);

        double c1 = rk_coef[s][0], c2 = rk_coef[s][1];
        double c3 = rk_coef[s][2], c4 = rk_coef[s][3];
        int saveBackup = (s == 0 && RKOrder > 1);
        const int nP = nPhases;
        const int isMulti = multiPhase;
        const double aMin = alphaMin;

        NVTX_PUSH("Explicit::RKupdate");
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));
                    if (saveBackup) {
                        rho0[idx]  = rho[idx];
                        rhoU0[idx] = rhoU[idx];
                        if (dimLocal >= 2) rhoV0[idx] = rhoV[idx];
                        if (dimLocal >= 3) rhoW0[idx] = rhoW[idx];
                        rhoE0[idx] = rhoE[idx];
                        if (isMulti) {
                            for (int ph = 0; ph < nP; ++ph) {
                                size_t off = (size_t)ph * tc;
                                alphaRho0Arr[off + idx] = alphaRhoArr[off + idx];
                                alpha0Arr[off + idx]    = alphaArr[off + idx];
                            }
                        }
                    }
                    if (RKOrder == 1) {
                        rho[idx]  += dt * rhsRho[idx];
                        rhoU[idx] += dt * rhsRhoU[idx];
                        if (dimLocal >= 2) rhoV[idx] += dt * rhsRhoV[idx];
                        if (dimLocal >= 3) rhoW[idx] += dt * rhsRhoW[idx];
                        rhoE[idx] += dt * rhsRhoE[idx];
                        if (isMulti) {
                            for (int ph = 0; ph < nP; ++ph) {
                                size_t offN = (size_t)ph * n;
                                size_t offT = (size_t)ph * tc;
                                alphaRhoArr[offT + idx] += dt * rhsAlphaRhoArr[offN + idx];
                                alphaArr   [offT + idx] += dt * rhsAlphaArr   [offN + idx];
                            }
                        }
                    } else {
                        rho[idx]  = (c1 * rho[idx]  + c2 * rho0[idx]  + c3 * dt * rhsRho[idx])  / c4;
                        rhoU[idx] = (c1 * rhoU[idx] + c2 * rhoU0[idx] + c3 * dt * rhsRhoU[idx]) / c4;
                        if (dimLocal >= 2)
                            rhoV[idx] = (c1 * rhoV[idx] + c2 * rhoV0[idx] + c3 * dt * rhsRhoV[idx]) / c4;
                        if (dimLocal >= 3)
                            rhoW[idx] = (c1 * rhoW[idx] + c2 * rhoW0[idx] + c3 * dt * rhsRhoW[idx]) / c4;
                        rhoE[idx] = (c1 * rhoE[idx] + c2 * rhoE0[idx] + c3 * dt * rhsRhoE[idx]) / c4;
                        if (isMulti) {
                            for (int ph = 0; ph < nP; ++ph) {
                                size_t offN = (size_t)ph * n;
                                size_t offT = (size_t)ph * tc;
                                alphaRhoArr[offT + idx] = (c1 * alphaRhoArr[offT + idx]
                                                         + c2 * alphaRho0Arr[offT + idx]
                                                         + c3 * dt * rhsAlphaRhoArr[offN + idx]) / c4;
                                alphaArr   [offT + idx] = (c1 * alphaArr   [offT + idx]
                                                         + c2 * alpha0Arr   [offT + idx]
                                                         + c3 * dt * rhsAlphaArr   [offN + idx]) / c4;
                            }
                        }
                    }

                    /* Multi-phase positivity / sum-to-one renormalization. */
                    if (isMulti) {
                        double rhoSum = 0.0;
                        for (int ph = 0; ph < nP; ++ph) {
                            size_t offT = (size_t)ph * tc;
                            double v = alphaRhoArr[offT + idx];
                            if (v < 1e-14) v = 1e-14;
                            alphaRhoArr[offT + idx] = v;
                            rhoSum += v;
                        }
                        rho[idx] = rhoSum;
                        double aSum = 0.0;
                        for (int ph = 0; ph < nP; ++ph) {
                            size_t offT = (size_t)ph * tc;
                            double a = alphaArr[offT + idx];
                            if (a < aMin) a = aMin;
                            alphaArr[offT + idx] = a;
                            aSum += a;
                        }
                        for (int ph = 0; ph < nP; ++ph) {
                            size_t offT = (size_t)ph * tc;
                            alphaArr[offT + idx] /= aSum;
                        }
                    }
                }
            }
        }
        NVTX_POP();
    }

    /* Finalize on device only.  The next step's dt computation also runs on
     * device; state is pulled back to the host just before I/O events (see
     * run_time_loop). */
    if (multiPhase)
        mixture_cons_to_prim_device(mesh, state, &config->multiPhaseParams);
    else
        state_cons_to_prim(state, mesh, &w->eos);
    mesh_apply_bcs_mpi_device(mesh, state, VARSET_PRIM, w->halo);

    NVTX_POP();
    return dt;
}
