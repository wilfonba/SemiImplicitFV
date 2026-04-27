#include "RKTimeStepping.hpp"
#include "Checkpoint.hpp"
#include "MixtureEOS.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "NvtxRange.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdio>

double computeAdvectiveTimeStep(const RectilinearMesh* mesh,
                                const SolutionState* state,
                                double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh->dim;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double dtCell = mesh_dx(mesh, i) / std::max(std::abs(state->velU[idx]), 1e-14);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh_dy(mesh, j) / std::max(std::abs(state->velV[idx]), 1e-14));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh_dz(mesh, k) / std::max(std::abs(state->velW[idx]), 1e-14));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeAcousticTimeStep(const RectilinearMesh* mesh,
                               const SolutionState* state,
                               const EOSData* eos,
                               double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh->dim;
    const double gamma = eos->gamma;
    const double pInf = eos->pInf;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double c = std::sqrt(gamma * std::max(state->pres[idx] + pInf, 1e-14)
                                     / std::max(state->rho[idx], 1e-14));

                double dtCell = mesh_dx(mesh, i) / (std::abs(state->velU[idx]) + c);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh_dy(mesh, j) / (std::abs(state->velV[idx]) + c));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh_dz(mesh, k) / (std::abs(state->velW[idx]) + c));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeAcousticTimeStep_config(const RectilinearMesh* mesh,
                                      const SolutionState* state,
                                      const EOSData* eos,
                                      const SimulationConfig* config,
                                      double cfl, double maxDt) {
    if (!config_is_multi_phase(config))
        return computeAcousticTimeStep(mesh, state, eos, cfl, maxDt);

    double dt = maxDt;
    int dim = mesh->dim;
    const MultiPhaseParams* mp = &config->multiPhaseParams;
    int nPhases = mp->nPhases;
    size_t tc = state->totalCells;

    double alphas[MAX_PHASES];
    double alphaRhos[MAX_PHASES];

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = state->alpha[ph * tc + idx];
                for (int ph = 0; ph < nPhases; ++ph)
                    alphaRhos[ph] = state->alphaRho[ph * tc + idx];

                double c = mixture_sound_speed(
                    state->rho[idx], state->pres[idx], alphas, alphaRhos,
                    nPhases, mp->phases);

                double dtCell = mesh_dx(mesh, i) / (std::abs(state->velU[idx]) + c);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh_dy(mesh, j) / (std::abs(state->velV[idx]) + c));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh_dz(mesh, k) / (std::abs(state->velW[idx]) + c));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeAdvectiveTimeStep_mpi(const RectilinearMesh* mesh,
                                    const SolutionState* state,
                                    double cfl, double maxDt,
                                    MPI_Comm comm) {
    double localDt = computeAdvectiveTimeStep(mesh, state, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep_mpi(const RectilinearMesh* mesh,
                                   const SolutionState* state,
                                   const EOSData* eos,
                                   double cfl, double maxDt,
                                   MPI_Comm comm) {
    double localDt = computeAcousticTimeStep(mesh, state, eos, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep_config_mpi(const RectilinearMesh* mesh,
                                          const SolutionState* state,
                                          const EOSData* eos,
                                          const SimulationConfig* config,
                                          double cfl, double maxDt,
                                          MPI_Comm comm) {
    double localDt = computeAcousticTimeStep_config(mesh, state, eos, config, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

/* GPU min-reduction over every interior cell.  Reads rho/velU/velV/velW/pres
 * and the mesh xExt/yExt/zExt node arrays directly from device memory. */
double computeAcousticTimeStep_device(const RectilinearMesh* mesh,
                                      const SolutionState* state,
                                      const EOSData* eos,
                                      double cfl, double maxDt) {
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = mesh->dim;
    const double gamma = eos->gamma;
    const double pInf  = eos->pInf;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* rho  = state->rho;
    double* velU = state->velU;
    double* velV = state->velV;
    double* velW = state->velW;
    double* pres = state->pres;

    double dt = maxDt;

    #pragma omp target teams distribute parallel for collapse(3) reduction(min:dt)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double rhoSafe = rho[idx] > 1e-14 ? rho[idx] : 1e-14;
                double pTot    = pres[idx] + pInf;
                if (pTot < 1e-14) pTot = 1e-14;
                double c = std::sqrt(gamma * pTot / rhoSafe);

                double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                double u = velU[idx]; if (u < 0) u = -u;
                double dtCell = dx / (u + c);
                if (dim >= 2) {
                    double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                    double v = velV[idx]; if (v < 0) v = -v;
                    double d = dy / (v + c);
                    if (d < dtCell) dtCell = d;
                }
                if (dim >= 3) {
                    double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                    double w = velW[idx]; if (w < 0) w = -w;
                    double d = dz / (w + c);
                    if (d < dtCell) dtCell = d;
                }
                double dtCfl = cfl * dtCell;
                if (dtCfl < dt) dt = dtCfl;
            }
        }
    }
    return dt;
}

/* Multi-phase variant.  Computes Wood's mixture sound speed inline — the
 * per-phase gamma/pInf are copied into small local arrays and captured by
 * value via map(to:).  Avoids calling mixture_sound_speed() directly so we
 * don't need to map MultiPhaseParams::phases (a host struct) to the device. */
static double computeAcousticTimeStep_multi_device(const RectilinearMesh* mesh,
                                                   const SolutionState* state,
                                                   const MultiPhaseParams* mp,
                                                   double cfl, double maxDt) {
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = mesh->dim;
    const int nPhases = mp->nPhases;
    const size_t tc = state->totalCells;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* rho    = state->rho;
    double* velU   = state->velU;
    double* velV   = state->velV;
    double* velW   = state->velW;
    double* pres   = state->pres;
    double* alpha  = state->alpha;
    double* alphaR = state->alphaRho;

    double phaseGamma[MAX_PHASES];
    double phasePinf[MAX_PHASES];
    for (int ph = 0; ph < nPhases; ++ph) {
        phaseGamma[ph] = mp->phases[ph].gamma;
        phasePinf[ph]  = mp->phases[ph].pInf;
    }

    double dt = maxDt;

    #pragma omp target teams distribute parallel for collapse(3) reduction(min:dt) \
        map(to: phaseGamma[0:MAX_PHASES], phasePinf[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double rhoSafe = rho[idx] > 1e-14 ? rho[idx] : 1e-14;
                double p = pres[idx];

                /* Wood's mixture sound speed: 1/(rho c^2) = sum alpha_k /(rho_k c_k^2) */
                double sumInvRhoC2 = 0.0;
                for (int ph = 0; ph < nPhases; ++ph) {
                    double aph = alpha[(size_t)ph * tc + idx];
                    double arh = alphaR[(size_t)ph * tc + idx];
                    double aSafe = aph > 1e-14 ? aph : 1e-14;
                    double arSafe = arh > 1e-14 ? arh : 1e-14;
                    double rho_k = arSafe / aSafe;
                    double gk = phaseGamma[ph];
                    double pInfk = phasePinf[ph];
                    double ck2 = gk * (p + pInfk) / rho_k;
                    if (ck2 < 1e-14) ck2 = 1e-14;
                    sumInvRhoC2 += aph / (rho_k * ck2);
                }
                double invRhoSum = sumInvRhoC2 > 1e-30 ? sumInvRhoC2 : 1e-30;
                double c2 = 1.0 / (rhoSafe * invRhoSum);
                double c = std::sqrt(c2 > 0.0 ? c2 : 0.0);

                double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                double u = velU[idx]; if (u < 0) u = -u;
                double dtCell = dx / (u + c);
                if (dim >= 2) {
                    double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                    double v = velV[idx]; if (v < 0) v = -v;
                    double d = dy / (v + c);
                    if (d < dtCell) dtCell = d;
                }
                if (dim >= 3) {
                    double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                    double w = velW[idx]; if (w < 0) w = -w;
                    double d = dz / (w + c);
                    if (d < dtCell) dtCell = d;
                }
                double dtCfl = cfl * dtCell;
                if (dtCfl < dt) dt = dtCfl;
            }
        }
    }
    return dt;
}

double computeAcousticTimeStep_config_mpi_device(const RectilinearMesh* mesh,
                                                 SolutionState* state,
                                                 const EOSData* eos,
                                                 const SimulationConfig* config,
                                                 double cfl, double maxDt,
                                                 MPI_Comm comm) {
    double localDt;
    if (!config_is_multi_phase(config)) {
        localDt = computeAcousticTimeStep_device(mesh, state, eos, cfl, maxDt);
    } else {
        localDt = computeAcousticTimeStep_multi_device(
            mesh, state, &config->multiPhaseParams, cfl, maxDt);
    }
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeViscousDt(const RectilinearMesh* mesh,
                        const SolutionState* state,
                        double mu, double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh->dim;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double dxMin = mesh_dx(mesh, i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh_dy(mesh, j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh_dz(mesh, k));

                double nu = mu / std::max(state->rho[idx], 1e-14);
                double dtCell = dxMin * dxMin / (2.0 * dim * nu);

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeViscousDt_mpi(const RectilinearMesh* mesh,
                            const SolutionState* state,
                            double mu, double cfl, double maxDt,
                            MPI_Comm comm) {
    double localDt = computeViscousDt(mesh, state, mu, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeViscousDt_config(const RectilinearMesh* mesh,
                               const SolutionState* state,
                               const SimulationConfig* config,
                               double cfl, double maxDt) {
    const ViscousParams* vp = &config->viscousParams;
    int perPhase = (vp->nPhaseMu > 0);
    if (!perPhase) {
        return computeViscousDt(mesh, state, vp->mu, cfl, maxDt);
    }

    double dt = maxDt;
    int dim = mesh->dim;
    int nPhases = config->multiPhaseParams.nPhases;
    size_t tc = state->totalCells;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double muEff = 0.0;
                for (int ph = 0; ph < nPhases; ++ph)
                    muEff += state->alpha[ph * tc + idx] * vp->phaseMu[ph];

                if (muEff <= 0.0) continue;

                double dxMin = mesh_dx(mesh, i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh_dy(mesh, j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh_dz(mesh, k));

                double nu = muEff / std::max(state->rho[idx], 1e-14);
                double dtCell = dxMin * dxMin / (2.0 * dim * nu);

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeViscousDt_config_mpi(const RectilinearMesh* mesh,
                                   const SolutionState* state,
                                   const SimulationConfig* config,
                                   double cfl, double maxDt,
                                   MPI_Comm comm) {
    double localDt = computeViscousDt_config(mesh, state, config, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

/* GPU min-reduction for the viscous dt.  Handles the constant-mu and
 * per-phase mixture cases uniformly — the per-phase branch is taken based on
 * the nPhaseMu flag which is uniform per launch. */
static double computeViscousDt_config_device(const RectilinearMesh* mesh,
                                             const SolutionState* state,
                                             const SimulationConfig* config,
                                             double cfl, double maxDt)
{
    const int nx = mesh->nx, ny = mesh->ny, nz = mesh->nz;
    const int ngx = mesh->ngx, ngy = mesh->ngy, ngz = mesh->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = mesh->dim;
    const ViscousParams* vp = &config->viscousParams;
    const int perPhase = (vp->nPhaseMu > 0);
    const int nPhases = config->multiPhaseParams.nPhases;
    const double muConst = vp->mu;
    const size_t tc = state->totalCells;
    const double* xExt = mesh->xNodesExt;
    const double* yExt = mesh->yNodesExt;
    const double* zExt = mesh->zNodesExt;
    double* rho   = state->rho;
    double* alpha = state->alpha;

    double phaseMu[MAX_PHASES];
    for (int ph = 0; ph < MAX_PHASES; ++ph)
        phaseMu[ph] = (perPhase && ph < nPhases) ? vp->phaseMu[ph] : 0.0;

    double dt = maxDt;

    #pragma omp target teams distribute parallel for collapse(3) reduction(min:dt) \
        map(to: phaseMu[0:MAX_PHASES])
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t idx = (size_t)((i + ngx) + nxTot * ((j + ngy) + nyTot * (k + ngz)));

                double muEff = muConst;
                if (perPhase) {
                    muEff = 0.0;
                    for (int ph = 0; ph < nPhases; ++ph)
                        muEff += alpha[(size_t)ph * tc + idx] * phaseMu[ph];
                }
                if (muEff <= 0.0) continue;

                double dx = xExt[i + ngx + 1] - xExt[i + ngx];
                double dxMin = dx;
                if (dim >= 2) {
                    double dy = yExt[j + ngy + 1] - yExt[j + ngy];
                    if (dy < dxMin) dxMin = dy;
                }
                if (dim >= 3) {
                    double dz = zExt[k + ngz + 1] - zExt[k + ngz];
                    if (dz < dxMin) dxMin = dz;
                }

                double rhoSafe = rho[idx] > 1e-14 ? rho[idx] : 1e-14;
                double nu = muEff / rhoSafe;
                double dtCell = dxMin * dxMin / (2.0 * dim * nu);
                double dtCfl = cfl * dtCell;
                if (dtCfl < dt) dt = dtCfl;
            }
        }
    }
    return dt;
}

double computeViscousDt_config_mpi_device(const RectilinearMesh* mesh,
                                          const SolutionState* state,
                                          const SimulationConfig* config,
                                          double cfl, double maxDt,
                                          MPI_Comm comm) {
    double localDt = computeViscousDt_config_device(mesh, state, config, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeCapillaryDt(const RectilinearMesh* mesh,
                          const SolutionState* state,
                          double sigma, double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh->dim;

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                double dxMin = mesh_dx(mesh, i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh_dy(mesh, j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh_dz(mesh, k));

                double dtCell = std::sqrt(
                    std::max(state->rho[idx], 1e-14) * dxMin * dxMin * dxMin / sigma);

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }
    return dt;
}

double computeCapillaryDt_mpi(const RectilinearMesh* mesh,
                              const SolutionState* state,
                              double sigma, double cfl, double maxDt,
                              MPI_Comm comm) {
    double localDt = computeCapillaryDt(mesh, state, sigma, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

TimeLoopParams time_loop_params_defaults(void) {
    TimeLoopParams p;
    p.endTime = 0.0;
    p.outputInterval = 0.0;
    p.printInterval = 1;
    p.checkNaN = 1;
    p.acousticDtFn = NULL;
    p.acousticDtCtx = NULL;
    p.checkpoint = 0;
    p.startTime = 0.0;
    p.restarting = 0;
    return p;
}

void run_time_loop(
    Runtime* rt,
    SimulationConfig* config,
    const RectilinearMesh* mesh,
    SolutionState* state,
    VTKSession* vtk,
    StepFn stepFn,
    void* stepCtx,
    const TimeLoopParams* params)
{
    /* Write initial VTK (skip on restart — already written before checkpoint) */
    if (!params->restarting) {
        vtk_session_write(vtk, state, params->startTime);
    }

    if (params->printInterval > 0) {
        std::ostringstream oss;
        oss << "Running simulation to t = " << params->endTime << "...\n";
        runtime_print(rt, oss.str().c_str());
    }

    double time = params->startTime;
    double nextOutput = params->outputInterval;
    if (time > 0.0) {
        while (nextOutput <= time + 1e-12 * params->outputInterval)
            nextOutput += params->outputInterval;
    }

    int doCheckpoint = params->checkpoint;
    double nextCheckpoint = doCheckpoint ? params->outputInterval : params->endTime + 1.0;
    if (doCheckpoint && time > 0.0) {
        while (nextCheckpoint <= time + 1e-12 * params->outputInterval)
            nextCheckpoint += params->outputInterval;
    }
    double wallTotal = 0.0;

    while (time < params->endTime) {
        double targetDt = params->endTime - time;
        if (nextOutput < params->endTime) {
            targetDt = std::min(targetDt, nextOutput - time);
        }

        config->time = time;
        auto t0 = std::chrono::high_resolution_clock::now();
        double dt;
        {
            NVTX_PUSH("TimeStep");
            dt = stepFn(targetDt, stepCtx);
            NVTX_POP();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double stepWall = std::chrono::duration<double>(t1 - t0).count();
        wallTotal += stepWall;

        time += dt;
        config->time = time;
        config->step++;

        /* Pull primitive + conservative state back to the host only when an
         * I/O event (VTK dump, NaN scan, checkpoint) actually needs it.  In
         * the GPU-offload path this is the single per-interval sync; steps
         * with no output keep the state device-resident across RK stages.
         * Compiled without `-mp=gpu` the `target update` pragmas become
         * no-ops, so host-only builds are unaffected. */
        bool doOutputNow = (time >= nextOutput - 1e-12 * params->outputInterval);
        bool doCheckpointNow = (doCheckpoint &&
                                time >= nextCheckpoint - 1e-12 * params->outputInterval);
        if (doOutputNow || doCheckpointNow) {
            solution_state_update_prim_from_device(state);
        }

        if (doOutputNow) {
            NVTX_PUSH("VTK Output");
            vtk_session_write(vtk, state, time);
            NVTX_POP();
            nextOutput += params->outputInterval;

            /* NaN check at each I/O step */
            if (params->checkNaN) {
                const char* nanField = NULL;
                for (int k = 0; k < mesh->nz && !nanField; ++k)
                    for (int j = 0; j < mesh->ny && !nanField; ++j)
                        for (int i = 0; i < mesh->nx && !nanField; ++i) {
                            size_t idx = mesh_index(mesh, i, j, k);
                            if (std::isnan(state->rho[idx]))  { nanField = "rho";  break; }
                            if (std::isnan(state->rhoU[idx])) { nanField = "rhoU"; break; }
                            if (std::isnan(state->rhoE[idx])) { nanField = "rhoE"; break; }
                            if (std::isnan(state->pres[idx])) { nanField = "pres"; break; }
                            if (mesh->dim >= 2 && std::isnan(state->rhoV[idx])) { nanField = "rhoV"; break; }
                            if (mesh->dim >= 3 && std::isnan(state->rhoW[idx])) { nanField = "rhoW"; break; }
                        }

                int localNaN = nanField ? 1 : 0;
                int globalNaN = 0;
                MPI_Allreduce(&localNaN, &globalNaN, 1, MPI_INT, MPI_MAX,
                              rt->mpiCtx->cartComm);

                if (globalNaN) {
                    std::ostringstream oss;
                    if (nanField) {
                        oss << "ERROR: NaN detected in field '" << nanField
                            << "' at step " << config->step
                            << ", t = " << time << ". Aborting.\n";
                    } else {
                        oss << "ERROR: NaN detected on another rank at step "
                            << config->step << ", t = " << time << ". Aborting.\n";
                    }
                    runtime_print(rt, oss.str().c_str());
                    vtk_session_finalize(vtk);
                    MPI_Abort(rt->mpiCtx->cartComm, 1);
                }
            }
        }

        /* Checkpoint writing */
        if (doCheckpointNow) {
            write_checkpoint("Checkpoint", mesh, state, config, rt->rank);
            nextCheckpoint += params->outputInterval;
            std::ostringstream oss;
            oss << "  Checkpoint written at t = " << time << "\n";
            runtime_print(rt, oss.str().c_str());
        }

        if (params->printInterval > 0 &&
            (config->step % params->printInterval == 0 || config->step == 1)) {
            double pct = 100.0 * time / params->endTime;

            std::ostringstream oss;
            oss << "  Step " << std::setw(6) << config->step << " (" << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%)"
                << " | t = " << std::scientific << std::setprecision(3) << std::setw(10) << time
                << " | dt = " << std::scientific << std::setprecision(3) << std::setw(10) << dt;

            if (params->acousticDtFn) {
                double dtAcoustic = params->acousticDtFn(params->acousticDtCtx);
                double acousticCFL = dt / std::max(dtAcoustic, 1e-30);
                oss << " | CFL_ac = " << std::fixed << std::setprecision(1) << acousticCFL;
            }

            oss << " | T/step = " << std::scientific << std::setprecision(2) << stepWall << " s";

            if (config->useIGR) {
                double localMaxSigma = 0.0;
                for (int k = 0; k < mesh->nz; ++k)
                    for (int j = 0; j < mesh->ny; ++j)
                        for (int i = 0; i < mesh->nx; ++i)
                            localMaxSigma = std::max(localMaxSigma,
                                std::abs(state->sigma[mesh_index(mesh, i, j, k)]));

                double maxSigma = runtime_reduce_max(rt, localMaxSigma);
                oss << " | max|sigma| = " << std::scientific << std::setprecision(2) << maxSigma;
            }

            oss << "\n";
            runtime_print(rt, oss.str().c_str());
        }
    }

    if (params->printInterval > 0) {
        std::ostringstream summary;
        summary << "\nSimulation complete: " << config->step << " steps, wall time = "
                << std::fixed << std::setprecision(3) << wallTotal << " s\n";
        runtime_print(rt, summary.str().c_str());
    }

    vtk_session_finalize(vtk);
}
