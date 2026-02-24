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
    /* Write initial VTK */
    vtk_session_write(vtk, state, params->startTime);

    {
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
        config->step++;

        if (time >= nextOutput - 1e-12 * params->outputInterval) {
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
        if (doCheckpoint && time >= nextCheckpoint - 1e-12 * params->outputInterval) {
            write_checkpoint("Checkpoint", mesh, state, config, rt->rank);
            nextCheckpoint += params->outputInterval;
            std::ostringstream oss;
            oss << "  Checkpoint written at t = " << time << "\n";
            runtime_print(rt, oss.str().c_str());
        }

        if (config->step % params->printInterval == 0 || config->step == 1) {
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

    {
        std::ostringstream summary;
        summary << "\nSimulation complete: " << config->step << " steps, wall time = "
                << std::fixed << std::setprecision(3) << wallTotal << " s\n";
        runtime_print(rt, summary.str().c_str());
    }

    vtk_session_finalize(vtk);
}
