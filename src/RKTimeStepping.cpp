#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "ImmersedBoundary.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace SemiImplicitFV {

double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt,
                                const ImmersedBoundaryMethod* ibm) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (ibm && ibm->isSolid(idx)) continue;

                double dtCell = mesh.dx(i) / std::max(std::abs(state.velU[idx]), 1e-14);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh.dy(j) / std::max(std::abs(state.velV[idx]), 1e-14));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh.dz(k) / std::max(std::abs(state.velW[idx]), 1e-14));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt,
                               const ImmersedBoundaryMethod* ibm) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (ibm && ibm->isSolid(idx)) continue;

                double c = eos.soundSpeed(state.getPrimitiveState(idx));

                double dtCell = mesh.dx(i) / (std::abs(state.velU[idx]) + c);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh.dy(j) / (std::abs(state.velV[idx]) + c));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh.dz(k) / (std::abs(state.velW[idx]) + c));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               const SimulationConfig& config,
                               double cfl, double maxDt,
                               const ImmersedBoundaryMethod* ibm) {
    if (!config.isMultiPhase())
        return computeAcousticTimeStep(mesh, state, eos, cfl, maxDt, ibm);

    double dt = maxDt;
    int dim = mesh.dim();
    const auto& mp = config.multiPhaseParams;
    int nPhases = mp.nPhases;

    // Pre-allocate outside the loop to avoid per-cell heap allocations
    std::vector<double> alphas(nPhases - 1);
    std::vector<double> alphaRhos(nPhases);

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (ibm && ibm->isSolid(idx)) continue;

                for (int ph = 0; ph < nPhases - 1; ++ph)
                    alphas[ph] = state.alpha[ph][idx];
                for (int ph = 0; ph < nPhases; ++ph)
                    alphaRhos[ph] = state.alphaRho[ph][idx];

                double c = MixtureEOS::mixtureSoundSpeed(
                    state.rho[idx], state.pres[idx], alphas, alphaRhos, mp);

                double dtCell = mesh.dx(i) / (std::abs(state.velU[idx]) + c);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh.dy(j) / (std::abs(state.velV[idx]) + c));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh.dz(k) / (std::abs(state.velW[idx]) + c));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt,
                                MPI_Comm comm,
                                const ImmersedBoundaryMethod* ibm) {
    double localDt = computeAdvectiveTimeStep(mesh, state, cfl, maxDt, ibm);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt,
                               MPI_Comm comm,
                               const ImmersedBoundaryMethod* ibm) {
    double localDt = computeAcousticTimeStep(mesh, state, eos, cfl, maxDt, ibm);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               const SimulationConfig& config,
                               double cfl, double maxDt,
                               MPI_Comm comm,
                               const ImmersedBoundaryMethod* ibm) {
    double localDt = computeAcousticTimeStep(mesh, state, eos, config, cfl, maxDt, ibm);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeViscousDt(const RectilinearMesh& mesh,
                        const SolutionState& state,
                        double mu, double cfl, double maxDt,
                        const ImmersedBoundaryMethod* ibm) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (ibm && ibm->isSolid(idx)) continue;

                double dxMin = mesh.dx(i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh.dz(k));

                double nu = mu / std::max(state.rho[idx], 1e-14);
                double dtCell = dxMin * dxMin / (2.0 * dim * nu);

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeViscousDt(const RectilinearMesh& mesh,
                        const SolutionState& state,
                        double mu, double cfl, double maxDt,
                        MPI_Comm comm,
                        const ImmersedBoundaryMethod* ibm) {
    double localDt = computeViscousDt(mesh, state, mu, cfl, maxDt, ibm);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeCapillaryDt(const RectilinearMesh& mesh,
                          const SolutionState& state,
                          double sigma, double cfl, double maxDt,
                          const ImmersedBoundaryMethod* ibm) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                if (ibm && ibm->isSolid(idx)) continue;

                double dxMin = mesh.dx(i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh.dz(k));

                double dtCell = std::sqrt(
                    std::max(state.rho[idx], 1e-14) * dxMin * dxMin * dxMin / sigma);

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeCapillaryDt(const RectilinearMesh& mesh,
                          const SolutionState& state,
                          double sigma, double cfl, double maxDt,
                          MPI_Comm comm,
                          const ImmersedBoundaryMethod* ibm) {
    double localDt = computeCapillaryDt(mesh, state, sigma, cfl, maxDt, ibm);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

void runTimeLoop(
    Runtime& rt,
    SimulationConfig& config,
    const RectilinearMesh& mesh,
    SolutionState& state,
    VTKSession& vtk,
    const std::function<double(double)>& stepFn,
    const TimeLoopParams& params)
{
    // Write initial VTK at t=0
    vtk.write(state, 0.0);

    rt.print("Running simulation to t = ", params.endTime, "...\n");

    double time = 0.0;
    double nextOutput = params.outputInterval;
    double wallTotal = 0.0;

    while (time < params.endTime) {
        // Clamp dt to hit the next output time or endTime exactly
        double targetDt = params.endTime - time;
        if (nextOutput < params.endTime) {
            targetDt = std::min(targetDt, nextOutput - time);
        }

        config.time = time;
        auto t0 = std::chrono::high_resolution_clock::now();
        double dt = stepFn(targetDt);
        auto t1 = std::chrono::high_resolution_clock::now();
        double stepWall = std::chrono::duration<double>(t1 - t0).count();
        wallTotal += stepWall;

        time += dt;
        config.step++;

        if (time >= nextOutput - 1e-12 * params.outputInterval) {
            vtk.write(state, time);
            nextOutput += params.outputInterval;

            // NaN check at each I/O step
            if (params.checkNaN) {
                const char* nanField = nullptr;
                for (int k = 0; k < mesh.nz() && !nanField; ++k)
                    for (int j = 0; j < mesh.ny() && !nanField; ++j)
                        for (int i = 0; i < mesh.nx() && !nanField; ++i) {
                            std::size_t idx = mesh.index(i, j, k);
                            if (std::isnan(state.rho[idx]))  { nanField = "rho";  break; }
                            if (std::isnan(state.rhoU[idx])) { nanField = "rhoU"; break; }
                            if (std::isnan(state.rhoE[idx])) { nanField = "rhoE"; break; }
                            if (std::isnan(state.pres[idx])) { nanField = "pres"; break; }
                            if (mesh.dim() >= 2 && std::isnan(state.rhoV[idx])) { nanField = "rhoV"; break; }
                            if (mesh.dim() >= 3 && std::isnan(state.rhoW[idx])) { nanField = "rhoW"; break; }
                        }

                int localNaN = nanField ? 1 : 0;
                int globalNaN = 0;
                MPI_Allreduce(&localNaN, &globalNaN, 1, MPI_INT, MPI_MAX,
                              rt.mpiContext().comm());

                if (globalNaN) {
                    if (nanField) {
                        rt.print("ERROR: NaN detected in field '", nanField,
                                 "' at step ", config.step,
                                 ", t = ", time, ". Aborting.\n");
                    } else {
                        rt.print("ERROR: NaN detected on another rank at step ",
                                 config.step, ", t = ", time, ". Aborting.\n");
                    }
                    vtk.finalize();
                    MPI_Abort(rt.mpiContext().comm(), 1);
                }
            }
        }

        if (config.step % params.printInterval == 0 || config.step == 1) {
            double pct = 100.0 * time / params.endTime;

            std::ostringstream oss;
            oss << "  Step " << std::setw(6) << config.step << " (" << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%)"
                << " | t = " << std::scientific << std::setprecision(3) << std::setw(10) << time
                << " | dt = " << std::scientific << std::setprecision(3) << std::setw(10) << dt;

            if (params.acousticDtFn) {
                double dtAcoustic = params.acousticDtFn();
                double acousticCFL = dt / std::max(dtAcoustic, 1e-30);
                oss << " | CFL_ac = " << std::fixed << std::setprecision(1) << acousticCFL;
            }

            oss << " | T/step = " << std::scientific << std::setprecision(2) << stepWall << " s";

            if (config.useIGR) {
                double localMaxSigma = 0.0;
                for (int k = 0; k < mesh.nz(); ++k)
                    for (int j = 0; j < mesh.ny(); ++j)
                        for (int i = 0; i < mesh.nx(); ++i)
                            localMaxSigma = std::max(localMaxSigma,
                                std::abs(state.sigma[mesh.index(i, j, k)]));

                double maxSigma = rt.reduceMax(localMaxSigma);
                oss << " | max|sigma| = " << std::scientific << std::setprecision(2) << maxSigma;
            }

            oss << "\n";
            rt.print(oss.str());
        }
    }

    {
        std::ostringstream summary;
        summary << "\nSimulation complete: " << config.step << " steps, wall time = "
                << std::fixed << std::setprecision(3) << wallTotal << " s\n";
        rt.print(summary.str());
    }

    vtk.finalize();
}

} // namespace SemiImplicitFV
