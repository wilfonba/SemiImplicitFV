#include "RKTimeStepping.hpp"
#include "MixtureEOS.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace SemiImplicitFV {

double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

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
                               double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
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
                               double cfl, double maxDt) {
    if (!config.isMultiPhase())
        return computeAcousticTimeStep(mesh, state, eos, cfl, maxDt);

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
                                MPI_Comm comm) {
    double localDt = computeAdvectiveTimeStep(mesh, state, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt,
                               MPI_Comm comm) {
    double localDt = computeAcousticTimeStep(mesh, state, eos, cfl, maxDt);
    double globalDt;
    MPI_Allreduce(&localDt, &globalDt, 1, MPI_DOUBLE, MPI_MIN, comm);
    return globalDt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               const SimulationConfig& config,
                               double cfl, double maxDt,
                               MPI_Comm comm) {
    double localDt = computeAcousticTimeStep(mesh, state, eos, config, cfl, maxDt);
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
        }

        if (config.step % params.printInterval == 0 || config.step == 1) {
            double pct = 100.0 * time / params.endTime;

            std::ostringstream oss;
            oss << "  Step " << std::setw(6) << config.step
                << " | t = " << std::scientific << std::setprecision(3) << std::setw(10) << time
                << " | dt = " << std::scientific << std::setprecision(3) << std::setw(10) << dt
                << " | t/T = " << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%"
                << " | step wall = " << std::scientific << std::setprecision(2) << stepWall << " s";

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
