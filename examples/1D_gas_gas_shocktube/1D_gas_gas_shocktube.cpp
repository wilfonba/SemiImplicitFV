#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "LFSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "MixtureEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>

using namespace SemiImplicitFV;

// Gas-gas shock tube: two ideal gases with different gamma values
// Left:  helium-like (gamma=5/3), alpha1=1-eps, rho=1, p=1, u=0
// Right: air-like (gamma=1.4), alpha1=eps, rho=0.125, p=0.1, u=0
void initializeTwoPhaseShockTube(const RectilinearMesh& mesh,
                                  SolutionState& state,
                                  const MultiPhaseParams& mp) {
    double xMid = 0.5;
    double alphaMin = mp.alphaMin;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x = mesh.cellCentroidX(i);

        double rho, p;
        double alpha1; // volume fraction of phase 0 (helium-like)

        if (x < xMid) {
            // Left state: mostly phase 0 (helium)
            rho = 1.0;
            p = 1.0;
            alpha1 = 1.0 - alphaMin;
        } else {
            // Right state: mostly phase 1 (air)
            rho = 0.125;
            p = 0.1;
            alpha1 = alphaMin;
        }

        double alpha2 = 1.0 - alpha1;

        // Distribute density proportionally to volume fractions
        double rho1 = rho; // phasic density same as mixture for simplicity
        double rho2 = rho;
        state.alphaRho[0][idx] = alpha1 * rho1;
        state.alphaRho[1][idx] = alpha2 * rho2;
        state.alpha[0][idx] = alpha1;

        state.rho[idx] = state.alphaRho[0][idx] + state.alphaRho[1][idx];
        state.rhoU[idx] = 0.0;
        state.velU[idx] = 0.0;
        state.pres[idx] = p;

        // Compute total energy from mixture EOS
        std::vector<double> alphas = {alpha1};
        double ke = 0.0;
        state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(state.rho[idx], p, alphas, ke, mp);
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    const int numCells = 200;
    const double length = 1.0;
    const double endTime = 0.15;

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 2;
    config.RKOrder = 1;
    config.reconOrder = ReconstructionOrder::WENO1;
    config.explicitParams.cfl = 0.5;
    config.explicitParams.maxDt = 1e-2;

    // Two-phase setup: phase 0 = helium-like (gamma=5/3), phase 1 = air-like (gamma=1.4)
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases = {{5.0/3.0, 0.0}, {1.4, 0.0}};
    config.multiPhaseParams.alphaMin = 1e-6;

    config.validate();

    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, length);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);

    rt.print("=== 1D Two-Phase Shock Tube ===\n");
    rt.print("  Cells:  ", numCells, "\n");
    rt.print("  Phases: 2 (helium gamma=5/3, air gamma=1.4)\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    // EOS used as fallback (not primary for multi-phase)
    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, config);

    ExplicitSolver solver(mesh, riemannSolver, eos, nullptr, config);
    rt.attachSolver(solver, mesh);

    initializeTwoPhaseShockTube(mesh, state, config.multiPhaseParams);

    VTKSession vtk(rt, "two_phase_shocktube", mesh);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = 0.01, .printInterval = 10});

    // Print some final diagnostics
    double minAlpha = 1.0, maxAlpha = 0.0;
    double minRho = 1e30, maxRho = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        minAlpha = std::min(minAlpha, state.alpha[0][idx]);
        maxAlpha = std::max(maxAlpha, state.alpha[0][idx]);
        minRho = std::min(minRho, state.rho[idx]);
        maxRho = std::max(maxRho, state.rho[idx]);
    }

    rt.print("\n=== Final Diagnostics ===\n");
    rt.print("  Alpha_0 range: [", minAlpha, ", ", maxAlpha, "]\n");
    rt.print("  Density range: [", minRho, ", ", maxRho, "]\n");

    return 0;
}
