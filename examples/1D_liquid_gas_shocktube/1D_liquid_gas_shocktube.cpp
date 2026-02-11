#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "LFSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IdealGasEOS.hpp"
#include "MixtureEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>

using namespace SemiImplicitFV;

// Water-air shock tube (Shyue 1998, Abgrall & Karni 2001).
//
// Two-phase stiffened-gas model:
//   Phase 0 (water):  gamma = 4.4,  pInf = 6.0e8 Pa
//   Phase 1 (air):    gamma = 1.4,  pInf = 0
//
// Initial conditions on [0, 1], interface at x = 0.7:
//   Left  (water): rho = 1000 kg/m^3, p = 1.0e9 Pa, u = 0
//   Right (air):   rho = 50   kg/m^3, p = 1.0e5 Pa, u = 0
//
// The high-pressure water drives a strong rightward shock into the air
// and a leftward rarefaction fan in the water.  This tests the piInfEff
// stiffened-gas mixture support in the Riemann solver.
void initializeLiquidGasShockTube(const RectilinearMesh& mesh,
                                   SolutionState& state,
                                   const MultiPhaseParams& mp) {
    double xInterface = 0.7;
    double alphaMin = mp.alphaMin;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x = mesh.cellCentroidX(i);

        double rho, p;
        double alphaWater; // volume fraction of phase 0 (water)

        if (x < xInterface) {
            // Left state: water at high pressure
            rho = 1000.0;
            p = 1.0e9;
            alphaWater = 1.0 - alphaMin;
        } else {
            // Right state: air at atmospheric pressure
            rho = 50.0;
            p = 1.0e5;
            alphaWater = alphaMin;
        }

        double alphaAir = 1.0 - alphaWater;

        // Phasic densities equal mixture density in each region
        state.alphaRho[0][idx] = alphaWater * rho;
        state.alphaRho[1][idx] = alphaAir * rho;
        state.alpha[0][idx] = alphaWater;

        state.rho[idx] = state.alphaRho[0][idx] + state.alphaRho[1][idx];
        state.rhoU[idx] = 0.0;
        state.velU[idx] = 0.0;
        state.pres[idx] = p;

        // Compute total energy from mixture EOS (accounts for pInf)
        std::vector<double> alphas = {alphaWater};
        double ke = 0.0;
        state.rhoE[idx] = MixtureEOS::mixtureTotalEnergy(state.rho[idx], p, alphas, ke, mp);
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    const int numCells = 500;
    const double length = 1.0;
    const double endTime = 2.4e-4;

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.reconOrder = ReconstructionOrder::WENO5;
    config.explicitParams.cfl = 0.5;
    config.explicitParams.maxDt = 1e-4;

    // Two-phase stiffened-gas setup
    // Phase 0 = water (gamma=4.4, pInf=6e8)
    // Phase 1 = air   (gamma=1.4, pInf=0)
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases = {{4.4, 6.0e8}, {1.4, 0.0}};
    config.multiPhaseParams.alphaMin = 1e-8;

    config.validate();

    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, length);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);

    rt.print("=== 1D Liquid-Gas Shock Tube (Water-Air) ===\n");
    rt.print("  Cells:  ", numCells, "\n");
    rt.print("  Phases: water (gamma=4.4, pInf=6e8), air (gamma=1.4, pInf=0)\n");
    rt.print("  Left:   rho=1000, p=1e9   (water)\n");
    rt.print("  Right:  rho=50,   p=1e5   (air)\n");
    rt.print("  End time: ", endTime, " s\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    // Fallback EOS (not primary for multi-phase)
    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, config);

    ExplicitSolver solver(mesh, riemannSolver, eos, nullptr, config);
    rt.attachSolver(solver, mesh);

    initializeLiquidGasShockTube(mesh, state, config.multiPhaseParams);

    VTKSession vtk(rt, "liquid_gas_shocktube", mesh);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = endTime / 20.0, .printInterval = 50});

    // Final diagnostics
    double minAlpha = 1.0, maxAlpha = 0.0;
    double minRho = 1e30, maxRho = 0.0;
    double minP = 1e30, maxP = 0.0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        minAlpha = std::min(minAlpha, state.alpha[0][idx]);
        maxAlpha = std::max(maxAlpha, state.alpha[0][idx]);
        minRho = std::min(minRho, state.rho[idx]);
        maxRho = std::max(maxRho, state.rho[idx]);
        minP = std::min(minP, state.pres[idx]);
        maxP = std::max(maxP, state.pres[idx]);
    }

    rt.print("\n=== Final Diagnostics ===\n");
    rt.print("  Alpha_water range: [", minAlpha, ", ", maxAlpha, "]\n");
    rt.print("  Density range:     [", minRho, ", ", maxRho, "]\n");
    rt.print("  Pressure range:    [", minP, ", ", maxP, "]\n");

    return 0;
}
