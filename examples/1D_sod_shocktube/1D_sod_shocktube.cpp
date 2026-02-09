#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "LFSolver.hpp"
#include "ExplicitSolver.hpp"
#include "IGR.hpp"
#include "IdealGasEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"

#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <string>

using namespace SemiImplicitFV;

// Initialize Sod shock tube conditions (sharp IC smoothed with heat equation)
void initializeSodProblem(const RectilinearMesh& mesh, SolutionState& state,
                          const IdealGasEOS& eos, int smoothIters = 10) {
    (void)smoothIters;
    double xMid = 0.5;

    PrimitiveState left;
    left.rho = 1.0;
    left.u = {0.0, 0.0, 0.0};
    left.p = 1.0;
    left.sigma = 0.0;

    PrimitiveState right;
    right.rho = 0.125;
    right.u = {0.0, 0.0, 0.0};
    right.p = 0.1;
    right.sigma = 0.0;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        const PrimitiveState& W = (mesh.cellCentroidX(i) < xMid) ? left : right;

        PrimitiveState Wt = W;
        Wt.T = eos.temperature(W);
        state.setPrimitiveState(idx, Wt);
        state.setConservativeState(idx, eos.toConservative(W));
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    const int numCells = 100;
    const double length = 1.0;
    [[maybe_unused]] const double constDt = 1e-4;
    const double endTime = 0.2;

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = true;
    config.reconOrder = ReconstructionOrder::UPWIND3;
    config.explicitParams.cfl = 0.1;
    config.igrParams.alphaCoeff = 10.0;
    config.igrParams.IGRIters = 5;
    config.validate();

    // ---- Mesh setup ----
    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, length);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);

    rt.print("Created mesh with ", numCells, " total cells");
    if (rt.size() > 1) rt.print(" on ", rt.size(), " ranks");
    rt.print(".\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, config);
    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, config);
    rt.attachSolver(solver, mesh);

    initializeSodProblem(mesh, state, *eos, 0);
    rt.smoothFields(state, mesh, 10);

    VTKSession vtk(rt, "1D_sod", mesh);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = 0.002, .printInterval = 1});

    return 0;
}
