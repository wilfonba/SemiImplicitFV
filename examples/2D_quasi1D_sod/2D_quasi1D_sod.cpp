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

void initializeRiemannProblem(const RectilinearMesh& mesh, SolutionState& state, const IdealGasEOS& eos, int testDir) {
    double xMid = 0.5;

    PrimitiveState left;
    left.rho = 1.0; left.u = {0.0, 0.0, 0.0}; left.p = 1.0; left.sigma = 0.0;
    PrimitiveState right;
    right.rho = 0.125; right.u = {0.0, 0.0, 0.0}; right.p = 0.1; right.sigma = 0.0;

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);
            const PrimitiveState* W;
            if (testDir == 1) { if (x <= xMid) W = &left; else W = &right; }
            else { if (y <= xMid) W = &left; else W = &right; }
            PrimitiveState Wt = *W;
            Wt.T = eos.temperature(*W);
            state.setPrimitiveState(idx, Wt);
            state.setConservativeState(idx, eos.toConservative(*W));
        }
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    const int N1 = 200;
    const int N2 = 200;
    const double length = 1.0;
    const double endTime = 0.2;
    int testDir = 1;

    SimulationConfig config;
    config.dim = 2; config.nGhost = 4; config.RKOrder = 3;
    config.useIGR = true;
    config.reconOrder = ReconstructionOrder::UPWIND5;
    config.explicitParams.cfl = 0.1;
    config.igrParams.alphaCoeff = 10.0;
    config.igrParams.IGRIters = 5;
    config.validate();

    // ---- Mesh setup ----
    RectilinearMesh mesh = rt.createUniformMesh(config, N1, 0.0, length, N2, 0.0, length);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::Outflow);

    rt.print("Created ", N1, "x", N2, " mesh");
    if (rt.size() > 1) rt.print(" on ", rt.size(), " ranks");
    rt.print(".\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<LFSolver>(eos, config);
    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, config);
    rt.attachSolver(solver, mesh);

    initializeRiemannProblem(mesh, state, *eos, testDir);
    rt.smoothFields(state, mesh, 10);

    VTKSession vtk(rt, "quasi1D_sod", mesh, config);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = 0.002, .printInterval = 20});

    return 0;
}
