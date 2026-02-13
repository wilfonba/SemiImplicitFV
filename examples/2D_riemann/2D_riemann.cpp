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

// Four constant states separated by discontinuities at (x,y) = (0.8, 0.8)
void initializeRiemannProblem(const RectilinearMesh& mesh, SolutionState& state, const IdealGasEOS& eos) {
    double xMid = 0.8;
    double yMid = 0.8;

    // Quadrant 1: upper-right (x > 0.5, y > 0.5)
    PrimitiveState q1;
    q1.rho = 1.5;
    q1.u = {0.0, 0.0, 0.0};
    q1.p = 1.5;
    q1.sigma = 0.0;

    // Quadrant 2: upper-left (x < 0.5, y > 0.5)
    PrimitiveState q2;
    q2.rho = 0.532;
    q2.u = {1.206, 0.0, 0.0};
    q2.p = 0.3;
    q2.sigma = 0.0;

    // Quadrant 3: lower-left (x < 0.5, y < 0.5)
    PrimitiveState q3;
    q3.rho = 0.138;
    q3.u = {1.206, 1.206, 0.0};
    q3.p = 0.029;
    q3.sigma = 0.0;

    // Quadrant 4: lower-right (x > 0.5, y < 0.5)
    PrimitiveState q4;
    q4.rho = 0.532;
    q4.u = {0.0, 1.206, 0.0};
    q4.p = 0.3;
    q4.sigma = 0.0;

    for (int j = 0; j < mesh.ny(); ++j) {
        for (int i = 0; i < mesh.nx(); ++i) {
            std::size_t idx = mesh.index(i, j, 0);
            double x = mesh.cellCentroidX(i);
            double y = mesh.cellCentroidY(j);

            const PrimitiveState* W;
            if (x >= xMid && y >= yMid)      W = &q1;
            else if (x < xMid && y >= yMid)  W = &q2;
            else if (x < xMid && y < yMid)   W = &q3;
            else                             W = &q4;

            PrimitiveState Wt = *W;
            Wt.T = eos.temperature(*W);
            state.setPrimitiveState(idx, Wt);

            ConservativeState U = eos.toConservative(*W);
            state.setConservativeState(idx, U);
        }
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    const int N = 500;
    const double length = 1.0;
    const double endTime = 0.8;
    const double outputInterval = 0.008;
    const int printInterval = 1;

    SimulationConfig config;
    config.dim = 2;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = false;
    config.wenoEps = 1e-16;
    config.reconOrder = ReconstructionOrder::WENO5;
    config.explicitParams.cfl = 0.6;
    config.igrParams.alphaCoeff = 2.0;
    config.igrParams.IGRIters = 5;
    config.validate();

    // ---- Mesh setup ----
    RectilinearMesh mesh = rt.createUniformMesh(config, N, 0.0, length, N, 0.0, length);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YLow,  BoundaryCondition::Outflow);
    rt.setBoundaryCondition(mesh, RectilinearMesh::YHigh, BoundaryCondition::Outflow);

    rt.print("Created ", N, "x", N, " mesh");
    if (rt.size() > 1) rt.print(" on ", rt.size(), " ranks");
    rt.print(".\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);
    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);

    ExplicitSolver solver(mesh, riemannSolver, eos, igrSolver, config);

    rt.attachSolver(solver, mesh);

    initializeRiemannProblem(mesh, state, *eos);
    rt.smoothFields(state, mesh, 3);

    VTKSession vtk(rt, "riemann2d", mesh, config);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = outputInterval, .printInterval = printInterval});

    return 0;
}
