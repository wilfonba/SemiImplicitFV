#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "RusanovSolver.hpp"
#include "LFSolver.hpp"
#include "HLLCSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "ExplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "JacobiPressureSolver.hpp"
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

// ---- Problem parameters ----
static constexpr double rho0    = 1.225;      // Background density  [kg/m³]
static constexpr double p0      = 101325.0;   // Background pressure [Pa]
static constexpr double u0      = 10.0;       // Advection velocity  [m/s]  (Mach ≈ 0.15)
static constexpr double amp     = 0.01;       // Perturbation amplitude (1 %)
static constexpr double xCenter = 0.5;        // Initial pulse centre
static constexpr double sigma   = 0.05;       // Pulse width

// Gaussian density profile with periodic wrapping
double densityProfile(double x, double xc, double L) {
    double dx = x - xc;
    dx -= L * std::round(dx / L);   // shortest periodic distance
    return rho0 * (1.0 + amp * std::exp(-(dx * dx) / (sigma * sigma)));
}

void initializeProblem(const RectilinearMesh& mesh, SolutionState& state,
                       const IdealGasEOS& eos, double xc, double L) {
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x = mesh.cellCentroidX(i);

        PrimitiveState W;
        W.rho   = densityProfile(x, xc, L);
        W.u     = {u0, 0.0, 0.0};
        W.p     = p0;
        W.sigma = 0.0;
        W.T     = eos.temperature(W);

        state.setPrimitiveState(idx, W);

        ConservativeState U = eos.toConservative(W);
        state.setConservativeState(idx, U);
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    // ---- Setup ----
    const int    numCells = 1000;
    const double length   = 1.0;
    const double endTime  = length / u0;   // one full domain traversal

    // Simulation config
    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 3;
    config.RKOrder = 1;
    config.useIGR = true;
    config.semiImplicit = true;
    config.reconOrder = ReconstructionOrder::UPWIND1;
    config.semiImplicitParams.cfl = 0.6;
    config.semiImplicitParams.maxDt = 1e-2;
    config.semiImplicitParams.pressureTol = 1e-9;
    config.semiImplicitParams.maxPressureIters = 200;
    config.igrParams.alphaCoeff = 10.0;
    config.igrParams.IGRIters = 5;
    config.validate();

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);

    PrimitiveState ref;
    ref.rho = rho0;
    ref.u   = {u0, 0.0, 0.0};
    ref.p   = p0;
    double c0   = eos->soundSpeed(ref);
    double mach = u0 / c0;

    rt.print("  Cells:    ", numCells, "\n");
    rt.print("  u0:       ", u0, " m/s\n");
    rt.print("  c0:       ", c0, " m/s\n");
    rt.print("  Mach:     ", mach, "\n");
    rt.print("  End time: ", endTime, " s  (one domain traversal)\n\n");

    // ---- Mesh setup ----
    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, length, {1, 0, 0});
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::Periodic);

    // Allocate solution state
    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto igrSolver = std::make_shared<IGRSolver>(config.igrParams);
    auto riemannSolver  = std::make_shared<LFSolver>(eos, config);
    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();

    SemiImplicitSolver solver(mesh, riemannSolver, pressureSolver, eos, igrSolver, config);
    rt.attachSolver(solver, mesh);

    initializeProblem(mesh, state, *eos, xCenter, length);

    VTKSession vtk(rt, "1D_advection", mesh);

    auto stepFn = [&](double targetDt) {
        return solver.step(config, mesh, state, targetDt);
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = endTime / 100.0, .printInterval = 1});

    return 0;
}
