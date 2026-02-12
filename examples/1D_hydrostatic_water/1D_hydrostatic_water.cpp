#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "ExplicitSolver.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"
#include "StiffenedGasEOS.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

using namespace SemiImplicitFV;

// ============================================================
//  Solver toggle — flip to switch between explicit and
//  semi-implicit time stepping.
// ============================================================
static constexpr bool useSemiImplicit = true;

// Initialize uniform water column at rest.
// Gravity will drive the formation of the hydrostatic pressure gradient.
void initializeUniform(const RectilinearMesh& mesh, SolutionState& state,
                        const StiffenedGasEOS& eos,
                        double rho0, double p0) {
    PrimitiveState W;
    W.rho = rho0;
    W.u = {0.0, 0.0, 0.0};
    W.p = p0;
    W.sigma = 0.0;
    W.T = eos.temperature(W);

    ConservativeState U = eos.toConservative(W);

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        state.setPrimitiveState(idx, W);
        state.setConservativeState(idx, U);
    }
}

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    // Domain: 10m water column
    const int numCells = 200;
    const double height = 10.0;      // m
    const double endTime = 1.6;

    // Stiffened gas parameters for water
    const double gamma = 4.4;
    const double pInf  = 6.0e8;     // Pa
    const double R     = 1816.0;    // J/(kg*K), diagnostic only

    // Gravity: constant, pointing in -x direction
    const double g = 9.81;          // m/s^2

    // Uniform initial conditions
    const double rho0 = 1000.0;     // kg/m^3
    const double p0   = 1.0e5;      // Pa (atmospheric)

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 4;
    config.RKOrder = 3;
    config.useIGR = false;
    config.semiImplicit = useSemiImplicit;
    config.reconOrder = ReconstructionOrder::WENO5;

    if (useSemiImplicit) {
        config.semiImplicitParams.cfl = 0.4;
        config.semiImplicitParams.maxDt = 1e-3;
        config.semiImplicitParams.pressureTol = 1e-9;
        config.semiImplicitParams.maxPressureIters = 200;
    } else {
        config.explicitParams.cfl = 0.4;
    }

    // Body force: constant gravity in -x direction
    config.bodyForceParams.a[0] = -g;

    config.validate();

    RectilinearMesh mesh = rt.createUniformMesh(config, numCells, 0.0, height);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XLow,  BoundaryCondition::NoSlipWall);
    rt.setBoundaryCondition(mesh, RectilinearMesh::XHigh, BoundaryCondition::NoSlipWall);

    rt.print("1D Hydrostatic Water Column (uniform IC)\n");
    rt.print("  solver = ", useSemiImplicit ? "Semi-implicit" : "Explicit", "\n");
    rt.print("  cells = ", numCells, ", height = ", height, " m, g = ", g, " m/s^2\n");
    rt.print("  rho0 = ", rho0, " kg/m^3, p0 = ", p0, " Pa\n");

    // Expected equilibrium: p(x) = p0 + rho*g*(H/2 - x)
    double dp = rho0 * g * height;
    rt.print("  expected dp (bottom-top) = ", dp, " Pa\n\n");

    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<StiffenedGasEOS>(gamma, pInf, R, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);

    // Build solver
    std::function<double(double)> stepFn;
    std::unique_ptr<ExplicitSolver> explicitSolver;
    std::unique_ptr<SemiImplicitSolver> semiImplicitSolver;

    if (useSemiImplicit) {
        auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();
        semiImplicitSolver = std::make_unique<SemiImplicitSolver>(
            mesh, riemannSolver, pressureSolver, eos, nullptr, config);
        rt.attachSolver(*semiImplicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return semiImplicitSolver->step(config, mesh, state, targetDt);
        };
    } else {
        explicitSolver = std::make_unique<ExplicitSolver>(
            mesh, riemannSolver, eos, nullptr, config);
        rt.attachSolver(*explicitSolver, mesh);
        stepFn = [&](double targetDt) {
            return explicitSolver->step(config, mesh, state, targetDt);
        };
    }

    initializeUniform(mesh, state, *eos, rho0, p0);

    VTKSession vtk(rt, "1D_hydrostatic_water", mesh);

    runTimeLoop(rt, config, mesh, state, vtk, stepFn,
                {.endTime = endTime, .outputInterval = 0.01, .printInterval = 100});

    // Compare final pressure to expected hydrostatic profile:
    //   p_eq(x) = p0 + rho0*g*(H/2 - x)
    // This is the constant-density equilibrium centered so mean pressure = p0.
    double localMaxVel = 0.0, localMaxPresErr = 0.0;
    double localP_bottom = 0.0, localP_top = 0.0;
    bool haveBottom = false, haveTop = false;

    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, 0, 0);
        double x = mesh.cellCentroidX(i);

        double vel = state.rhoU[idx] / state.rho[idx];
        localMaxVel = std::max(localMaxVel, std::abs(vel));

        ConservativeState U;
        U.rho = state.rho[idx];
        U.rhoU = {state.rhoU[idx], 0.0, 0.0};
        U.rhoE = state.rhoE[idx];
        double p = eos->pressure(U);

        double p_eq = p0 + rho0 * g * (height / 2.0 - x);
        double pErr = std::abs(p - p_eq) / (p0 + pInf);
        localMaxPresErr = std::max(localMaxPresErr, pErr);

        if (i == 0)            { localP_bottom = p; haveBottom = true; }
        if (i == mesh.nx() - 1) { localP_top = p; haveTop = true; }
    }

    double maxVel     = rt.reduceMax(localMaxVel);
    double maxPresErr = rt.reduceMax(localMaxPresErr);

    rt.print("\n--- Results ---\n");
    rt.print("  max|u|           = ", std::scientific, maxVel, " m/s\n");
    rt.print("  max|dp/(p+pInf)| = ", std::scientific, maxPresErr, " (vs hydrostatic eq.)\n");
    if (haveBottom)
        rt.print("  p_bottom         = ", std::fixed, std::setprecision(1), localP_bottom, " Pa\n");
    if (haveTop)
        rt.print("  p_top            = ", std::fixed, std::setprecision(1), localP_top, " Pa\n");
    if (haveBottom && haveTop)
        rt.print("  dp (bottom-top)  = ", std::fixed, std::setprecision(1),
                 localP_bottom - localP_top, " Pa  (expected ", dp, ")\n");

    return 0;
}
