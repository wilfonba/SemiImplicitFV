// Auto-generated from 1D_advection_SI.jsonc
// Compile: link against SemiImplicitFV library

#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "State.hpp"
#include "HLLCSolver.hpp"
#include "SimulationConfig.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"
#include "IdealGasEOS.hpp"
#include "SemiImplicitSolver.hpp"
#include "GaussSeidelPressureSolver.hpp"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>
#include <functional>

using namespace SemiImplicitFV;

int main(int argc, char** argv) {
    Runtime rt(argc, argv);

    SimulationConfig config;
    config.dim = 1;
    config.nGhost = 3;
    config.RKOrder = 3;
    config.semiImplicit = true;
    config.reconOrder = ReconstructionOrder::WENO5;
    config.semiImplicitParams.cfl = 0.8;
    config.semiImplicitParams.maxDt = 0.01;
    config.semiImplicitParams.maxPressureIters = 100;
    config.semiImplicitParams.pressureTol = 1e-10;
    config.validate();

    RectilinearMesh mesh = rt.createUniformMesh(config, 4096, 0.0, 1.0);
    rt.setBoundaryCondition(mesh, 0, BoundaryCondition::Periodic);
    rt.setBoundaryCondition(mesh, 1, BoundaryCondition::Periodic);



    SolutionState state;
    state.allocate(mesh.totalCells(), config);

    auto eos = std::make_shared<IdealGasEOS>(1.4, 287.0, config);
    auto riemannSolver = std::make_shared<HLLCSolver>(eos, config);


    // Initialize fields
    // Default state
    const double def_rho = 1.225;
    const double def_u   = 10.0;
    const double def_v   = 0.0;
    const double def_w   = 0.0;
    const double def_p   = 101325.0;

    { int k = 0;
    { int j = 0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, j, k);
        double x = mesh.cellCentroidX(i);
        [[maybe_unused]] double y = 0.0;
        [[maybe_unused]] double z = 0.0;

        double rho = def_rho, u = def_u, v = def_v, w = def_w, p = def_p;

        { // Patch 0
            rho = 1.225 * (1.0 + 0.01 * std::exp(-std::pow((x - 0.5) / 0.05, 2)));
        }

        PrimitiveState W;
        W.rho = rho; W.u = {u, v, w}; W.p = p;
        W.T = eos->temperature(W);
        state.setPrimitiveState(idx, W);
        state.setConservativeState(idx, eos->toConservative(W));
    }
    }
    }

    auto pressureSolver = std::make_shared<GaussSeidelPressureSolver>();
    auto solver = std::make_unique<SemiImplicitSolver>(
        mesh, riemannSolver, pressureSolver, eos, nullptr, config);
    rt.attachSolver(*solver, mesh);
    std::function<double(double)> stepFn = [&](double targetDt) {
        return solver->step(config, mesh, state, targetDt);
    };





    VTKSession vtk(rt, "1D_advection_SI", mesh, config, "VTK");

    TimeLoopParams tlp;
    tlp.endTime = 0.1;
    tlp.outputInterval = 0.001;
    tlp.printInterval = 1;
    tlp.acousticDtFn = [&]() {
        return computeAcousticTimeStep(mesh, state, *eos, config,
                                       1.0, 1e30, rt.mpiContext().comm());
    };
    runTimeLoop(rt, config, mesh, state, vtk, stepFn, tlp);

    // Compute error norms against exact solution.
    // The pulse travels u*t = 10*0.1 = 1.0 = domain length,
    // so with periodic BCs the exact solution equals the initial condition.
    double L1 = 0.0, L2 = 0.0, Linf = 0.0;
    { int k = 0;
    { int j = 0;
    for (int i = 0; i < mesh.nx(); ++i) {
        std::size_t idx = mesh.index(i, j, k);
        double x = mesh.cellCentroidX(i);
        double dx = mesh.dx(i);
        double rhoExact = 1.225 * (1.0 + 0.01 * std::exp(-std::pow((x - 0.5) / 0.05, 2)));
        double err = std::abs(state.rho[idx] - rhoExact);
        L1 += err * dx;
        L2 += err * err * dx;
        Linf = std::max(Linf, err);
    }
    }
    }
    L2 = std::sqrt(L2);

    std::cout << "\n=== Error Norms (density) ===\n";
    std::cout << "  L1   = " << L1   << "\n";
    std::cout << "  L2   = " << L2   << "\n";
    std::cout << "  Linf = " << Linf << "\n";

    return 0;
}
