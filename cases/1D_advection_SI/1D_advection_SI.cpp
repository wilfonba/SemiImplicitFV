// Auto-generated from 1D_advection_SI.jsonc
// Compile: link against SemiImplicitFV library

#include "SimulationConfig.hpp"
#include "State.hpp"
#include "EquationOfState.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "RiemannSolver.hpp"
#include "Runtime.hpp"
#include "VTKSession.hpp"
#include "RKTimeStepping.hpp"
#include "SemiImplicitSolver.hpp"
#include "PressureSolver.hpp"

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>

struct AcousticDtContext {
    RectilinearMesh* mesh;
    SolutionState* state;
    EOSData* eos;
    SimulationConfig* config;
    Runtime* rt;
};

static double acoustic_dt_callback(void* ctx) {
    AcousticDtContext* ac = (AcousticDtContext*)ctx;
    return computeAcousticTimeStep_config_mpi(ac->mesh, ac->state, ac->eos, ac->config,
                                              1.0, 1e30, ac->rt->mpiCtx->cartComm);
}

struct StepContext {
    SemiImplicitSolverWork* solver;
    SimulationConfig* config;
    RectilinearMesh* mesh;
    SolutionState* state;
};

static double step_callback(double targetDt, void* ctx) {
    StepContext* sc = (StepContext*)ctx;
    return semi_implicit_step(sc->solver, sc->config, sc->mesh, sc->state, targetDt);
}

int main(int argc, char** argv) {
    Runtime rt;
    runtime_init(&rt, &argc, &argv);

    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.nGhost = 3;
    config.RKOrder = 3;
    config.semiImplicit = 1;
    config.reconOrder = WENO5;
    config.semiImplicitParams.cfl = 0.8;
    config.semiImplicitParams.maxDt = 0.01;
    config.semiImplicitParams.maxPressureIters = 100;
    config.semiImplicitParams.pressureTol = 1e-10;
    config_validate(&config);

    int periods[3] = {1, 0, 0};
    RectilinearMesh mesh;
    runtime_create_uniform_mesh_1d(&rt, &mesh, &config, 4096, 0.0, 1.0, periods);
    runtime_set_bc(&rt, &mesh, XLOW, BC_PERIODIC);
    runtime_set_bc(&rt, &mesh, XHIGH, BC_PERIODIC);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, config.dim);

    // Initialize fields
    // Default state
    const double def_rho = 1.225;
    const double def_u   = 10.0;
    const double def_v   = 0.0;
    const double def_w   = 0.0;
    const double def_p   = 101325.0;

    { int k = 0;
    { int j = 0;
    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, j, k);
        double x = mesh_cellCentroidX(&mesh, i);
        [[maybe_unused]] double y = 0.0;
        [[maybe_unused]] double z = 0.0;

        double rho = def_rho, u = def_u, v = def_v, w = def_w, p = def_p;

        { // Patch 0
            rho = 1.225 * (1.0 + 0.01 * std::exp(-std::pow((x - 0.5) / 0.05, 2)));
        }

        PrimitiveState W = {};
        W.rho = rho; W.u[0] = u; W.u[1] = v; W.u[2] = w; W.p = p;
        W.T = eos_temperature(&eos, &W);
        state_set_primitive(&state, idx, &W);
        { ConservativeState U = eos_to_conservative(&eos, &W); state_set_conservative(&state, idx, &U); }
    }
    }
    }

    PressureSolverData ps;
    pressure_solver_init(&ps, PS_GAUSS_SEIDEL);
    SemiImplicitSolverWork solver;
    semi_implicit_solver_init(&solver, &mesh, &eos, RS_HLLC, &ps, NULL, &config);
    runtime_attach_semi_implicit(&rt, &solver, &mesh);
    StepContext stepCtx = {&solver, &config, &mesh, &state};
    AcousticDtContext acousticCtx = {&mesh, &state, &eos, &config, &rt};

    VTKSession vtk;
    vtk_session_init(&vtk, &rt, "1D_advection_SI", &mesh, &config, "VTK", VTK_TEXT, 0);

    TimeLoopParams tlp = time_loop_params_defaults();
    tlp.endTime = 0.1;
    tlp.outputInterval = 0.001;
    tlp.printInterval = 1;
    tlp.acousticDtFn = acoustic_dt_callback;
    tlp.acousticDtCtx = &acousticCtx;
    run_time_loop(&rt, &config, &mesh, &state, &vtk, step_callback, &stepCtx, &tlp);

    // Compute error norms against exact solution.
    // The pulse travels u*t = 10*0.1 = 1.0 = domain length,
    // so with periodic BCs the exact solution equals the initial condition.
    double L1 = 0.0, L2 = 0.0, Linf = 0.0;
    { int k = 0;
    { int j = 0;
    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, j, k);
        double x = mesh_cellCentroidX(&mesh, i);
        double dx = mesh_dx(&mesh, i);
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

    vtk_session_finalize(&vtk);
    semi_implicit_solver_free(&solver);
    pressure_solver_free(&ps);
    solution_state_free(&state);
    mesh_free(&mesh);
    runtime_free(&rt);

    return 0;
}
