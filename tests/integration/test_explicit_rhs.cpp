#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"
#include "ExplicitSolver.hpp"
#include "Runtime.hpp"
#include "HaloExchange.hpp"
#include "MPIContext.hpp"

TEST(ExplicitRHS, SingleStep_RunsWithoutNaN) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.nGhost = 2;
    config.RKOrder = 1;
    config.reconOrder = UPWIND3;
    config.explicitParams.cfl = 0.5;

    Runtime rt;
    memset(&rt, 0, sizeof(rt));
    MPI_Comm_rank(MPI_COMM_WORLD, &rt.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt.size);

    int periods[3] = {0, 0, 0};
    RectilinearMesh mesh;
    runtime_create_uniform_mesh_1d(&rt, &mesh, &config, 16, 0.0, 1.0, periods);
    runtime_set_bc(&rt, &mesh, XLOW, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, XHIGH, BC_OUTFLOW);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    /* Sod shock tube initial conditions */
    for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        double x = mesh_cellCentroidX(&mesh, i);
        if (x < 0.5) {
            state.rho[idx] = 1.0;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = 1.0 / 0.4;
        } else {
            state.rho[idx] = 0.125;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = 0.1 / 0.4;
        }
        state.velU[idx] = 0.0;
        state.pres[idx] = (x < 0.5) ? 1.0 : 0.1;
    }

    ExplicitSolverWork solver;
    explicit_solver_init(&solver, &mesh, &eos, RS_LF, NULL, &config);
    runtime_attach_explicit(&rt, &solver, &mesh);

    double dt = explicit_step(&solver, &config, &mesh, &state, 1e-4);

    EXPECT_GT(dt, 0.0);

    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        EXPECT_FALSE(std::isnan(state.rho[idx])) << "NaN at cell " << i;
        EXPECT_FALSE(std::isnan(state.rhoU[idx])) << "NaN at cell " << i;
        EXPECT_FALSE(std::isnan(state.rhoE[idx])) << "NaN at cell " << i;
    }

    explicit_solver_free(&solver);
    solution_state_free(&state);
    mesh_free(&mesh);
    if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
    if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
}

TEST(ExplicitRHS, SingleStep_ConservesMass) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.nGhost = 2;
    config.RKOrder = 1;
    config.reconOrder = UPWIND3;
    config.explicitParams.cfl = 0.5;

    Runtime rt;
    memset(&rt, 0, sizeof(rt));
    MPI_Comm_rank(MPI_COMM_WORLD, &rt.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt.size);

    int periods[3] = {0, 0, 0};
    RectilinearMesh mesh;
    runtime_create_uniform_mesh_1d(&rt, &mesh, &config, 32, 0.0, 1.0, periods);
    runtime_set_bc(&rt, &mesh, XLOW, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, XHIGH, BC_OUTFLOW);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    /* Uniform pressure, zero velocity: no flux at outflow boundaries */
    for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        double x = mesh_cellCentroidX(&mesh, i);
        state.rho[idx] = 1.0 + 0.5 * std::sin(2.0 * M_PI * x);
        state.rhoU[idx] = 0.0;
        state.rhoE[idx] = 1.0 / 0.4;
        state.velU[idx] = 0.0;
        state.pres[idx] = 1.0;
    }

    double mass_before = 0.0;
    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        mass_before += state.rho[idx] * mesh_cell_volume(&mesh, i, 0, 0);
    }

    ExplicitSolverWork solver;
    explicit_solver_init(&solver, &mesh, &eos, RS_HLLC, NULL, &config);
    runtime_attach_explicit(&rt, &solver, &mesh);

    explicit_step(&solver, &config, &mesh, &state, 1e-5);

    double mass_after = 0.0;
    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        mass_after += state.rho[idx] * mesh_cell_volume(&mesh, i, 0, 0);
    }

    /* With zero velocity and outflow BCs, mass should be conserved */
    EXPECT_NEAR(mass_after, mass_before, mass_before * 1e-10);

    explicit_solver_free(&solver);
    solution_state_free(&state);
    mesh_free(&mesh);
    if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
    if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
}
