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

/* Integration test: MTHINC with explicit time stepping on a multi-phase
   advection problem. Verifies that the full pipeline (reconstruction with
   MTHINC → Riemann solver → update) runs without NaN and preserves alpha
   bounds over multiple steps. */
TEST(MTHINCIntegration, ExplicitStep_NoNaN_AlphaBounds) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.RKOrder = 1;
    config.reconOrder = UPWIND1;
    config.explicitParams.cfl = 0.3;
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases[0] = {1.4, 0.0};
    config.multiPhaseParams.phases[1] = {1.6, 0.0};
    config.multiPhaseParams.alphaMin = 1e-8;
    config.mthincParams.enabled = 1;
    config.mthincParams.beta = 2.3;
    config_validate(&config);

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
    size_t tc = state.totalCells;

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    /* Two-phase setup: diffuse interface at x=0.5, uniform pressure */
    for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        double x = mesh_cellCentroidX(&mesh, i);
        double a = 0.5 * (1.0 + std::tanh((x - 0.5) / 0.05));
        a = std::clamp(a, 0.0, 1.0);
        state.rho[idx] = 1.0;
        state.rhoU[idx] = 0.5;  /* moving right */
        state.rhoE[idx] = 1.0 / 0.4 + 0.5 * 0.5 * 0.5;
        state.velU[idx] = 0.5;
        state.pres[idx] = 1.0;
        state.alpha[0 * tc + idx] = a;
        state.alpha[1 * tc + idx] = 1.0 - a;
        state.alphaRho[0 * tc + idx] = a * 1.0;
        state.alphaRho[1 * tc + idx] = (1.0 - a) * 1.0;
    }

    ExplicitSolverWork solver;
    explicit_solver_init(&solver, &mesh, &eos, RS_HLLC, NULL, &config);
    runtime_attach_explicit(&rt, &solver, &mesh);

    /* Take 5 explicit steps */
    for (int step = 0; step < 5; ++step) {
        double dt = explicit_step(&solver, &config, &mesh, &state, 1e-4);
        EXPECT_GT(dt, 0.0) << "dt <= 0 at step " << step;

        /* Check no NaN and alpha bounds */
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            EXPECT_FALSE(std::isnan(state.rho[idx]))
                << "NaN rho at cell " << i << " step " << step;
            EXPECT_FALSE(std::isnan(state.rhoE[idx]))
                << "NaN rhoE at cell " << i << " step " << step;

            double a0 = state.alpha[0 * tc + idx];
            double a1 = state.alpha[1 * tc + idx];
            EXPECT_GE(a0, -1e-10) << "alpha[0] < 0 at cell " << i << " step " << step;
            EXPECT_LE(a0, 1.0 + 1e-10) << "alpha[0] > 1 at cell " << i << " step " << step;
            EXPECT_GE(a1, -1e-10) << "alpha[1] < 0 at cell " << i << " step " << step;
            EXPECT_LE(a1, 1.0 + 1e-10) << "alpha[1] > 1 at cell " << i << " step " << step;
        }
    }

    explicit_solver_free(&solver);
    solution_state_free(&state);
    mesh_free(&mesh);
    if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
    if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
}

/* Integration test: MTHINC should sharpen the interface compared to
   running without it over multiple time steps */
TEST(MTHINCIntegration, SharperInterface_ThanWithout) {
    auto runCase = [](bool enableMthinc) -> double {
        SimulationConfig config = config_defaults();
        config.dim = 1;
        config.RKOrder = 1;
        config.reconOrder = UPWIND1;
        config.explicitParams.cfl = 0.3;
        config.multiPhaseParams.nPhases = 2;
        config.multiPhaseParams.phases[0] = {1.4, 0.0};
        config.multiPhaseParams.phases[1] = {1.6, 0.0};
        config.multiPhaseParams.alphaMin = 1e-8;
        config.mthincParams.enabled = enableMthinc ? 1 : 0;
        config.mthincParams.beta = 2.3;
        config_validate(&config);

        Runtime rt;
        memset(&rt, 0, sizeof(rt));
        MPI_Comm_rank(MPI_COMM_WORLD, &rt.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &rt.size);

        int periods[3] = {0, 0, 0};
        RectilinearMesh mesh;
        runtime_create_uniform_mesh_1d(&rt, &mesh, &config, 64, 0.0, 1.0, periods);
        runtime_set_bc(&rt, &mesh, XLOW, BC_OUTFLOW);
        runtime_set_bc(&rt, &mesh, XHIGH, BC_OUTFLOW);

        SolutionState state;
        solution_state_init(&state, mesh_total_cells(&mesh), &config);
        size_t tc = state.totalCells;

        EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

        /* Diffuse interface at x=0.5, zero velocity, uniform pressure */
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            double a = 0.5 * (1.0 + std::tanh((x - 0.5) / 0.03));
            a = std::clamp(a, 0.0, 1.0);
            state.rho[idx] = 1.0;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = 1.0 / 0.4;
            state.velU[idx] = 0.0;
            state.pres[idx] = 1.0;
            state.alpha[0 * tc + idx] = a;
            state.alpha[1 * tc + idx] = 1.0 - a;
            state.alphaRho[0 * tc + idx] = a;
            state.alphaRho[1 * tc + idx] = 1.0 - a;
        }

        ExplicitSolverWork solver;
        explicit_solver_init(&solver, &mesh, &eos, RS_HLLC, NULL, &config);
        runtime_attach_explicit(&rt, &solver, &mesh);

        for (int step = 0; step < 10; ++step) {
            explicit_step(&solver, &config, &mesh, &state, 1e-4);
        }

        /* Measure transition width: count cells with 0.01 < alpha < 0.99 */
        double width = 0.0;
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double a = state.alpha[0 * tc + idx];
            if (a > 0.01 && a < 0.99) width += 1.0;
        }

        explicit_solver_free(&solver);
        solution_state_free(&state);
        mesh_free(&mesh);
        if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
        if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
        return width;
    };

    double widthWithout = runCase(false);
    double widthWith = runCase(true);

    /* MTHINC should maintain or narrow the interface */
    EXPECT_LE(widthWith, widthWithout + 1.0)
        << "MTHINC produced a wider interface than without ("
        << widthWith << " vs " << widthWithout << " cells)";
}

/* Integration test: 2D explicit step with MTHINC, circular interface */
TEST(MTHINCIntegration, ExplicitStep_2D_NoNaN) {
    SimulationConfig config = config_defaults();
    config.dim = 2;
    config.RKOrder = 1;
    config.reconOrder = UPWIND1;
    config.explicitParams.cfl = 0.3;
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases[0] = {1.4, 0.0};
    config.multiPhaseParams.phases[1] = {1.6, 0.0};
    config.multiPhaseParams.alphaMin = 1e-8;
    config.mthincParams.enabled = 1;
    config.mthincParams.beta = 2.3;
    config_validate(&config);

    Runtime rt;
    memset(&rt, 0, sizeof(rt));
    MPI_Comm_rank(MPI_COMM_WORLD, &rt.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rt.size);

    int periods[3] = {0, 0, 0};
    RectilinearMesh mesh;
    runtime_create_uniform_mesh_2d(&rt, &mesh, &config,
                                   16, 0.0, 1.0, 16, 0.0, 1.0, periods);
    runtime_set_bc(&rt, &mesh, XLOW, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, XHIGH, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, YLOW, BC_OUTFLOW);
    runtime_set_bc(&rt, &mesh, YHIGH, BC_OUTFLOW);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);
    size_t tc = state.totalCells;

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    /* Circular interface at center, zero velocity */
    for (int j = -mesh.ngy; j < mesh.ny + mesh.ngy; ++j) {
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, j, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            double y = mesh_cellCentroidY(&mesh, j);
            double r = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
            double a = (r < 0.25) ? 1.0 : 0.0;
            state.rho[idx] = 1.0;
            state.rhoU[idx] = 0.0;
            state.rhoV[idx] = 0.0;
            state.rhoE[idx] = 1.0 / 0.4;
            state.velU[idx] = 0.0;
            state.velV[idx] = 0.0;
            state.pres[idx] = 1.0;
            state.alpha[0 * tc + idx] = a;
            state.alpha[1 * tc + idx] = 1.0 - a;
            state.alphaRho[0 * tc + idx] = a;
            state.alphaRho[1 * tc + idx] = 1.0 - a;
        }
    }

    ExplicitSolverWork solver;
    explicit_solver_init(&solver, &mesh, &eos, RS_HLLC, NULL, &config);
    runtime_attach_explicit(&rt, &solver, &mesh);

    /* Take 3 steps */
    for (int step = 0; step < 3; ++step) {
        double dt = explicit_step(&solver, &config, &mesh, &state, 1e-4);
        EXPECT_GT(dt, 0.0) << "dt <= 0 at step " << step;
    }

    /* Check no NaN in interior */
    for (int j = 0; j < mesh.ny; ++j) {
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, j, 0);
            EXPECT_FALSE(std::isnan(state.rho[idx]))
                << "NaN rho at (" << i << "," << j << ")";
            EXPECT_FALSE(std::isnan(state.rhoE[idx]))
                << "NaN rhoE at (" << i << "," << j << ")";
        }
    }

    explicit_solver_free(&solver);
    solution_state_free(&state);
    mesh_free(&mesh);
    if (rt.halo) { halo_free(rt.halo); free(rt.halo); }
    if (rt.mpiCtx) { mpi_context_free(rt.mpiCtx); free(rt.mpiCtx); }
}
