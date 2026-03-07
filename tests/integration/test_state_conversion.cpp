#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"

TEST(StateConversion, ConsToPrimRoundTrip_1D) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.RKOrder = 1;
    config_validate(&config);

    RectilinearMesh mesh;
    mesh_init_uniform(&mesh, &config, 16, 0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    /* Set conservative variables: varying density, uniform velocity and pressure */
    for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        double rho = 1.0 + 0.5 * std::sin(2.0 * M_PI * mesh_cellCentroidX(&mesh, i));
        state.rho[idx] = rho;
        state.rhoU[idx] = rho * 10.0;
        state.rhoE[idx] = 101325.0 / (1.4 - 1.0) + 0.5 * rho * 100.0;
    }

    /* Save original conservatives */
    std::vector<double> rho_orig(state.totalCells);
    std::vector<double> rhoU_orig(state.totalCells);
    std::vector<double> rhoE_orig(state.totalCells);
    for (size_t i = 0; i < state.totalCells; ++i) {
        rho_orig[i] = state.rho[i];
        rhoU_orig[i] = state.rhoU[i];
        rhoE_orig[i] = state.rhoE[i];
    }

    /* cons -> prim -> cons */
    state_cons_to_prim(&state, &mesh, &eos);
    state_prim_to_cons(&state, &mesh, &eos);

    /* Compare */
    for (int i = 0; i < mesh.nx; ++i) {
        size_t idx = mesh_index(&mesh, i, 0, 0);
        EXPECT_NEAR(state.rho[idx], rho_orig[idx], 1e-12)
            << "Cell " << i << " rho";
        EXPECT_NEAR(state.rhoU[idx], rhoU_orig[idx], 1e-10)
            << "Cell " << i << " rhoU";
        EXPECT_NEAR(state.rhoE[idx], rhoE_orig[idx], 1e-6)
            << "Cell " << i << " rhoE";
    }

    solution_state_free(&state);
    mesh_free(&mesh);
}

TEST(StateConversion, ConsToPrimRoundTrip_2D) {
    SimulationConfig config = config_defaults();
    config.dim = 2;
    config.RKOrder = 1;
    config_validate(&config);

    RectilinearMesh mesh;
    mesh_init_uniform(&mesh, &config, 8, 0.0, 1.0, 8, 0.0, 1.0, 1, 0.0, 1.0);

    SolutionState state;
    solution_state_init(&state, mesh_total_cells(&mesh), &config);

    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 2);

    for (int j = -mesh.ngy; j < mesh.ny + mesh.ngy; ++j) {
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, j, 0);
            double rho = 1.0;
            state.rho[idx] = rho;
            state.rhoU[idx] = rho * 10.0;
            state.rhoV[idx] = rho * (-5.0);
            state.rhoE[idx] = 101325.0 / 0.4 + 0.5 * rho * (100.0 + 25.0);
        }
    }

    std::vector<double> rhoE_orig(state.totalCells);
    for (size_t i = 0; i < state.totalCells; ++i)
        rhoE_orig[i] = state.rhoE[i];

    state_cons_to_prim(&state, &mesh, &eos);
    state_prim_to_cons(&state, &mesh, &eos);

    for (int j = 0; j < mesh.ny; ++j) {
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, j, 0);
            EXPECT_NEAR(state.rhoE[idx], rhoE_orig[idx], 1e-6)
                << "Cell (" << i << "," << j << ")";
        }
    }

    solution_state_free(&state);
    mesh_free(&mesh);
}
