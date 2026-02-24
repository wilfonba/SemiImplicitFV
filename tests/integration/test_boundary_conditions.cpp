#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"

struct BCTestFixture {
    SimulationConfig config;
    RectilinearMesh mesh;
    SolutionState state;
    EOSData eos;

    void setup(int nx, BoundaryCondition bcLow, BoundaryCondition bcHigh) {
        config = config_defaults();
        config.dim = 1;
        config.nGhost = 3;
        config.RKOrder = 1;
        config.reconOrder = WENO5;

        mesh_init_uniform(&mesh, &config, nx, 0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0);
        mesh_set_bc(&mesh, XLOW, bcLow);
        mesh_set_bc(&mesh, XHIGH, bcHigh);

        solution_state_init(&state, mesh_total_cells(&mesh), &config);
        eos = eos_create_ideal_gas(1.4, 287.0, 1);

        /* Fill interior cells with a known state */
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            state.rho[idx] = 1.0 + x;
            state.rhoU[idx] = (1.0 + x) * 100.0;
            state.rhoE[idx] = 250000.0;
        }
    }

    void teardown() {
        solution_state_free(&state);
        mesh_free(&mesh);
    }
};

TEST(BoundaryConditions, Outflow_CopiesInterior) {
    BCTestFixture f;
    f.setup(8, BC_OUTFLOW, BC_OUTFLOW);

    mesh_apply_bcs(&f.mesh, &f.state, VARSET_CONS);

    /* Check low-side ghost cells copy from the nearest interior cell */
    size_t interior0 = mesh_index(&f.mesh, 0, 0, 0);
    for (int g = 1; g <= f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, -g, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[interior0], 1e-14)
            << "Ghost cell -" << g;
        EXPECT_NEAR(f.state.rhoU[ghost], f.state.rhoU[interior0], 1e-14);
    }

    /* Check high-side ghost cells */
    size_t interiorN = mesh_index(&f.mesh, f.mesh.nx - 1, 0, 0);
    for (int g = 0; g < f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, f.mesh.nx + g, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[interiorN], 1e-14)
            << "Ghost cell nx+" << g;
    }

    f.teardown();
}

TEST(BoundaryConditions, Symmetry_MirrorsCells) {
    BCTestFixture f;
    f.setup(8, BC_SYMMETRY, BC_SYMMETRY);

    mesh_apply_bcs(&f.mesh, &f.state, VARSET_CONS);

    /* Symmetry BC: mirrors interior cells without velocity reflection */
    for (int g = 1; g <= f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, -g, 0, 0);
        size_t mirror = mesh_index(&f.mesh, g - 1, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[mirror], 1e-14)
            << "Symmetry low ghost -" << g << " rho";
        EXPECT_NEAR(f.state.rhoU[ghost], f.state.rhoU[mirror], 1e-14)
            << "Symmetry low ghost -" << g << " rhoU";
    }

    f.teardown();
}

TEST(BoundaryConditions, Periodic_WrapsAround) {
    BCTestFixture f;
    f.setup(8, BC_PERIODIC, BC_PERIODIC);

    mesh_apply_bcs(&f.mesh, &f.state, VARSET_CONS);

    /* Low-side ghosts should match high-side interior cells */
    for (int g = 1; g <= f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, -g, 0, 0);
        size_t wrap = mesh_index(&f.mesh, f.mesh.nx - g, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[wrap], 1e-14)
            << "Periodic low ghost -" << g;
    }

    /* High-side ghosts should match low-side interior cells */
    for (int g = 0; g < f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, f.mesh.nx + g, 0, 0);
        size_t wrap = mesh_index(&f.mesh, g, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[wrap], 1e-14)
            << "Periodic high ghost +" << g;
    }

    f.teardown();
}

TEST(BoundaryConditions, SlipWall_ReflectsNormalVelocity) {
    BCTestFixture f;
    f.setup(8, BC_SLIP_WALL, BC_SLIP_WALL);

    mesh_apply_bcs(&f.mesh, &f.state, VARSET_CONS);

    /* Slip wall: normal velocity (u) reflected, tangential preserved */
    for (int g = 1; g <= f.mesh.ngx; ++g) {
        size_t ghost = mesh_index(&f.mesh, -g, 0, 0);
        size_t mirror = mesh_index(&f.mesh, g - 1, 0, 0);
        EXPECT_NEAR(f.state.rho[ghost], f.state.rho[mirror], 1e-14);
        EXPECT_NEAR(f.state.rhoU[ghost], -f.state.rhoU[mirror], 1e-14);
    }

    f.teardown();
}
