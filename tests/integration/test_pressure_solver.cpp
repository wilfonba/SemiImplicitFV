#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "PressureSolver.hpp"

/* Test pressure solver on a simple 1D Poisson problem:
   d^2(p)/dx^2 = rhs, with outflow (zero-gradient) BCs.
   We set rhs to a known function and verify convergence. */

struct PressureSolverFixture {
    SimulationConfig config;
    RectilinearMesh mesh;
    int nx;
    size_t totalCells;

    std::vector<double> rho_arr;
    std::vector<double> rhoc2_arr;
    std::vector<double> rhs_arr;
    std::vector<double> pressure_arr;

    void setup(int nx_) {
        nx = nx_;
        config = config_defaults();
        config.dim = 1;
        config.nGhost = 1;
        config.RKOrder = 1;

        mesh_init_uniform(&mesh, &config, nx, 0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0);
        mesh_set_bc(&mesh, XLOW, BC_OUTFLOW);
        mesh_set_bc(&mesh, XHIGH, BC_OUTFLOW);

        totalCells = mesh_total_cells(&mesh);
        rho_arr.assign(totalCells, 1.0);
        rhoc2_arr.assign(totalCells, 1.0);
        rhs_arr.assign(totalCells, 0.0);
        pressure_arr.assign(totalCells, 0.0);

        /* Set a simple sinusoidal RHS on interior cells */
        for (int i = 0; i < mesh.nx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            rhs_arr[idx] = std::sin(M_PI * x);
        }
    }

    void teardown() {
        mesh_free(&mesh);
    }
};

TEST(PressureSolver, GaussSeidel_Converges) {
    PressureSolverFixture f;
    f.setup(32);

    PressureSolverData ps;
    pressure_solver_init(&ps, PS_GAUSS_SEIDEL);

    int iters = pressure_solve(&ps, &f.mesh,
                               f.rho_arr.data(), f.rhoc2_arr.data(),
                               f.rhs_arr.data(), f.pressure_arr.data(),
                               f.totalCells, 1.0, 1e-10, 50000);

    /* Should converge in a reasonable number of iterations */
    EXPECT_GT(iters, 0);
    EXPECT_LT(iters, 50000);

    /* Verify the solution is not all zeros */
    double maxP = 0.0;
    for (int i = 0; i < f.mesh.nx; ++i) {
        size_t idx = mesh_index(&f.mesh, i, 0, 0);
        maxP = std::max(maxP, std::abs(f.pressure_arr[idx]));
    }
    EXPECT_GT(maxP, 1e-10);

    pressure_solver_free(&ps);
    f.teardown();
}

TEST(PressureSolver, Jacobi_Converges) {
    PressureSolverFixture f;
    f.setup(32);

    PressureSolverData ps;
    pressure_solver_init(&ps, PS_JACOBI);

    int iters = pressure_solve(&ps, &f.mesh,
                               f.rho_arr.data(), f.rhoc2_arr.data(),
                               f.rhs_arr.data(), f.pressure_arr.data(),
                               f.totalCells, 1.0, 1e-10, 50000);

    EXPECT_GT(iters, 0);
    EXPECT_LT(iters, 50000);

    double maxP = 0.0;
    for (int i = 0; i < f.mesh.nx; ++i) {
        size_t idx = mesh_index(&f.mesh, i, 0, 0);
        maxP = std::max(maxP, std::abs(f.pressure_arr[idx]));
    }
    EXPECT_GT(maxP, 1e-10);

    pressure_solver_free(&ps);
    f.teardown();
}

TEST(PressureSolver, GS_and_Jacobi_Agree) {
    PressureSolverFixture f_gs;
    f_gs.setup(32);

    PressureSolverFixture f_jac;
    f_jac.setup(32);

    PressureSolverData ps_gs, ps_jac;
    pressure_solver_init(&ps_gs, PS_GAUSS_SEIDEL);
    pressure_solver_init(&ps_jac, PS_JACOBI);

    pressure_solve(&ps_gs, &f_gs.mesh,
                   f_gs.rho_arr.data(), f_gs.rhoc2_arr.data(),
                   f_gs.rhs_arr.data(), f_gs.pressure_arr.data(),
                   f_gs.totalCells, 1.0, 1e-10, 50000);

    pressure_solve(&ps_jac, &f_jac.mesh,
                   f_jac.rho_arr.data(), f_jac.rhoc2_arr.data(),
                   f_jac.rhs_arr.data(), f_jac.pressure_arr.data(),
                   f_jac.totalCells, 1.0, 1e-10, 50000);

    /* Both should converge to the same solution */
    for (int i = 0; i < f_gs.mesh.nx; ++i) {
        size_t idx = mesh_index(&f_gs.mesh, i, 0, 0);
        EXPECT_NEAR(f_gs.pressure_arr[idx], f_jac.pressure_arr[idx], 1e-6)
            << "Cell " << i;
    }

    pressure_solver_free(&ps_gs);
    pressure_solver_free(&ps_jac);
    f_gs.teardown();
    f_jac.teardown();
}
