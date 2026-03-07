#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "Reconstruction.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"

/* Helper: create a 1D config + mesh + state, fill rho with a function,
   run reconstruct(), check face values against expectations. */

struct ReconTestFixture {
    SimulationConfig config;
    RectilinearMesh mesh;
    SolutionState state;
    EOSData eos;
    ReconstructorData recon;

    void setup(int nx, ReconstructionOrder order) {
        config = config_defaults();
        config.dim = 1;
        config.RKOrder = 1;
        config.reconOrder = order;
        config_validate(&config);

        mesh_init_uniform(&mesh, &config, nx, 0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0);

        solution_state_init(&state, mesh_total_cells(&mesh), &config);

        eos = eos_create_ideal_gas(1.4, 287.0, 1);

        reconstructor_init(&recon, order, 1e-6, 1.4, 0.0);
        reconstructor_allocate(&recon, &mesh);
    }

    void teardown() {
        reconstructor_free(&recon);
        solution_state_free(&state);
        mesh_free(&mesh);
    }

    /* Fill state with constant rho=val, u=0, p=1, then do cons->prim */
    void fillConstant(double val) {
        for (size_t idx = 0; idx < state.totalCells; ++idx) {
            state.rho[idx] = val;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = 1.0 / (1.4 - 1.0);  /* p = 1.0 */
            state.velU[idx] = 0.0;
            state.pres[idx] = 1.0;
        }
    }

    /* Fill state with linear rho = a + b*x, u=0, p=1 */
    void fillLinear(double a, double b) {
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            double rho = a + b * x;
            state.rho[idx] = rho;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = 1.0 / (1.4 - 1.0);
            state.velU[idx] = 0.0;
            state.pres[idx] = 1.0;
        }
    }
};

/* ---- Constant field: all schemes should reproduce exactly ---- */

TEST(Reconstruction, WENO5_ConstantField) {
    ReconTestFixture f;
    f.setup(20, WENO5);
    f.fillConstant(2.0);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    /* Check interior x-faces */
    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, 2.0, 1e-12)
            << "Face " << i << " left";
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, 2.0, 1e-12)
            << "Face " << i << " right";
    }
    f.teardown();
}

TEST(Reconstruction, WENO3_ConstantField) {
    ReconTestFixture f;
    f.setup(20, WENO3);
    f.fillConstant(3.0);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, 3.0, 1e-12);
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, 3.0, 1e-12);
    }
    f.teardown();
}

TEST(Reconstruction, UPWIND1_ConstantField) {
    ReconTestFixture f;
    f.setup(20, UPWIND1);
    f.fillConstant(5.0);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, 5.0, 1e-12);
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, 5.0, 1e-12);
    }
    f.teardown();
}

/* ---- Linear field: high-order schemes should reproduce exactly ---- */

TEST(Reconstruction, WENO5_LinearField) {
    ReconTestFixture f;
    f.setup(20, WENO5);
    f.fillLinear(1.0, 2.0);  /* rho = 1 + 2*x */

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    double dx = 1.0 / 20.0;
    /* Check a few interior faces (skip boundaries where stencil may degrade) */
    for (int i = 3; i <= f.mesh.nx - 3; ++i) {
        double xFace = i * dx;  /* face position */
        double expected = 1.0 + 2.0 * xFace;
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " left";
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " right";
    }
    f.teardown();
}

TEST(Reconstruction, UPWIND5_LinearField) {
    ReconTestFixture f;
    f.setup(20, UPWIND5);
    f.fillLinear(1.0, 2.0);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    double dx = 1.0 / 20.0;
    for (int i = 3; i <= f.mesh.nx - 3; ++i) {
        double xFace = i * dx;
        double expected = 1.0 + 2.0 * xFace;
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " left";
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " right";
    }
    f.teardown();
}

TEST(Reconstruction, UPWIND3_LinearField) {
    ReconTestFixture f;
    f.setup(20, UPWIND3);
    f.fillLinear(1.0, 2.0);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    double dx = 1.0 / 20.0;
    for (int i = 2; i <= f.mesh.nx - 2; ++i) {
        double xFace = i * dx;
        double expected = 1.0 + 2.0 * xFace;
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_NEAR(x_face_left(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " left";
        EXPECT_NEAR(x_face_right(&f.recon, fi)->rho, expected, 1e-10)
            << "Face " << i << " right";
    }
    f.teardown();
}
