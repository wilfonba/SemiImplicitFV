#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "Reconstruction.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"

/* Helper: set up a multi-phase config + mesh + state for MTHINC testing.
   Two phases with ideal gas EOS, MTHINC enabled. */
struct MthincTestFixture {
    SimulationConfig config;
    RectilinearMesh mesh;
    SolutionState state;
    EOSData eos;
    ReconstructorData recon;

    void setup1D(int nx, double beta) {
        config = config_defaults();
        config.dim = 1;
        config.RKOrder = 1;
        config.reconOrder = UPWIND1;
        config.multiPhaseParams.nPhases = 2;
        config.multiPhaseParams.phases[0] = {1.4, 0.0};
        config.multiPhaseParams.phases[1] = {1.6, 0.0};
        config.multiPhaseParams.alphaMin = 1e-8;
        config.mthincParams.enabled = 1;
        config.mthincParams.beta = beta;
        config_validate(&config);

        mesh_init_uniform(&mesh, &config, nx, 0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0);
        solution_state_init(&state, mesh_total_cells(&mesh), &config);
        eos = eos_create_ideal_gas(1.4, 287.0, 1);
        reconstructor_init(&recon, UPWIND1, 1e-6, 1.4, 0.0);
        reconstructor_allocate(&recon, &mesh);
    }

    void setup2D(int nx, int ny, double beta) {
        config = config_defaults();
        config.dim = 2;
        config.RKOrder = 1;
        config.reconOrder = UPWIND1;
        config.multiPhaseParams.nPhases = 2;
        config.multiPhaseParams.phases[0] = {1.4, 0.0};
        config.multiPhaseParams.phases[1] = {1.6, 0.0};
        config.multiPhaseParams.alphaMin = 1e-8;
        config.mthincParams.enabled = 1;
        config.mthincParams.beta = beta;
        config_validate(&config);

        mesh_init_uniform(&mesh, &config, nx, 0.0, 1.0, ny, 0.0, 1.0, 1, 0.0, 1.0);
        solution_state_init(&state, mesh_total_cells(&mesh), &config);
        eos = eos_create_ideal_gas(1.4, 287.0, 1);
        reconstructor_init(&recon, UPWIND1, 1e-6, 1.4, 0.0);
        reconstructor_allocate(&recon, &mesh);
    }

    void teardown() {
        reconstructor_free(&recon);
        solution_state_free(&state);
        mesh_free(&mesh);
    }

    /* Fill all cells with uniform thermodynamic state and a given alpha profile.
       alpha[0] = alphaVal, alpha[1] = 1 - alphaVal */
    void fillUniformWithAlpha(double rho, double p, double alphaVal) {
        size_t tc = state.totalCells;
        for (size_t idx = 0; idx < tc; ++idx) {
            state.rho[idx] = rho;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = p / (1.4 - 1.0);
            state.velU[idx] = 0.0;
            state.pres[idx] = p;
            state.alpha[0 * tc + idx] = alphaVal;
            state.alpha[1 * tc + idx] = 1.0 - alphaVal;
            state.alphaRho[0 * tc + idx] = alphaVal * rho;
            state.alphaRho[1 * tc + idx] = (1.0 - alphaVal) * rho;
        }
    }

    /* Fill with a step function in alpha: alpha=1 for x < xInterface, alpha=0 for x >= xInterface */
    void fillStep1D(double rho, double p, double xInterface) {
        size_t tc = state.totalCells;
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            double a = (x < xInterface) ? 1.0 : 0.0;
            state.rho[idx] = rho;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = p / (1.4 - 1.0);
            state.velU[idx] = 0.0;
            state.pres[idx] = p;
            state.alpha[0 * tc + idx] = a;
            state.alpha[1 * tc + idx] = 1.0 - a;
            state.alphaRho[0 * tc + idx] = a * rho;
            state.alphaRho[1 * tc + idx] = (1.0 - a) * rho;
        }
    }

    /* Fill with a diffuse interface (tanh profile) in 1D */
    void fillDiffuse1D(double rho, double p, double xCenter, double width) {
        size_t tc = state.totalCells;
        for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
            size_t idx = mesh_index(&mesh, i, 0, 0);
            double x = mesh_cellCentroidX(&mesh, i);
            double a = 0.5 * (1.0 + std::tanh((x - xCenter) / width));
            a = std::clamp(a, 0.0, 1.0);
            state.rho[idx] = rho;
            state.rhoU[idx] = 0.0;
            state.rhoE[idx] = p / (1.4 - 1.0);
            state.velU[idx] = 0.0;
            state.pres[idx] = p;
            state.alpha[0 * tc + idx] = a;
            state.alpha[1 * tc + idx] = 1.0 - a;
            state.alphaRho[0 * tc + idx] = a * rho;
            state.alphaRho[1 * tc + idx] = (1.0 - a) * rho;
        }
    }

    /* Fill with a circular interface in 2D: alpha=1 inside circle */
    void fillCircle2D(double rho, double p, double cx, double cy, double radius) {
        size_t tc = state.totalCells;
        for (int j = -mesh.ngy; j < mesh.ny + mesh.ngy; ++j) {
            for (int i = -mesh.ngx; i < mesh.nx + mesh.ngx; ++i) {
                size_t idx = mesh_index(&mesh, i, j, 0);
                double x = mesh_cellCentroidX(&mesh, i);
                double y = mesh_cellCentroidY(&mesh, j);
                double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                double a = (r < radius) ? 1.0 : 0.0;
                state.rho[idx] = rho;
                state.rhoU[idx] = 0.0;
                if (state.rhoV) state.rhoV[idx] = 0.0;
                state.rhoE[idx] = p / (1.4 - 1.0);
                state.velU[idx] = 0.0;
                if (state.velV) state.velV[idx] = 0.0;
                state.pres[idx] = p;
                state.alpha[0 * tc + idx] = a;
                state.alpha[1 * tc + idx] = 1.0 - a;
                state.alphaRho[0 * tc + idx] = a * rho;
                state.alphaRho[1 * tc + idx] = (1.0 - a) * rho;
            }
        }
    }
};

/* ---- Pure cells: MTHINC should NOT modify alpha at faces ---- */
TEST(MTHINC, PureCells_NoModification) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    f.fillUniformWithAlpha(1.0, 1.0, 1.0);  /* alpha = 1 everywhere (pure) */

    /* Run reconstruct without MTHINC to get baseline */
    SimulationConfig noMthinc = f.config;
    noMthinc.mthincParams.enabled = 0;
    reconstruct(&f.recon, &noMthinc, &f.mesh, &f.state);

    /* Save baseline face alpha values */
    std::vector<double> baselineLeft(f.recon.numXFaces);
    std::vector<double> baselineRight(f.recon.numXFaces);
    for (size_t fi = 0; fi < f.recon.numXFaces; ++fi) {
        baselineLeft[fi] = f.recon.xLeft[fi].alpha[0];
        baselineRight[fi] = f.recon.xRight[fi].alpha[0];
    }

    /* Now run with MTHINC */
    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (size_t fi = 0; fi < f.recon.numXFaces; ++fi) {
        EXPECT_DOUBLE_EQ(f.recon.xLeft[fi].alpha[0], baselineLeft[fi])
            << "MTHINC modified pure cell alpha at xLeft face " << fi;
        EXPECT_DOUBLE_EQ(f.recon.xRight[fi].alpha[0], baselineRight[fi])
            << "MTHINC modified pure cell alpha at xRight face " << fi;
    }
    f.teardown();
}

/* ---- Alpha bounds: face values must remain in [0, 1] ---- */
TEST(MTHINC, AlphaBounds_1D) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.05);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        double aL = f.recon.xLeft[fi].alpha[0];
        double aR = f.recon.xRight[fi].alpha[0];
        EXPECT_GE(aL, 0.0) << "xLeft alpha < 0 at face " << i;
        EXPECT_LE(aL, 1.0) << "xLeft alpha > 1 at face " << i;
        EXPECT_GE(aR, 0.0) << "xRight alpha < 0 at face " << i;
        EXPECT_LE(aR, 1.0) << "xRight alpha > 1 at face " << i;
    }
    f.teardown();
}

/* ---- Alpha bounds in 2D ---- */
TEST(MTHINC, AlphaBounds_2D) {
    MthincTestFixture f;
    f.setup2D(16, 16, 2.3);
    f.fillCircle2D(1.0, 1.0, 0.5, 0.5, 0.25);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (size_t fi = 0; fi < f.recon.numXFaces; ++fi) {
        EXPECT_GE(f.recon.xLeft[fi].alpha[0], 0.0);
        EXPECT_LE(f.recon.xLeft[fi].alpha[0], 1.0);
        EXPECT_GE(f.recon.xRight[fi].alpha[0], 0.0);
        EXPECT_LE(f.recon.xRight[fi].alpha[0], 1.0);
    }
    for (size_t fi = 0; fi < f.recon.numYFaces; ++fi) {
        EXPECT_GE(f.recon.yLeft[fi].alpha[0], 0.0);
        EXPECT_LE(f.recon.yLeft[fi].alpha[0], 1.0);
        EXPECT_GE(f.recon.yRight[fi].alpha[0], 0.0);
        EXPECT_LE(f.recon.yRight[fi].alpha[0], 1.0);
    }
    f.teardown();
}

/* ---- Complement: alpha[0] + alpha[1] should sum to ~1 at faces ---- */
TEST(MTHINC, AlphaComplement_1D) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.05);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        double sumL = f.recon.xLeft[fi].alpha[0] + f.recon.xLeft[fi].alpha[1];
        double sumR = f.recon.xRight[fi].alpha[0] + f.recon.xRight[fi].alpha[1];
        /* Each phase is MTHINC'd independently, so the sum won't be exactly 1,
           but it should be close since both phases have complementary profiles */
        EXPECT_NEAR(sumL, 1.0, 0.05) << "alpha sum at xLeft face " << i;
        EXPECT_NEAR(sumR, 1.0, 0.05) << "alpha sum at xRight face " << i;
    }
    f.teardown();
}

/* ---- Monotonicity: for a step function, MTHINC face values should be
        monotone across the interface ---- */
TEST(MTHINC, Monotonicity_StepFunction) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    /* Place interface at x=0.5. Cells to the left have alpha=1, right have alpha=0.
       The diffuse cells near x=0.5 are the interface cells. */
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.08);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    /* The diffuse profile goes from ~0 (left) to ~1 (right), so face
       alpha values should be monotonically non-decreasing */
    for (int i = 1; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        size_t fp = x_face_index(&f.recon, i - 1, 0, 0);
        double cur = f.recon.xLeft[fi].alpha[0];
        double prev = f.recon.xLeft[fp].alpha[0];
        EXPECT_GE(cur + 1e-12, prev)
            << "Non-monotone xLeft alpha at faces " << (i-1) << " -> " << i;
    }
    f.teardown();
}

/* ---- Interface sharpening: MTHINC should produce sharper transitions
        than standard reconstruction ---- */
TEST(MTHINC, InterfaceSharpening) {
    MthincTestFixture f;
    f.setup1D(40, 4.0);  /* High beta for strong sharpening */
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.08);

    /* First: run without MTHINC */
    SimulationConfig noMthinc = f.config;
    noMthinc.mthincParams.enabled = 0;
    reconstruct(&f.recon, &noMthinc, &f.mesh, &f.state);

    /* Measure transition width without MTHINC */
    double noMthincWidth = 0.0;
    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        double a = f.recon.xLeft[fi].alpha[0];
        if (a > 0.01 && a < 0.99) noMthincWidth += 1.0;
    }

    /* Now with MTHINC */
    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    double mthincWidth = 0.0;
    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        double a = f.recon.xLeft[fi].alpha[0];
        if (a > 0.01 && a < 0.99) mthincWidth += 1.0;
    }

    /* MTHINC should produce a sharper (narrower) transition */
    EXPECT_LE(mthincWidth, noMthincWidth)
        << "MTHINC did not sharpen the interface";
    f.teardown();
}

/* ---- gammaEff/piInfEff consistency: after MTHINC, effective EOS
        params at faces should be consistent with the modified alpha ---- */
TEST(MTHINC, EffectiveEOS_Consistency) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.05);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    /* For two ideal gas phases: gammaEff = 1 + 1 / (alpha0/(g0-1) + alpha1/(g1-1))
       piInfEff should be 0 since both phases have pInf=0 */
    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        const PrimitiveState* fL = x_face_left(&f.recon, fi);

        double a0 = fL->alpha[0];
        double a1 = fL->alpha[1];
        double denom = a0 / (1.4 - 1.0) + a1 / (1.6 - 1.0);
        double expectedGamma = (denom > 0.0) ? 1.0 + 1.0 / denom : 1.4;

        EXPECT_NEAR(fL->gammaEff, expectedGamma, 1e-10)
            << "gammaEff mismatch at xLeft face " << i;
        EXPECT_NEAR(fL->piInfEff, 0.0, 1e-14)
            << "piInfEff should be 0 at xLeft face " << i;
    }
    f.teardown();
}

/* ---- Config validation: MTHINC requires multi-phase ---- */
TEST(MTHINC, ConfigValidation_RequiresMultiPhase) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.mthincParams.enabled = 1;
    config.mthincParams.beta = 2.3;
    /* nPhases = 0 (single-phase) */
    EXPECT_THROW(config_validate(&config), std::invalid_argument);
}

/* ---- Config validation: beta must be > 0 ---- */
TEST(MTHINC, ConfigValidation_BetaPositive) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.mthincParams.enabled = 1;
    config.mthincParams.beta = -1.0;
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases[0] = {1.4, 0.0};
    config.multiPhaseParams.phases[1] = {1.6, 0.0};
    EXPECT_THROW(config_validate(&config), std::invalid_argument);
}

/* ---- Ghost cells: MTHINC requires nGhost >= 2 ---- */
TEST(MTHINC, GhostCellRequirement) {
    SimulationConfig config = config_defaults();
    config.dim = 1;
    config.reconOrder = UPWIND1;  /* Only needs 1 ghost cell normally */
    config.mthincParams.enabled = 1;
    config.mthincParams.beta = 2.3;
    config.multiPhaseParams.nPhases = 2;
    config.multiPhaseParams.phases[0] = {1.4, 0.0};
    config.multiPhaseParams.phases[1] = {1.6, 0.0};
    config_validate(&config);

    /* MTHINC bumps nGhost from 1 to 2 */
    EXPECT_GE(config.nGhost, 2);
}

/* ---- No NaN: MTHINC should not produce NaN values ---- */
TEST(MTHINC, NoNaN_1D) {
    MthincTestFixture f;
    f.setup1D(20, 2.3);
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.05);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        EXPECT_FALSE(std::isnan(f.recon.xLeft[fi].alpha[0])) << "NaN at xLeft face " << i;
        EXPECT_FALSE(std::isnan(f.recon.xRight[fi].alpha[0])) << "NaN at xRight face " << i;
        EXPECT_FALSE(std::isnan(f.recon.xLeft[fi].gammaEff)) << "NaN gammaEff at xLeft face " << i;
        EXPECT_FALSE(std::isnan(f.recon.xRight[fi].gammaEff)) << "NaN gammaEff at xRight face " << i;
    }
    f.teardown();
}

/* ---- High beta: extreme sharpness should not break anything ---- */
TEST(MTHINC, HighBeta_NoCrash) {
    MthincTestFixture f;
    f.setup1D(20, 100.0);
    f.fillDiffuse1D(1.0, 1.0, 0.5, 0.05);

    reconstruct(&f.recon, &f.config, &f.mesh, &f.state);

    for (int i = 0; i <= f.mesh.nx; ++i) {
        size_t fi = x_face_index(&f.recon, i, 0, 0);
        double aL = f.recon.xLeft[fi].alpha[0];
        double aR = f.recon.xRight[fi].alpha[0];
        EXPECT_FALSE(std::isnan(aL));
        EXPECT_FALSE(std::isnan(aR));
        EXPECT_GE(aL, 0.0);
        EXPECT_LE(aL, 1.0);
        EXPECT_GE(aR, 0.0);
        EXPECT_LE(aR, 1.0);
    }
    f.teardown();
}
