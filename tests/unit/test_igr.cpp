#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "IGR.hpp"
#include "SimulationConfig.hpp"

/* ---- igr_compute_alpha ---- */

TEST(IGR, ComputeAlpha) {
    IGRParams params;
    memset(&params, 0, sizeof(params));
    params.alphaCoeff = 10.0;

    double dx = 0.01;
    double alpha = igr_compute_alpha(&params, dx);
    EXPECT_NEAR(alpha, 10.0 * 0.01 * 0.01, 1e-14);
}

/* ---- Velocity gradient: uniform flow ---- */

TEST(IGR, VelocityGradient_UniformFlow) {
    /* u = (1, 0, 0) everywhere */
    double u[3] = {1.0, 0.0, 0.0};
    GradientTensor grad;

    igr_compute_velocity_gradient(u, u, u, u, u, u,
                                  0.1, 0.1, 0.1, 3, grad);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(grad[i][j], 0.0, 1e-14)
                << "grad[" << i << "][" << j << "]";
}

/* ---- Velocity gradient: linear shear u = (y, 0, 0) ---- */

TEST(IGR, VelocityGradient_LinearShear) {
    /* u = (y, 0, 0). For cell at y=0.5 with dy=0.1:
       u_ym = (0.4, 0, 0), u_yp = (0.6, 0, 0)
       du/dy = (0.6 - 0.4) / (2*0.1) = 1.0
       All neighbors in x and z have the same u (no x or z variation) */

    double u_xm[3] = {0.5, 0.0, 0.0};
    double u_xp[3] = {0.5, 0.0, 0.0};
    double u_ym[3] = {0.4, 0.0, 0.0};
    double u_yp[3] = {0.6, 0.0, 0.0};
    double u_zm[3] = {0.5, 0.0, 0.0};
    double u_zp[3] = {0.5, 0.0, 0.0};

    GradientTensor grad;
    igr_compute_velocity_gradient(u_xm, u_xp, u_ym, u_yp, u_zm, u_zp,
                                  0.1, 0.1, 0.1, 3, grad);

    /* du/dy = grad[0][1] should be 1.0 */
    EXPECT_NEAR(grad[0][1], 1.0, 1e-14);
    /* All other components should be zero */
    EXPECT_NEAR(grad[0][0], 0.0, 1e-14);
    EXPECT_NEAR(grad[0][2], 0.0, 1e-14);
    EXPECT_NEAR(grad[1][0], 0.0, 1e-14);
    EXPECT_NEAR(grad[1][1], 0.0, 1e-14);
    EXPECT_NEAR(grad[1][2], 0.0, 1e-14);
    EXPECT_NEAR(grad[2][0], 0.0, 1e-14);
    EXPECT_NEAR(grad[2][1], 0.0, 1e-14);
    EXPECT_NEAR(grad[2][2], 0.0, 1e-14);
}

/* ---- Velocity gradient: 1D stretching u = (ax, 0, 0) ---- */

TEST(IGR, VelocityGradient_1DStretching) {
    double a = 3.0;
    double dx = 0.1;
    double x0 = 1.0;

    double u_xm[3] = {a * (x0 - dx), 0.0, 0.0};
    double u_xp[3] = {a * (x0 + dx), 0.0, 0.0};
    double u_ym[3] = {a * x0, 0.0, 0.0};
    double u_yp[3] = {a * x0, 0.0, 0.0};
    double u_zm[3] = {a * x0, 0.0, 0.0};
    double u_zp[3] = {a * x0, 0.0, 0.0};

    GradientTensor grad;
    igr_compute_velocity_gradient(u_xm, u_xp, u_ym, u_yp, u_zm, u_zp,
                                  dx, dx, dx, 1, grad);

    /* du/dx = grad[0][0] = a = 3.0 */
    EXPECT_NEAR(grad[0][0], a, 1e-13);
}

/* ---- igr_compute_rhs: zero gradient gives zero RHS ---- */

TEST(IGR, ComputeRHS_ZeroGradient) {
    SimulationConfig config = config_defaults();
    config.dim = 3;

    GradientTensor grad;
    memset(grad, 0, sizeof(grad));

    double rhs = igr_compute_rhs(&config, grad, 1.0);
    EXPECT_NEAR(rhs, 0.0, 1e-14);
}

/* ---- igr_compute_rhs: known shear flow ---- */

TEST(IGR, ComputeRHS_PureStretching) {
    SimulationConfig config = config_defaults();
    config.dim = 2;

    /* du/dx = 2, dv/dy = 3 (diagonal gradient) */
    GradientTensor grad;
    memset(grad, 0, sizeof(grad));
    grad[0][0] = 2.0;
    grad[1][1] = 3.0;

    double alpha = 0.001;
    double rhs = igr_compute_rhs(&config, grad, alpha);

    /* tr(gradU^T * gradU) = sum_ij gradU[i][j]*gradU[j][i]
       = grad[0][0]*grad[0][0] + grad[1][1]*grad[1][1] = 4 + 9 = 13 */
    /* tr(gradU)^2 = (2 + 3)^2 = 25 */
    /* rhs = alpha * (13 + 25) = 0.001 * 38 = 0.038 */
    EXPECT_NEAR(rhs, alpha * 38.0, 1e-14);
}
