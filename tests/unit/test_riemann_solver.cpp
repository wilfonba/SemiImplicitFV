#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "RiemannSolver.hpp"
#include "State.hpp"

/* Helper to create a primitive state with gammaEff/piInfEff set */
static PrimitiveState makePrim(double rho, double u, double v, double w,
                                double p, double gamma) {
    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = rho;
    W.u[0] = u;
    W.u[1] = v;
    W.u[2] = w;
    W.p = p;
    W.gammaEff = gamma;
    W.piInfEff = 0.0;
    return W;
}

/* ---- Identical states: flux = physical flux ---- */

TEST(RiemannSolver, LF_IdenticalStates) {
    PrimitiveState W = makePrim(1.0, 1.0, 0.0, 0.0, 1.0, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f = computeLFFlux(&W, &W, normal, &fc);

    /* Physical mass flux = rho * u = 1.0 */
    EXPECT_NEAR(f.massFlux, 1.0, 1e-12);
    /* Momentum flux = rho*u*u + p = 1.0 + 1.0 = 2.0 */
    EXPECT_NEAR(f.momentumFlux[0], 2.0, 1e-12);
}

TEST(RiemannSolver, Rusanov_IdenticalStates) {
    PrimitiveState W = makePrim(1.0, 1.0, 0.0, 0.0, 1.0, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f = computeRusanovFlux(&W, &W, normal, &fc);

    EXPECT_NEAR(f.massFlux, 1.0, 1e-12);
    EXPECT_NEAR(f.momentumFlux[0], 2.0, 1e-12);
}

TEST(RiemannSolver, HLLC_IdenticalStates) {
    PrimitiveState W = makePrim(1.0, 1.0, 0.0, 0.0, 1.0, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f = computeHLLCFlux(&W, &W, normal, &fc);

    EXPECT_NEAR(f.massFlux, 1.0, 1e-12);
    EXPECT_NEAR(f.momentumFlux[0], 2.0, 1e-12);
}

/* ---- HLLC symmetry: swapping L/R and negating normal gives same mass flux ---- */

TEST(RiemannSolver, HLLC_Symmetry) {
    /* Swapping L/R and negating the normal should negate the mass flux
       (the flux direction reverses with the normal) */
    PrimitiveState L = makePrim(1.0, 0.0, 0.0, 0.0, 1.0, 1.4);
    PrimitiveState R = makePrim(0.125, 0.0, 0.0, 0.0, 0.1, 1.4);
    double nPos[3] = {1.0, 0.0, 0.0};
    double nNeg[3] = {-1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f1 = computeHLLCFlux(&L, &R, nPos, &fc);
    RiemannFlux f2 = computeHLLCFlux(&R, &L, nNeg, &fc);

    EXPECT_NEAR(f1.massFlux, -f2.massFlux, 1e-12);
}

/* ---- Sod shock tube: verify flux direction ---- */

TEST(RiemannSolver, HLLC_SodShock_FluxDirection) {
    /* Left: high pressure, Right: low pressure => flow goes right */
    PrimitiveState L = makePrim(1.0, 0.0, 0.0, 0.0, 1.0, 1.4);
    PrimitiveState R = makePrim(0.125, 0.0, 0.0, 0.0, 0.1, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f = computeHLLCFlux(&L, &R, normal, &fc);

    /* Mass should flow from left to right (positive mass flux) */
    EXPECT_GT(f.massFlux, 0.0);
    /* Momentum flux should be positive (pressure push + mass flow) */
    EXPECT_GT(f.momentumFlux[0], 0.0);
}

/* ---- Stationary contact: zero mass flux ---- */

TEST(RiemannSolver, HLLC_StationaryContact) {
    /* Same pressure and velocity, different density */
    PrimitiveState L = makePrim(1.0, 0.0, 0.0, 0.0, 1.0, 1.4);
    PrimitiveState R = makePrim(0.5, 0.0, 0.0, 0.0, 1.0, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 1, 0, 0};

    RiemannFlux f = computeHLLCFlux(&L, &R, normal, &fc);

    /* At a contact discontinuity with equal pressures and zero velocity,
       there should be no mass flux */
    EXPECT_NEAR(f.massFlux, 0.0, 1e-12);
}

/* ---- 2D test: y-direction flux ---- */

TEST(RiemannSolver, HLLC_YDirection) {
    PrimitiveState L = makePrim(1.0, 0.0, 1.0, 0.0, 1.0, 1.4);
    PrimitiveState R = makePrim(1.0, 0.0, 1.0, 0.0, 1.0, 1.4);
    double normal[3] = {0.0, 1.0, 0.0};
    FluxConfig fc = {2, 1, 0, 0};

    RiemannFlux f = computeHLLCFlux(&L, &R, normal, &fc);

    /* Mass flux = rho * v_n = 1.0 * 1.0 = 1.0 */
    EXPECT_NEAR(f.massFlux, 1.0, 1e-12);
}

/* ---- No-pressure mode (semi-implicit advection) ---- */

TEST(RiemannSolver, HLLC_NoPressure) {
    PrimitiveState L = makePrim(1.0, 1.0, 0.0, 0.0, 100.0, 1.4);
    PrimitiveState R = makePrim(1.0, 1.0, 0.0, 0.0, 100.0, 1.4);
    double normal[3] = {1.0, 0.0, 0.0};
    FluxConfig fc = {1, 0, 0, 0};  /* includePressure = 0 */

    RiemannFlux f = computeHLLCFlux(&L, &R, normal, &fc);

    /* Without pressure: mass flux = rho * u = 1.0 */
    EXPECT_NEAR(f.massFlux, 1.0, 1e-12);
    /* Momentum flux = rho*u*u (no pressure term) */
    EXPECT_NEAR(f.momentumFlux[0], 1.0, 1e-12);
}
