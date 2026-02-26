#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "EquationOfState.hpp"
#include "State.hpp"

/* ---- Ideal Gas Tests ---- */

TEST(EOS_IdealGas, Pressure) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    ConservativeState U;
    memset(&U, 0, sizeof(U));
    U.rho = 1.0;
    U.rhoU[0] = 0.0;
    U.rhoE = 2.5;  /* p = (gamma-1)*rhoE = 0.4*2.5 = 1.0 for stationary gas */

    double p = eos_pressure(&eos, &U);
    EXPECT_NEAR(p, 1.0, 1e-14);
}

TEST(EOS_IdealGas, PressureWithVelocity) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    ConservativeState U;
    memset(&U, 0, sizeof(U));
    U.rho = 1.0;
    U.rhoU[0] = 1.0;  /* u = 1 */
    U.rhoE = 3.0;      /* ke = 0.5*rhoU^2/rho = 0.5, internal = 2.5, p = 0.4*2.5 = 1.0 */

    double p = eos_pressure(&eos, &U);
    EXPECT_NEAR(p, 1.0, 1e-14);
}

TEST(EOS_IdealGas, SoundSpeed) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1.0;
    W.p = 1.0;

    double c = eos_sound_speed(&eos, &W);
    double expected = std::sqrt(1.4 * 1.0 / 1.0);
    EXPECT_NEAR(c, expected, 1e-14);
}

TEST(EOS_IdealGas, Temperature) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1.225;
    W.p = 101325.0;

    double T = eos_temperature(&eos, &W);
    EXPECT_NEAR(T, 101325.0 / (1.225 * 287.0), 1e-10);
}

TEST(EOS_IdealGas, RoundTrip_1D) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1.225;
    W.u[0] = 100.0;
    W.p = 101325.0;

    ConservativeState U = eos_to_conservative(&eos, &W);
    PrimitiveState W2 = eos_to_primitive(&eos, &U);

    EXPECT_NEAR(W2.rho, W.rho, 1e-12);
    EXPECT_NEAR(W2.u[0], W.u[0], 1e-10);
    EXPECT_NEAR(W2.p, W.p, 1e-8);
}

TEST(EOS_IdealGas, RoundTrip_3D) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 3);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 2.0;
    W.u[0] = 50.0;
    W.u[1] = -30.0;
    W.u[2] = 10.0;
    W.p = 200000.0;

    ConservativeState U = eos_to_conservative(&eos, &W);
    PrimitiveState W2 = eos_to_primitive(&eos, &U);

    EXPECT_NEAR(W2.rho, W.rho, 1e-12);
    EXPECT_NEAR(W2.u[0], W.u[0], 1e-10);
    EXPECT_NEAR(W2.u[1], W.u[1], 1e-10);
    EXPECT_NEAR(W2.u[2], W.u[2], 1e-10);
    EXPECT_NEAR(W2.p, W.p, 1e-6);
}

TEST(EOS_IdealGas, TotalEnergy) {
    EOSData eos = eos_create_ideal_gas(1.4, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1.0;
    W.u[0] = 2.0;
    W.p = 1.0;

    double eTotal = eos_total_energy(&eos, &W);
    /* internal = p/(rho*(gamma-1)) = 1.0/(1.0*0.4) = 2.5 */
    /* ke = 0.5*u^2 = 2.0 */
    /* total = 4.5 */
    EXPECT_NEAR(eTotal, 4.5, 1e-14);
}

/* ---- Stiffened Gas Tests ---- */

TEST(EOS_StiffenedGas, Pressure) {
    double gamma = 4.4;
    double pInf = 6e8;
    EOSData eos = eos_create_stiffened_gas(gamma, pInf, 287.0, 1);

    /* For stiffened gas: p = (gamma-1)*e - gamma*pInf */
    /* If we want p = 1e9: e = (p + gamma*pInf)/(gamma-1) */
    double p_target = 1e9;
    double e_internal = (p_target + gamma * pInf) / (gamma - 1.0);

    ConservativeState U;
    memset(&U, 0, sizeof(U));
    U.rho = 1000.0;
    U.rhoU[0] = 0.0;
    U.rhoE = e_internal;

    double p = eos_pressure(&eos, &U);
    EXPECT_NEAR(p, p_target, 1e-2);  /* Tolerance for large pressures */
}

TEST(EOS_StiffenedGas, RoundTrip) {
    double gamma = 4.4;
    double pInf = 6e8;
    EOSData eos = eos_create_stiffened_gas(gamma, pInf, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1000.0;
    W.u[0] = 10.0;
    W.p = 1e9;

    ConservativeState U = eos_to_conservative(&eos, &W);
    PrimitiveState W2 = eos_to_primitive(&eos, &U);

    EXPECT_NEAR(W2.rho, W.rho, 1e-10);
    EXPECT_NEAR(W2.u[0], W.u[0], 1e-10);
    EXPECT_NEAR(W2.p, W.p, W.p * 1e-12);
}

TEST(EOS_StiffenedGas, SoundSpeed) {
    double gamma = 4.4;
    double pInf = 6e8;
    EOSData eos = eos_create_stiffened_gas(gamma, pInf, 287.0, 1);

    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = 1000.0;
    W.p = 1e9;

    double c = eos_sound_speed(&eos, &W);
    double expected = std::sqrt(gamma * (W.p + pInf) / W.rho);
    EXPECT_NEAR(c, expected, 1e-6);
}
