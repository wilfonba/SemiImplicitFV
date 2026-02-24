#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "MixtureEOS.hpp"
#include "SimulationConfig.hpp"

/* ---- Single-phase limit: mixture EOS should match single-phase ---- */

TEST(MixtureEOS, SinglePhaseLimit_Pressure) {
    /* alpha = [1.0, 0.0], phase 0 = ideal gas with gamma=1.4 */
    PhaseEOS phases[2] = {{1.4, 0.0}, {1.4, 0.0}};
    double alphas[2] = {1.0, 0.0};

    /* rhoE_internal = p / (gamma-1) = 1.0 / 0.4 = 2.5 */
    double rhoE_internal = 2.5;
    double p = mixture_pressure(rhoE_internal, alphas, 2, phases);

    EXPECT_NEAR(p, 1.0, 1e-12);
}

TEST(MixtureEOS, SinglePhaseLimit_Stiffened) {
    /* alpha = [1.0, 0.0], phase 0 = stiffened gas */
    double gamma = 4.4;
    double pInf = 6e8;
    PhaseEOS phases[2] = {{gamma, pInf}, {1.4, 0.0}};
    double alphas[2] = {1.0, 0.0};

    double p_target = 1e9;
    /* rhoE_internal = (p + gamma*pInf)/(gamma-1) */
    double rhoE_internal = (p_target + gamma * pInf) / (gamma - 1.0);
    double p = mixture_pressure(rhoE_internal, alphas, 2, phases);

    EXPECT_NEAR(p, p_target, p_target * 1e-12);
}

/* ---- Effective gamma ---- */

TEST(MixtureEOS, EffectiveGamma_SinglePhase) {
    MultiPhaseParams mp;
    memset(&mp, 0, sizeof(mp));
    mp.nPhases = 2;
    mp.phases[0] = {1.4, 0.0};
    mp.phases[1] = {1.4, 0.0};

    double alphas[2] = {1.0, 0.0};
    double gammaEff, piInfEff;
    effective_gamma_and_pi_inf(alphas, 2, &mp, &gammaEff, &piInfEff);

    EXPECT_NEAR(gammaEff, 1.4, 1e-12);
    EXPECT_NEAR(piInfEff, 0.0, 1e-12);
}

TEST(MixtureEOS, EffectiveGamma_EqualMixture) {
    /* 50/50 mix of two ideal gases with different gammas */
    MultiPhaseParams mp;
    memset(&mp, 0, sizeof(mp));
    mp.nPhases = 2;
    mp.phases[0] = {1.4, 0.0};   /* gamma = 1.4 */
    mp.phases[1] = {1.6, 0.0};   /* gamma = 1.6 */

    double alphas[2] = {0.5, 0.5};
    double gammaEff, piInfEff;
    effective_gamma_and_pi_inf(alphas, 2, &mp, &gammaEff, &piInfEff);

    /* gammaEff = 1 + 1/(0.5/0.4 + 0.5/0.6) = 1 + 1/(1.25 + 0.8333) = 1 + 1/2.0833 */
    double sumInv = 0.5 / 0.4 + 0.5 / 0.6;
    double expected = 1.0 + 1.0 / sumInv;
    EXPECT_NEAR(gammaEff, expected, 1e-12);
    EXPECT_NEAR(piInfEff, 0.0, 1e-12);  /* Both pInf = 0 */
}

/* ---- Sound speed ---- */

TEST(MixtureEOS, SoundSpeed_SinglePhase) {
    PhaseEOS phases[2] = {{1.4, 0.0}, {1.4, 0.0}};
    double alphas[2] = {1.0, 0.0};
    double alphaRhos[2] = {1.0, 0.0};

    double rho = 1.0;
    double p = 1.0;
    double c = mixture_sound_speed(rho, p, alphas, alphaRhos, 2, phases);

    /* Should match single-phase: sqrt(gamma * p / rho) */
    double expected = std::sqrt(1.4 * 1.0 / 1.0);
    EXPECT_NEAR(c, expected, 1e-10);
}

/* ---- Total energy ---- */

TEST(MixtureEOS, TotalEnergy_SinglePhase) {
    PhaseEOS phases[2] = {{1.4, 0.0}, {1.4, 0.0}};
    double alphas[2] = {1.0, 0.0};

    double rho = 1.0;
    double p = 1.0;
    double ke = 0.5;  /* 0.5*rho*u^2 for u=1 */

    double E = mixture_total_energy(rho, p, alphas, 2, ke, phases);

    /* E = ke + p/(gamma-1) = 0.5 + 2.5 = 3.0 */
    EXPECT_NEAR(E, 3.0, 1e-12);
}
