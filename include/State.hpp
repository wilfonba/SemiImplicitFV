#ifndef STATE_HPP
#define STATE_HPP

#include <array>

namespace SemiImplicitFV {

// Conservative variables (per-cell bundle for EOS conversions)
struct ConservativeState {
    double rho;                    // Density
    std::array<double, 3> rhoU;    // Momentum components
    double rhoE;                   // Total energy

    ConservativeState() : rho(0.0), rhoU{0.0}, rhoE(0.0) {}
};

// Primitive variables (per-cell bundle for EOS conversions)
struct PrimitiveState {
    double rho;                    // Density
    std::array<double, 3> u;       // Velocity components
    double p;                      // Physical pressure
    double T;                      // Temperature
    double sigma;                  // Entropic pressure (IGR)
    double gammaEff;               // Effective gamma for multi-phase (0 = single-phase)
    double piInfEff;               // Effective piInf for multi-phase stiffened gas (0 = ideal gas)
    double alpha[8];               // Reconstructed volume fractions at face (multi-phase)

    PrimitiveState() : rho(0.0), u{0.0}, p(0.0), T(0.0), sigma(0.0), gammaEff(0.0), piInfEff(0.0), alpha{} {}
};

} // namespace SemiImplicitFV

#endif // STATE_HPP

