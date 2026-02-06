#ifndef EQUATION_OF_STATE_HPP
#define EQUATION_OF_STATE_HPP

#include "State.hpp"
#include <string>

namespace SemiImplicitFV {

// Abstract equation of state interface
class EquationOfStateBase {
public:
    virtual ~EquationOfStateBase() = default;

    // Compute pressure from conservative state
    virtual double pressure(const ConservativeState& U) const = 0;

    // Compute temperature from primitive state
    virtual double temperature(const PrimitiveState& W) const = 0;

    // Compute sound speed
    virtual double soundSpeed(const PrimitiveState& W) const = 0;

    // Compute specific internal energy
    virtual double internalEnergy(const PrimitiveState& W) const = 0;

    // Compute total energy from primitives
    virtual double totalEnergy(const PrimitiveState& W) const = 0;

    // Convert conservative to primitive
    virtual PrimitiveState toPrimitive(const ConservativeState& U) const = 0;

    // Convert primitive to conservative
    virtual ConservativeState toConservative(const PrimitiveState& W) const = 0;

    virtual std::string name() const = 0;
};

// Ideal gas equation of state
class IdealGasEOS : public EquationOfStateBase {
public:
    explicit IdealGasEOS(double gamma = 1.4, double R = 287.0);

    double pressure(const ConservativeState& U) const override;
    double temperature(const PrimitiveState& W) const override;
    double soundSpeed(const PrimitiveState& W) const override;
    double internalEnergy(const PrimitiveState& W) const override;
    double totalEnergy(const PrimitiveState& W) const override;

    PrimitiveState toPrimitive(const ConservativeState& U) const override;
    ConservativeState toConservative(const PrimitiveState& W) const override;

    std::string name() const override { return "IdealGas"; }

    // Accessors
    double gamma() const { return gamma_; }
    double R() const { return R_; }

private:
    double gamma_;  // Ratio of specific heats
    double R_;      // Specific gas constant
};

// Stiffened gas EOS (useful for liquids/multi-phase)
class StiffenedGasEOS : public EquationOfStateBase {
public:
    StiffenedGasEOS(double gamma, double pInf, double R);

    double pressure(const ConservativeState& U) const override;
    double temperature(const PrimitiveState& W) const override;
    double soundSpeed(const PrimitiveState& W) const override;
    double internalEnergy(const PrimitiveState& W) const override;
    double totalEnergy(const PrimitiveState& W) const override;

    PrimitiveState toPrimitive(const ConservativeState& U) const override;
    ConservativeState toConservative(const PrimitiveState& W) const override;

    std::string name() const override { return "StiffenedGas"; }

private:
    double gamma_;
    double pInf_;   // Stiffness pressure
    double R_;
};

} // namespace SemiImplicitFV

#endif // EQUATION_OF_STATE_HPP
