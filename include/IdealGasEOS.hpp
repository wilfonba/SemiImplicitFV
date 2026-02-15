#ifndef IDEAL_GAS_EOS_HPP
#define IDEAL_GAS_EOS_HPP

#include "EquationOfState.hpp"
#include "State.hpp"

namespace SemiImplicitFV {

class IdealGasEOS : public EquationOfState {
public:
    explicit IdealGasEOS(double gamma = 1.4, double R = 287.0,
                         const SimulationConfig& config = {});

    double pressure(const ConservativeState& U) const override;
    double temperature(const PrimitiveState& W) const override;
    double soundSpeed(const PrimitiveState& W) const override;
    double internalEnergy(const PrimitiveState& W) const override;
    double totalEnergy(const PrimitiveState& W) const override;

    PrimitiveState toPrimitive(const ConservativeState& U) const override;
    ConservativeState toConservative(const PrimitiveState& W) const override;

    std::string name() const override { return "IdealGas"; }

    // Accessors
    double gamma() const override { return gamma_; }
    double pInf() const override { return 0.0; }
    double R() const { return R_; }

private:
    double gamma_;  // Ratio of specific heats
    double R_;      // Specific gas constant
};

} // namespace SemiImplicitFV

#endif // IDEAL_GAS_EOS_HPP

