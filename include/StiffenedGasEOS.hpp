#ifndef STIFFENED_GAS_EOS_HPP
#define STIFFENED_GAS_EOS_HPP

#include "EquationOfState.hpp"
#include "State.hpp"

namespace SemiImplicitFV {

class StiffenedGasEOS : public EquationOfState {
public:
    StiffenedGasEOS(double gamma, double pInf, double R,
                    const SimulationConfig& config = {});

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

#endif // STIFFENED_GAS_EOS_HPP

