#ifndef EQUATION_OF_STATE_HPP
#define EQUATION_OF_STATE_HPP

#include "State.hpp"
#include "SimulationConfig.hpp"
#include <string>

namespace SemiImplicitFV {

// Abstract equation of state interface
class EquationOfState {
public:
    explicit EquationOfState(const SimulationConfig& config = {}) : config_(config) {}
    virtual ~EquationOfState() = default;

    const SimulationConfig& config() const { return config_; }
    int dim() const { return config_.dim; }

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

protected:
    SimulationConfig config_;
};

} // namespace SemiImplicitFV

#endif // EQUATION_OF_STATE_HPP

