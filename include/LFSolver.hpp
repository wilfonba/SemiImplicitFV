#ifndef LF_SOLVER_HPP
#define LF_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

class LFSolver : public RiemannSolver {
public:
    explicit LFSolver(std::shared_ptr<EquationOfState> eos,
                          const SimulationConfig& config = {})
        : RiemannSolver(std::move(eos), config) {}

    RiemannFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override { return "Lax-Friedrichs"; }
};

} // namespace SemiImplicitFV

#endif // LF_SOLVER_HPP

