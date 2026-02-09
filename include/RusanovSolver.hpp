#ifndef RUSANOV_SOLVER_HPP
#define RUSANOV_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

class RusanovSolver : public RiemannSolver {
public:
    explicit RusanovSolver(std::shared_ptr<EquationOfState> eos,
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

    std::string name() const override { return "Rusanov"; }
};

} // namespace SemiImplicitFV

#endif // RUSANOV_SOLVER_HPP

