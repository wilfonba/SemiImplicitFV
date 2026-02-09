#ifndef HLLC_SOLVER_HPP
#define HLLC_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

class HLLCSolver : public RiemannSolver {
public:
    explicit HLLCSolver(std::shared_ptr<EquationOfState> eos,
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

    std::string name() const override { return "HLLC"; }
};

} // namespace SemiImplicitFV

#endif // HLLC_SOLVER_HPP

