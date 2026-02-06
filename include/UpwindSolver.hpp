#ifndef UPWIND_SOLVER_HPP
#define UPWIND_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

class UpwindSolver : public RiemannSolver {
public:
    explicit UpwindSolver(std::shared_ptr<EquationOfState> eos,
                          bool includePressure = false)
        : RiemannSolver(std::move(eos), includePressure) {}

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

    std::string name() const override {
        return includePressure_ ? "Upwind" : "UpwindAdvective";
    }
};

} // namespace SemiImplicitFV

#endif // UPWIND_SOLVER_HPP
