#ifndef HLLC_SOLVER_HPP
#define HLLC_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

// When includePressure is false, all eigenvalues are u so this reduces to a simpler form
class HLLCSolver : public RiemannSolver {
public:
    explicit HLLCSolver(std::shared_ptr<EquationOfState> eos,
                        bool includePressure = false,
                        const SimulationConfig& config = {})
        : RiemannSolver(std::move(eos), includePressure, config) {}

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
        return includePressure_ ? "HLLC" : "HLLCAdvective";
    }
};

} // namespace SemiImplicitFV

#endif // HLLC_SOLVER_HPP

