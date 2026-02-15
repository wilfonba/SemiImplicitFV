#ifndef HLLC_SOLVER_HPP
#define HLLC_SOLVER_HPP

#include "RiemannSolver.hpp"

namespace SemiImplicitFV {

class HLLCSolver : public RiemannSolver {
public:
    explicit HLLCSolver(std::shared_ptr<EquationOfState> eos,
                        const SimulationConfig& config = {})
        : RiemannSolver(std::move(eos), config) {}

    std::string name() const override { return "HLLC"; }
};

} // namespace SemiImplicitFV

#endif // HLLC_SOLVER_HPP

