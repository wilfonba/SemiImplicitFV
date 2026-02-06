#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include <memory>
#include <string>
#include <vector>

namespace SemiImplicitFV {

class PressureSolver {
public:
    virtual ~PressureSolver() = default;

    virtual int solve(
        RectilinearMesh& mesh,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter
    ) = 0;

    virtual std::string name() const = 0;
};

} // namespace SemiImplicitFV

#endif // PRESSURE_SOLVER_HPP
