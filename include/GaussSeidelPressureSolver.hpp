#ifndef GAUSS_SEIDEL_PRESSURE_SOLVER_HPP
#define GAUSS_SEIDEL_PRESSURE_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "PressureSolver.hpp"
#include <memory>
#include <vector>

namespace SemiImplicitFV {

class GaussSeidelPressureSolver : public PressureSolver {
public:
    int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter
    ) override;

    int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter,
        HaloExchange& halo
    ) override;

    std::string name() const override { return "GaussSeidel"; }
};

} // namespace SemiImplicitFV

#endif // GAUSS_SEIDEL_PRESSURE_SOLVER_HPP
