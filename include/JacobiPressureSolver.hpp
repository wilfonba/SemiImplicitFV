#ifndef JACOBI_PRESSURE_SOLVER_HPP
#define JACOBI_PRESSURE_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include "PressureSolver.hpp"
#include <memory>
#include <vector>

namespace SemiImplicitFV {

class JacobiPressureSolver : public PressureSolver {
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

#ifdef ENABLE_MPI
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
#endif

    std::string name() const override { return "Jacobi"; }
};

} // namespace SemiImplicitFV

#endif // JACOBI_PRESSURE_SOLVER_HPP
