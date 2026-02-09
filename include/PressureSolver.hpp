#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "RectilinearMesh.hpp"
#include <memory>
#include <string>
#include <vector>

#ifdef ENABLE_MPI
namespace SemiImplicitFV { class HaloExchange; }
#endif

namespace SemiImplicitFV {

class PressureSolver {
public:
    virtual ~PressureSolver() = default;

    virtual int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter
    ) = 0;

#ifdef ENABLE_MPI
    virtual int solve(
        const RectilinearMesh& mesh,
        const std::vector<double>& rho,
        const std::vector<double>& rhoc2,
        const std::vector<double>& rhs,
        std::vector<double>& pressure,
        double dt,
        double tolerance,
        int maxIter,
        HaloExchange& halo
    ) = 0;
#endif

    virtual std::string name() const = 0;
};

} // namespace SemiImplicitFV

#endif // PRESSURE_SOLVER_HPP
