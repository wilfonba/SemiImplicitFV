#include "RKTimeStepping.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace SemiImplicitFV {

double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);

                double dtCell = mesh.dx(i) / std::max(std::abs(state.velU[idx]), 1e-14);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh.dy(j) / std::max(std::abs(state.velV[idx]), 1e-14));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh.dz(k) / std::max(std::abs(state.velW[idx]), 1e-14));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt) {
    double dt = maxDt;
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                std::size_t idx = mesh.index(i, j, k);
                double c = eos.soundSpeed(state.getPrimitiveState(idx));

                double dtCell = mesh.dx(i) / (std::abs(state.velU[idx]) + c);
                if (dim >= 2)
                    dtCell = std::min(dtCell, mesh.dy(j) / (std::abs(state.velV[idx]) + c));
                if (dim >= 3)
                    dtCell = std::min(dtCell, mesh.dz(k) / (std::abs(state.velW[idx]) + c));

                dt = std::min(dt, cfl * dtCell);
            }
        }
    }

    return dt;
}

} // namespace SemiImplicitFV
