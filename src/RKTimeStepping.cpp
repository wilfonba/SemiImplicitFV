#include "RKTimeStepping.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace SemiImplicitFV {

double sspRKBlendCoeff(const SimulationConfig& config, int stage) {
    // SSP-RK2: U^(n+1) = 1/2 * U^n + 1/2 * [U^(1) + dt*L(U^(1))]
    if (config.RKOrder == 2 && stage == 2) return 0.5;
    // SSP-RK3: stage 2: U^(2) = 3/4 * U^n + 1/4 * [U^(1) + dt*L(U^(1))]
    if (config.RKOrder == 3 && stage == 2) return 3.0 / 4.0;
    // SSP-RK3: stage 3: U^(n+1) = 1/3 * U^n + 2/3 * [U^(2) + dt*L(U^(2))]
    if (config.RKOrder == 3 && stage == 3) return 1.0 / 3.0;
    return 0.0;
}

double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt) {
    double maxSpeed = 0.0;
    double minDx = std::numeric_limits<double>::max();
    int dim = mesh.dim();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                double dxMin = mesh.dx(i);
                if (dim >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (dim >= 3) dxMin = std::min(dxMin, mesh.dz(k));
                minDx = std::min(minDx, dxMin);

                std::size_t idx = mesh.index(i, j, k);
                double speed2 = state.velU[idx] * state.velU[idx];
                if (dim >= 2) speed2 += state.velV[idx] * state.velV[idx];
                if (dim >= 3) speed2 += state.velW[idx] * state.velW[idx];
                double u = std::sqrt(speed2);
                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return maxDt;
    return cfl * minDx / maxSpeed;
}

double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt) {
    double maxSpeed = 0.0;
    double minDx = std::numeric_limits<double>::max();

    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                double dxMin = mesh.dx(i);
                if (mesh.dim() >= 2) dxMin = std::min(dxMin, mesh.dy(j));
                if (mesh.dim() >= 3) dxMin = std::min(dxMin, mesh.dz(k));
                minDx = std::min(minDx, dxMin);

                std::size_t idx = mesh.index(i, j, k);
                double speed2 = state.velU[idx] * state.velU[idx];
                if (mesh.dim() >= 2) speed2 += state.velV[idx] * state.velV[idx];
                if (mesh.dim() >= 3) speed2 += state.velW[idx] * state.velW[idx];
                double u = std::sqrt(speed2);
                double c = eos.soundSpeed(state.getPrimitiveState(idx));
                u += c;

                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return maxDt;
    return cfl * minDx / maxSpeed;
}

} // namespace SemiImplicitFV
