#include "ExplicitSolver.hpp"
#include "VariablesConversion.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace SemiImplicitFV {

ExplicitSolver::ExplicitSolver(
    std::shared_ptr<RiemannSolver> riemannSolver,
    std::shared_ptr<EquationOfState> eos,
    std::shared_ptr<IGRSolver> igrSolver,
    const ExplicitParams& params
)
    : riemannSolver_(std::move(riemannSolver))
    , eos_(std::move(eos))
    , igrSolver_(std::move(igrSolver))
    , params_(params)
{}

double ExplicitSolver::step(const RectilinearMesh& mesh, SolutionState& state, double targetDt) {
    double dt = computeAcousticTimeStep(mesh, state);
    if (targetDt > 0) {
        dt = std::min(dt, targetDt);
    }

    for (int stage = 0; stage < params_.RKOrder; ++stage) {
        computeRHS(mesh, state);
    }

    return dt;
}

void ExplicitSolver::computeRHS(const RectilinearMesh& mesh, SolutionState& state) {

    convertConservativeToPrimitiveVariables(mesh, state, eos_);
    mesh.applyBoundaryConditions(state);

    // Loop over all cells and compute fluxes using the Riemann solver
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i < mesh.nx(); ++i) {
                // Placeholder for flux computations
            }
        }
    }
}

double ExplicitSolver::computeAcousticTimeStep(const RectilinearMesh& mesh, const SolutionState& state) const {
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

                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return params_.maxDt;
    return params_.cfl * minDx / maxSpeed;
}

} // namespace SemiImplicitFV
