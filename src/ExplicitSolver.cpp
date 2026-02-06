#include "ExplicitSolver.hpp"
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

//double ExplicitSolver::step(RectilinearMesh& mesh, double targetDt) {
    //double dt = computeAcousticTimeStep(mesh);
    //if (targetDt > 0) {
        //dt = std::min(dt, targetDt);
    //}

    //mesh.applyBoundaryConditions();

//}

double ExplicitSolver::computeAcousticTimeStep(const RectilinearMesh& mesh) const {
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
                double u = std::sqrt(
                    mesh.velU[idx] * mesh.velU[idx] +
                    mesh.velV[idx] * mesh.velV[idx] +
                    mesh.velW[idx] * mesh.velW[idx]);

                maxSpeed = std::max(maxSpeed, u);
            }
        }
    }

    if (maxSpeed < 1e-14) return params_.maxDt;
    return params_.cfl * minDx / maxSpeed;
}

} // namespace SemiImplicitFV

