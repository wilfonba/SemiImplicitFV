#include "JacobiPressureSolver.hpp"
#include "PressureLaplacian.hpp"
#include "HaloExchange.hpp"
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

int JacobiPressureSolver::solve(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter
) {
    std::vector<double> pNew(pressure.size(), 0.0);
    double dt2 = dt * dt;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh.fillScalarGhosts(pressure);
        double maxResidual = 0.0;

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, offDiag);

                    double denom = 1.0 + coeff * diagL;
                    pNew[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = std::abs(pNew[idx] - pressure[idx]);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        for (int k = 0; k < mesh.nz(); ++k)
            for (int j = 0; j < mesh.ny(); ++j)
                for (int i = 0; i < mesh.nx(); ++i)
                    pressure[mesh.index(i, j, k)] = pNew[mesh.index(i, j, k)];

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

int JacobiPressureSolver::solve(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter,
    HaloExchange& halo
) {
    std::vector<double> pNew(pressure.size(), 0.0);
    double dt2 = dt * dt;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh.fillScalarGhosts(pressure, halo);
        double maxResidual = 0.0;

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, offDiag);

                    double denom = 1.0 + coeff * diagL;
                    pNew[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = std::abs(pNew[idx] - pressure[idx]);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        for (int k = 0; k < mesh.nz(); ++k)
            for (int j = 0; j < mesh.ny(); ++j)
                for (int i = 0; i < mesh.nx(); ++i)
                    pressure[mesh.index(i, j, k)] = pNew[mesh.index(i, j, k)];

        // Global residual reduction
        MPI_Allreduce(MPI_IN_PLACE, &maxResidual, 1, MPI_DOUBLE, MPI_MAX,
                      halo.mpi().comm());

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }

    return maxIter;
}

} // namespace SemiImplicitFV
