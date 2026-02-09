#include "GaussSeidelPressureSolver.hpp"
#include "PressureLaplacian.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace SemiImplicitFV {

int GaussSeidelPressureSolver::solve(
    const RectilinearMesh& mesh,
    const std::vector<double>& rho,
    const std::vector<double>& rhoc2,
    const std::vector<double>& rhs,
    std::vector<double>& pressure,
    double dt,
    double tolerance,
    int maxIter
) {
    double dt2 = dt * dt;
    double maxResidual = 0.0;

    for (int iter = 0; iter < maxIter; ++iter) {
        mesh.fillScalarGhosts(pressure);
        maxResidual = 0.0;

        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idx = mesh.index(i, j, k);
                    double coeff = rhoc2[idx] * dt2;

                    double offDiag;
                    double diagL = pressureLaplacian(mesh, rho, pressure, i, j, k, offDiag);

                    double denom = 1.0 + coeff * diagL;
                    double pOld = pressure[idx];
                    pressure[idx] = (rhs[idx] + coeff * offDiag) / denom;

                    double residual = std::abs(pressure[idx] - pOld);
                    maxResidual = std::max(maxResidual, residual);
                }
            }
        }

        if (maxResidual < tolerance) {
            return iter + 1;
        }
    }
    std::cout << maxResidual << std::endl;
    return maxIter;
}

} // namespace SemiImplicitFV
