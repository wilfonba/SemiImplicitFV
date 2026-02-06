#ifndef PRESSURE_LAPLACIAN_HPP
#define PRESSURE_LAPLACIAN_HPP

#include "RectilinearMesh.hpp"
#include <vector>

namespace SemiImplicitFV {

double pressureLaplacian(
    const RectilinearMesh& mesh,
    const std::vector<double>& pressure,
    int i, int j, int k,
    double& offDiag);

} // namespace SemiImplicitFV

#endif // PRESSURE_LAPLACIAN_HPP

