#ifndef PRESSURELAPLACIAN_HPP
#define PRESSURELAPLACIAN_HPP

#include "RectilinearMesh.hpp"
#include <vector>

namespace SemiImplicitFV {

double pressureLaplacian(
    const RectilinearMesh& mesh,
    const std::vector<double>& pressure,
    int i, int j, int k,
    double& offDiag);

} // namespace SemiImplicitFV

#endif /* end of include guard PRESSURELAPLACIAN_HPP */
