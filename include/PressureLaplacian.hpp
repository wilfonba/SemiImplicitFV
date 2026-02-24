#ifndef PRESSURE_LAPLACIAN_HPP
#define PRESSURE_LAPLACIAN_HPP

struct RectilinearMesh;

double pressureLaplacian(
    const struct RectilinearMesh* mesh,
    const double* rho,
    const double* pressure,
    int i, int j, int k,
    double* offDiag);

#endif /* PRESSURE_LAPLACIAN_HPP */
