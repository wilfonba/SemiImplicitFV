#include "PressureLaplacian.hpp"
#include <algorithm>

namespace SemiImplicitFV {

double pressureLaplacian(
    const RectilinearMesh& mesh,
    const std::vector<double>& pressure,
    int i, int j, int k,
    double& offDiag)
{
    offDiag = 0.0;
    double diagCoeff = 0.0;
    std::size_t idx = mesh.index(i, j, k);

    // X-direction
    {
        std::size_t xm = mesh.index(i - 1, j, k);
        std::size_t xp = mesh.index(i + 1, j, k);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[xm]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[xp]);
        double dL = 0.5 * (mesh.dx(i - 1) + mesh.dx(i));
        double dR = 0.5 * (mesh.dx(i) + mesh.dx(i + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dx(i));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dx(i));
        offDiag += cL * pressure[xm] + cR * pressure[xp];
        diagCoeff += cL + cR;
    }

    // Y-direction
    if (mesh.dim() >= 2) {
        std::size_t ym = mesh.index(i, j - 1, k);
        std::size_t yp = mesh.index(i, j + 1, k);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[ym]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[yp]);
        double dL = 0.5 * (mesh.dy(j - 1) + mesh.dy(j));
        double dR = 0.5 * (mesh.dy(j) + mesh.dy(j + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dy(j));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dy(j));
        offDiag += cL * pressure[ym] + cR * pressure[yp];
        diagCoeff += cL + cR;
    }

    // Z-direction
    if (mesh.dim() >= 3) {
        std::size_t zm = mesh.index(i, j, k - 1);
        std::size_t zp = mesh.index(i, j, k + 1);
        double rhoL = 0.5 * (mesh.rho[idx] + mesh.rho[zm]);
        double rhoR = 0.5 * (mesh.rho[idx] + mesh.rho[zp]);
        double dL = 0.5 * (mesh.dz(k - 1) + mesh.dz(k));
        double dR = 0.5 * (mesh.dz(k) + mesh.dz(k + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh.dz(k));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh.dz(k));
        offDiag += cL * pressure[zm] + cR * pressure[zp];
        diagCoeff += cL + cR;
    }

    return diagCoeff;
}

} // namespace SemiImplicitFV

