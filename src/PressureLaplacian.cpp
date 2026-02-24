#include "PressureLaplacian.hpp"
#include "RectilinearMesh.hpp"
#include <algorithm>

double pressureLaplacian(
    const struct RectilinearMesh* mesh,
    const double* rho,
    const double* pressure,
    int i, int j, int k,
    double* offDiag)
{
    *offDiag = 0.0;
    double diagCoeff = 0.0;
    size_t idx = mesh_index(mesh, i, j, k);

    /* X-direction */
    {
        size_t xm = mesh_index(mesh, i - 1, j, k);
        size_t xp = mesh_index(mesh, i + 1, j, k);
        double rhoL = 0.5 * (rho[idx] + rho[xm]);
        double rhoR = 0.5 * (rho[idx] + rho[xp]);
        double dL = 0.5 * (mesh_dx(mesh, i - 1) + mesh_dx(mesh, i));
        double dR = 0.5 * (mesh_dx(mesh, i) + mesh_dx(mesh, i + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dx(mesh, i));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dx(mesh, i));
        *offDiag += cL * pressure[xm] + cR * pressure[xp];
        diagCoeff += cL + cR;
    }

    /* Y-direction */
    if (mesh->dim >= 2) {
        size_t ym = mesh_index(mesh, i, j - 1, k);
        size_t yp = mesh_index(mesh, i, j + 1, k);
        double rhoL = 0.5 * (rho[idx] + rho[ym]);
        double rhoR = 0.5 * (rho[idx] + rho[yp]);
        double dL = 0.5 * (mesh_dy(mesh, j - 1) + mesh_dy(mesh, j));
        double dR = 0.5 * (mesh_dy(mesh, j) + mesh_dy(mesh, j + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dy(mesh, j));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dy(mesh, j));
        *offDiag += cL * pressure[ym] + cR * pressure[yp];
        diagCoeff += cL + cR;
    }

    /* Z-direction */
    if (mesh->dim >= 3) {
        size_t zm = mesh_index(mesh, i, j, k - 1);
        size_t zp = mesh_index(mesh, i, j, k + 1);
        double rhoL = 0.5 * (rho[idx] + rho[zm]);
        double rhoR = 0.5 * (rho[idx] + rho[zp]);
        double dL = 0.5 * (mesh_dz(mesh, k - 1) + mesh_dz(mesh, k));
        double dR = 0.5 * (mesh_dz(mesh, k) + mesh_dz(mesh, k + 1));
        double cL = 1.0 / (std::max(rhoL, 1e-14) * dL * mesh_dz(mesh, k));
        double cR = 1.0 / (std::max(rhoR, 1e-14) * dR * mesh_dz(mesh, k));
        *offDiag += cL * pressure[zm] + cR * pressure[zp];
        diagCoeff += cL + cR;
    }

    return diagCoeff;
}
