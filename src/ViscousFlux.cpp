#include "ViscousFlux.hpp"

namespace SemiImplicitFV {

void addViscousFluxes(
    const SimulationConfig& config,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    double mu,
    std::vector<double>& rhsRhoU,
    std::vector<double>& rhsRhoV,
    std::vector<double>& rhsRhoW,
    std::vector<double>& rhsRhoE)
{
    int dim = config.dim;

    // --- X-direction faces ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                // Left cell  = (i-1, j, k)
                // Right cell = (i,   j, k)
                std::size_t idxL = mesh.index(i - 1, j, k);
                std::size_t idxR = mesh.index(i, j, k);

                double dxi = 0.5 * (mesh.dx(i - 1) + mesh.dx(i));

                // Normal velocity gradients: du/dx, dv/dx, dw/dx
                double dudx = (state.velU[idxR] - state.velU[idxL]) / dxi;
                double dvdx = 0.0;
                double dwdx = 0.0;
                if (dim >= 2) dvdx = (state.velV[idxR] - state.velV[idxL]) / dxi;
                if (dim >= 3) dwdx = (state.velW[idxR] - state.velW[idxL]) / dxi;

                // Transverse gradients averaged from both cells
                double dudy = 0.0, dvdy = 0.0;
                if (dim >= 2) {
                    double dyj = mesh.dy(j);
                    std::size_t Ljm = mesh.index(i - 1, j - 1, k);
                    std::size_t Ljp = mesh.index(i - 1, j + 1, k);
                    std::size_t Rjm = mesh.index(i, j - 1, k);
                    std::size_t Rjp = mesh.index(i, j + 1, k);

                    double dudy_L = (state.velU[Ljp] - state.velU[Ljm]) / (2.0 * dyj);
                    double dudy_R = (state.velU[Rjp] - state.velU[Rjm]) / (2.0 * dyj);
                    dudy = 0.5 * (dudy_L + dudy_R);

                    double dvdy_L = (state.velV[Ljp] - state.velV[Ljm]) / (2.0 * dyj);
                    double dvdy_R = (state.velV[Rjp] - state.velV[Rjm]) / (2.0 * dyj);
                    dvdy = 0.5 * (dvdy_L + dvdy_R);
                }

                double dudz = 0.0, dwdz = 0.0;
                if (dim >= 3) {
                    double dzk = mesh.dz(k);
                    std::size_t Lkm = mesh.index(i - 1, j, k - 1);
                    std::size_t Lkp = mesh.index(i - 1, j, k + 1);
                    std::size_t Rkm = mesh.index(i, j, k - 1);
                    std::size_t Rkp = mesh.index(i, j, k + 1);

                    double dudz_L = (state.velU[Lkp] - state.velU[Lkm]) / (2.0 * dzk);
                    double dudz_R = (state.velU[Rkp] - state.velU[Rkm]) / (2.0 * dzk);
                    dudz = 0.5 * (dudz_L + dudz_R);

                    double dwdz_L = (state.velW[Lkp] - state.velW[Lkm]) / (2.0 * dzk);
                    double dwdz_R = (state.velW[Rkp] - state.velW[Rkm]) / (2.0 * dzk);
                    dwdz = 0.5 * (dwdz_L + dwdz_R);
                }

                double divU = dudx + dvdy + dwdz;

                // Viscous stress components (x-face normal = x)
                double tau_xx = mu * (2.0 * dudx - (2.0 / 3.0) * divU);
                double tau_xy = mu * (dvdx + dudy);
                double tau_xz = mu * (dwdx + dudz);

                // Face velocity (simple average)
                double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                double vFace = 0.0;
                double wFace = 0.0;
                if (dim >= 2) vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                if (dim >= 3) wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                // Viscous work: tau_x . u
                double work = tau_xx * uFace + tau_xy * vFace + tau_xz * wFace;

                double area = mesh.faceAreaX(j, k);

                // Accumulate: viscous flux has opposite sign convention to inviscid
                if (i >= 1) {
                    double coeff = area / mesh.cellVolume(i - 1, j, k);
                    rhsRhoU[idxL] += coeff * tau_xx;
                    if (dim >= 2) rhsRhoV[idxL] += coeff * tau_xy;
                    if (dim >= 3) rhsRhoW[idxL] += coeff * tau_xz;
                    rhsRhoE[idxL] += coeff * work;
                }

                if (i < mesh.nx()) {
                    double coeff = area / mesh.cellVolume(i, j, k);
                    rhsRhoU[idxR] -= coeff * tau_xx;
                    if (dim >= 2) rhsRhoV[idxR] -= coeff * tau_xy;
                    if (dim >= 3) rhsRhoW[idxR] -= coeff * tau_xz;
                    rhsRhoE[idxR] -= coeff * work;
                }
            }
        }
    }

    // --- Y-direction faces ---
    if (dim >= 2) {
        for (int k = 0; k < mesh.nz(); ++k) {
            for (int j = 0; j <= mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j - 1, k);
                    std::size_t idxR = mesh.index(i, j, k);

                    double dyj = 0.5 * (mesh.dy(j - 1) + mesh.dy(j));

                    // Normal gradients: du/dy, dv/dy, dw/dy
                    double dudy = (state.velU[idxR] - state.velU[idxL]) / dyj;
                    double dvdy = (state.velV[idxR] - state.velV[idxL]) / dyj;
                    double dwdy = 0.0;
                    if (dim >= 3) dwdy = (state.velW[idxR] - state.velW[idxL]) / dyj;

                    // Transverse x-gradients
                    double dxi = mesh.dx(i);
                    std::size_t Lim = mesh.index(i - 1, j - 1, k);
                    std::size_t Lip = mesh.index(i + 1, j - 1, k);
                    std::size_t Rim = mesh.index(i - 1, j, k);
                    std::size_t Rip = mesh.index(i + 1, j, k);

                    double dudx_L = (state.velU[Lip] - state.velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (state.velU[Rip] - state.velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);

                    double dvdx_L = (state.velV[Lip] - state.velV[Lim]) / (2.0 * dxi);
                    double dvdx_R = (state.velV[Rip] - state.velV[Rim]) / (2.0 * dxi);
                    double dvdx = 0.5 * (dvdx_L + dvdx_R);

                    // Transverse z-gradients
                    double dvdz = 0.0, dwdz = 0.0;
                    if (dim >= 3) {
                        double dzk = mesh.dz(k);
                        std::size_t Lkm = mesh.index(i, j - 1, k - 1);
                        std::size_t Lkp = mesh.index(i, j - 1, k + 1);
                        std::size_t Rkm = mesh.index(i, j, k - 1);
                        std::size_t Rkp = mesh.index(i, j, k + 1);

                        double dvdz_L = (state.velV[Lkp] - state.velV[Lkm]) / (2.0 * dzk);
                        double dvdz_R = (state.velV[Rkp] - state.velV[Rkm]) / (2.0 * dzk);
                        dvdz = 0.5 * (dvdz_L + dvdz_R);

                        double dwdz_L = (state.velW[Lkp] - state.velW[Lkm]) / (2.0 * dzk);
                        double dwdz_R = (state.velW[Rkp] - state.velW[Rkm]) / (2.0 * dzk);
                        dwdz = 0.5 * (dwdz_L + dwdz_R);
                    }

                    double divU = dudx + dvdy + dwdz;

                    // Viscous stress components (y-face normal = y)
                    double tau_yx = mu * (dudy + dvdx);
                    double tau_yy = mu * (2.0 * dvdy - (2.0 / 3.0) * divU);
                    double tau_yz = mu * (dwdy + dvdz);

                    // Face velocity
                    double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                    double vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                    double wFace = 0.0;
                    if (dim >= 3) wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                    double work = tau_yx * uFace + tau_yy * vFace + tau_yz * wFace;

                    double area = mesh.faceAreaY(i, k);

                    if (j >= 1) {
                        double coeff = area / mesh.cellVolume(i, j - 1, k);
                        rhsRhoU[idxL] += coeff * tau_yx;
                        rhsRhoV[idxL] += coeff * tau_yy;
                        if (dim >= 3) rhsRhoW[idxL] += coeff * tau_yz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (j < mesh.ny()) {
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRhoU[idxR] -= coeff * tau_yx;
                        rhsRhoV[idxR] -= coeff * tau_yy;
                        if (dim >= 3) rhsRhoW[idxR] -= coeff * tau_yz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }

    // --- Z-direction faces ---
    if (dim >= 3) {
        for (int k = 0; k <= mesh.nz(); ++k) {
            for (int j = 0; j < mesh.ny(); ++j) {
                for (int i = 0; i < mesh.nx(); ++i) {
                    std::size_t idxL = mesh.index(i, j, k - 1);
                    std::size_t idxR = mesh.index(i, j, k);

                    double dzk = 0.5 * (mesh.dz(k - 1) + mesh.dz(k));

                    // Normal gradients: du/dz, dv/dz, dw/dz
                    double dudz = (state.velU[idxR] - state.velU[idxL]) / dzk;
                    double dvdz = (state.velV[idxR] - state.velV[idxL]) / dzk;
                    double dwdz = (state.velW[idxR] - state.velW[idxL]) / dzk;

                    // Transverse x-gradients
                    double dxi = mesh.dx(i);
                    std::size_t Lim = mesh.index(i - 1, j, k - 1);
                    std::size_t Lip = mesh.index(i + 1, j, k - 1);
                    std::size_t Rim = mesh.index(i - 1, j, k);
                    std::size_t Rip = mesh.index(i + 1, j, k);

                    double dudx_L = (state.velU[Lip] - state.velU[Lim]) / (2.0 * dxi);
                    double dudx_R = (state.velU[Rip] - state.velU[Rim]) / (2.0 * dxi);
                    double dudx = 0.5 * (dudx_L + dudx_R);

                    double dwdx_L = (state.velW[Lip] - state.velW[Lim]) / (2.0 * dxi);
                    double dwdx_R = (state.velW[Rip] - state.velW[Rim]) / (2.0 * dxi);
                    double dwdx = 0.5 * (dwdx_L + dwdx_R);

                    // Transverse y-gradients
                    double dyj = mesh.dy(j);
                    std::size_t Ljm = mesh.index(i, j - 1, k - 1);
                    std::size_t Ljp = mesh.index(i, j + 1, k - 1);
                    std::size_t Rjm = mesh.index(i, j - 1, k);
                    std::size_t Rjp = mesh.index(i, j + 1, k);

                    double dvdy_L = (state.velV[Ljp] - state.velV[Ljm]) / (2.0 * dyj);
                    double dvdy_R = (state.velV[Rjp] - state.velV[Rjm]) / (2.0 * dyj);
                    double dvdy = 0.5 * (dvdy_L + dvdy_R);

                    double dwdy_L = (state.velW[Ljp] - state.velW[Ljm]) / (2.0 * dyj);
                    double dwdy_R = (state.velW[Rjp] - state.velW[Rjm]) / (2.0 * dyj);
                    double dwdy = 0.5 * (dwdy_L + dwdy_R);

                    double divU = dudx + dvdy + dwdz;

                    // Viscous stress components (z-face normal = z)
                    double tau_zx = mu * (dudz + dwdx);
                    double tau_zy = mu * (dvdz + dwdy);
                    double tau_zz = mu * (2.0 * dwdz - (2.0 / 3.0) * divU);

                    // Face velocity
                    double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                    double vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                    double wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                    double work = tau_zx * uFace + tau_zy * vFace + tau_zz * wFace;

                    double area = mesh.faceAreaZ(i, j);

                    if (k >= 1) {
                        double coeff = area / mesh.cellVolume(i, j, k - 1);
                        rhsRhoU[idxL] += coeff * tau_zx;
                        rhsRhoV[idxL] += coeff * tau_zy;
                        rhsRhoW[idxL] += coeff * tau_zz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (k < mesh.nz()) {
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRhoU[idxR] -= coeff * tau_zx;
                        rhsRhoV[idxR] -= coeff * tau_zy;
                        rhsRhoW[idxR] -= coeff * tau_zz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }
}

} // namespace SemiImplicitFV
