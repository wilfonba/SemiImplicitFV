#include "SurfaceTension.hpp"
#include <algorithm>
#include <cmath>

namespace SemiImplicitFV {

void addSurfaceTensionFluxes(
    const SimulationConfig& config,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    double sigma,
    std::vector<double>& rhsRhoU,
    std::vector<double>& rhsRhoV,
    std::vector<double>& rhsRhoW,
    std::vector<double>& rhsRhoE)
{
    int dim = config.dim;
    const double eps = config.surfaceTensionParams.epsGradAlpha;
    const auto& alpha = state.alpha[0];

    // --- X-direction faces ---
    for (int k = 0; k < mesh.nz(); ++k) {
        for (int j = 0; j < mesh.ny(); ++j) {
            for (int i = 0; i <= mesh.nx(); ++i) {
                std::size_t idxL = mesh.index(i - 1, j, k);
                std::size_t idxR = mesh.index(i, j, k);

                double dxi = 0.5 * (mesh.dx(i - 1) + mesh.dx(i));

                // Normal gradient: d(alpha)/dx
                double dadx = (alpha[idxR] - alpha[idxL]) / dxi;

                // Transverse gradients
                double dady = 0.0;
                if (dim >= 2) {
                    double dyj = mesh.dy(j);
                    std::size_t Ljm = mesh.index(i - 1, j - 1, k);
                    std::size_t Ljp = mesh.index(i - 1, j + 1, k);
                    std::size_t Rjm = mesh.index(i, j - 1, k);
                    std::size_t Rjp = mesh.index(i, j + 1, k);

                    double dady_L = (alpha[Ljp] - alpha[Ljm]) / (2.0 * dyj);
                    double dady_R = (alpha[Rjp] - alpha[Rjm]) / (2.0 * dyj);
                    dady = 0.5 * (dady_L + dady_R);
                }

                double dadz = 0.0;
                if (dim >= 3) {
                    double dzk = mesh.dz(k);
                    std::size_t Lkm = mesh.index(i - 1, j, k - 1);
                    std::size_t Lkp = mesh.index(i - 1, j, k + 1);
                    std::size_t Rkm = mesh.index(i, j, k - 1);
                    std::size_t Rkp = mesh.index(i, j, k + 1);

                    double dadz_L = (alpha[Lkp] - alpha[Lkm]) / (2.0 * dzk);
                    double dadz_R = (alpha[Rkp] - alpha[Rkm]) / (2.0 * dzk);
                    dadz = 0.5 * (dadz_L + dadz_R);
                }

                double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                double normGradReg = std::max(normGrad, eps);

                // Capillary tensor components for x-face (normal = x):
                // T_xx = sigma * (dady^2 + dadz^2) / |grad(alpha)|
                // T_xy = -sigma * dadx * dady / |grad(alpha)|
                // T_xz = -sigma * dadx * dadz / |grad(alpha)|
                double T_xx = sigma * (dady * dady + dadz * dadz) / normGradReg;
                double T_xy = -sigma * dadx * dady / normGradReg;
                double T_xz = -sigma * dadx * dadz / normGradReg;

                // Face velocity (simple average)
                double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                double vFace = 0.0;
                double wFace = 0.0;
                if (dim >= 2) vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                if (dim >= 3) wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                // Capillary work: T_cap . u
                double work = T_xx * uFace + T_xy * vFace + T_xz * wFace;

                double area = mesh.faceAreaX(j, k);

                if (i >= 1) {
                    double coeff = area / mesh.cellVolume(i - 1, j, k);
                    rhsRhoU[idxL] += coeff * T_xx;
                    if (dim >= 2) rhsRhoV[idxL] += coeff * T_xy;
                    if (dim >= 3) rhsRhoW[idxL] += coeff * T_xz;
                    rhsRhoE[idxL] += coeff * work;
                }

                if (i < mesh.nx()) {
                    double coeff = area / mesh.cellVolume(i, j, k);
                    rhsRhoU[idxR] -= coeff * T_xx;
                    if (dim >= 2) rhsRhoV[idxR] -= coeff * T_xy;
                    if (dim >= 3) rhsRhoW[idxR] -= coeff * T_xz;
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

                    // Normal gradient: d(alpha)/dy
                    double dady = (alpha[idxR] - alpha[idxL]) / dyj;

                    // Transverse x-gradient
                    double dxi = mesh.dx(i);
                    std::size_t Lim = mesh.index(i - 1, j - 1, k);
                    std::size_t Lip = mesh.index(i + 1, j - 1, k);
                    std::size_t Rim = mesh.index(i - 1, j, k);
                    std::size_t Rip = mesh.index(i + 1, j, k);

                    double dadx_L = (alpha[Lip] - alpha[Lim]) / (2.0 * dxi);
                    double dadx_R = (alpha[Rip] - alpha[Rim]) / (2.0 * dxi);
                    double dadx = 0.5 * (dadx_L + dadx_R);

                    // Transverse z-gradient
                    double dadz = 0.0;
                    if (dim >= 3) {
                        double dzk = mesh.dz(k);
                        std::size_t Lkm = mesh.index(i, j - 1, k - 1);
                        std::size_t Lkp = mesh.index(i, j - 1, k + 1);
                        std::size_t Rkm = mesh.index(i, j, k - 1);
                        std::size_t Rkp = mesh.index(i, j, k + 1);

                        double dadz_L = (alpha[Lkp] - alpha[Lkm]) / (2.0 * dzk);
                        double dadz_R = (alpha[Rkp] - alpha[Rkm]) / (2.0 * dzk);
                        dadz = 0.5 * (dadz_L + dadz_R);
                    }

                    double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                    double normGradReg = std::max(normGrad, eps);

                    // Capillary tensor components for y-face (normal = y):
                    // T_yx = -sigma * dady * dadx / |grad(alpha)|
                    // T_yy = sigma * (dadx^2 + dadz^2) / |grad(alpha)|
                    // T_yz = -sigma * dady * dadz / |grad(alpha)|
                    double T_yx = -sigma * dady * dadx / normGradReg;
                    double T_yy = sigma * (dadx * dadx + dadz * dadz) / normGradReg;
                    double T_yz = -sigma * dady * dadz / normGradReg;

                    // Face velocity
                    double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                    double vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                    double wFace = 0.0;
                    if (dim >= 3) wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                    double work = T_yx * uFace + T_yy * vFace + T_yz * wFace;

                    double area = mesh.faceAreaY(i, k);

                    if (j >= 1) {
                        double coeff = area / mesh.cellVolume(i, j - 1, k);
                        rhsRhoU[idxL] += coeff * T_yx;
                        rhsRhoV[idxL] += coeff * T_yy;
                        if (dim >= 3) rhsRhoW[idxL] += coeff * T_yz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (j < mesh.ny()) {
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRhoU[idxR] -= coeff * T_yx;
                        rhsRhoV[idxR] -= coeff * T_yy;
                        if (dim >= 3) rhsRhoW[idxR] -= coeff * T_yz;
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

                    // Normal gradient: d(alpha)/dz
                    double dadz = (alpha[idxR] - alpha[idxL]) / dzk;

                    // Transverse x-gradient
                    double dxi = mesh.dx(i);
                    std::size_t Lim = mesh.index(i - 1, j, k - 1);
                    std::size_t Lip = mesh.index(i + 1, j, k - 1);
                    std::size_t Rim = mesh.index(i - 1, j, k);
                    std::size_t Rip = mesh.index(i + 1, j, k);

                    double dadx_L = (alpha[Lip] - alpha[Lim]) / (2.0 * dxi);
                    double dadx_R = (alpha[Rip] - alpha[Rim]) / (2.0 * dxi);
                    double dadx = 0.5 * (dadx_L + dadx_R);

                    // Transverse y-gradient
                    double dyj = mesh.dy(j);
                    std::size_t Ljm = mesh.index(i, j - 1, k - 1);
                    std::size_t Ljp = mesh.index(i, j + 1, k - 1);
                    std::size_t Rjm = mesh.index(i, j - 1, k);
                    std::size_t Rjp = mesh.index(i, j + 1, k);

                    double dady_L = (alpha[Ljp] - alpha[Ljm]) / (2.0 * dyj);
                    double dady_R = (alpha[Rjp] - alpha[Rjm]) / (2.0 * dyj);
                    double dady = 0.5 * (dady_L + dady_R);

                    double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                    double normGradReg = std::max(normGrad, eps);

                    // Capillary tensor components for z-face (normal = z):
                    // T_zx = -sigma * dadz * dadx / |grad(alpha)|
                    // T_zy = -sigma * dadz * dady / |grad(alpha)|
                    // T_zz = sigma * (dadx^2 + dady^2) / |grad(alpha)|
                    double T_zx = -sigma * dadz * dadx / normGradReg;
                    double T_zy = -sigma * dadz * dady / normGradReg;
                    double T_zz = sigma * (dadx * dadx + dady * dady) / normGradReg;

                    // Face velocity
                    double uFace = 0.5 * (state.velU[idxL] + state.velU[idxR]);
                    double vFace = 0.5 * (state.velV[idxL] + state.velV[idxR]);
                    double wFace = 0.5 * (state.velW[idxL] + state.velW[idxR]);

                    double work = T_zx * uFace + T_zy * vFace + T_zz * wFace;

                    double area = mesh.faceAreaZ(i, j);

                    if (k >= 1) {
                        double coeff = area / mesh.cellVolume(i, j, k - 1);
                        rhsRhoU[idxL] += coeff * T_zx;
                        rhsRhoV[idxL] += coeff * T_zy;
                        rhsRhoW[idxL] += coeff * T_zz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (k < mesh.nz()) {
                        double coeff = area / mesh.cellVolume(i, j, k);
                        rhsRhoU[idxR] -= coeff * T_zx;
                        rhsRhoV[idxR] -= coeff * T_zy;
                        rhsRhoW[idxR] -= coeff * T_zz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }
}

} // namespace SemiImplicitFV
