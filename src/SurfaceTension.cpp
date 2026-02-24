#include "SurfaceTension.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <algorithm>
#include <cmath>

void add_surface_tension_fluxes(
    const SimulationConfig* config,
    const RectilinearMesh* mesh,
    const SolutionState* state,
    double sigma,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE)
{
    int dim = config->dim;
    const double eps = config->surfaceTensionParams.epsGradAlpha;
    /* alpha[0] is the first phase volume fraction: state->alpha + 0*totalCells */
    const double* alpha = state->alpha;

    /* --- X-direction faces --- */
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i <= mesh->nx; ++i) {
                size_t idxL = mesh_index(mesh, i - 1, j, k);
                size_t idxR = mesh_index(mesh, i, j, k);

                double dxi = 0.5 * (mesh_dx(mesh, i - 1) + mesh_dx(mesh, i));

                double dadx = (alpha[idxR] - alpha[idxL]) / dxi;

                double dady = 0.0;
                if (dim >= 2) {
                    double dyj = mesh_dy(mesh, j);
                    size_t Ljm = mesh_index(mesh, i - 1, j - 1, k);
                    size_t Ljp = mesh_index(mesh, i - 1, j + 1, k);
                    size_t Rjm = mesh_index(mesh, i, j - 1, k);
                    size_t Rjp = mesh_index(mesh, i, j + 1, k);

                    double dady_L = (alpha[Ljp] - alpha[Ljm]) / (2.0 * dyj);
                    double dady_R = (alpha[Rjp] - alpha[Rjm]) / (2.0 * dyj);
                    dady = 0.5 * (dady_L + dady_R);
                }

                double dadz = 0.0;
                if (dim >= 3) {
                    double dzk = mesh_dz(mesh, k);
                    size_t Lkm = mesh_index(mesh, i - 1, j, k - 1);
                    size_t Lkp = mesh_index(mesh, i - 1, j, k + 1);
                    size_t Rkm = mesh_index(mesh, i, j, k - 1);
                    size_t Rkp = mesh_index(mesh, i, j, k + 1);

                    double dadz_L = (alpha[Lkp] - alpha[Lkm]) / (2.0 * dzk);
                    double dadz_R = (alpha[Rkp] - alpha[Rkm]) / (2.0 * dzk);
                    dadz = 0.5 * (dadz_L + dadz_R);
                }

                double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                double normGradReg = std::max(normGrad, eps);

                double T_xx = sigma * (dady * dady + dadz * dadz) / normGradReg;
                double T_xy = -sigma * dadx * dady / normGradReg;
                double T_xz = -sigma * dadx * dadz / normGradReg;

                double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                double vFace = 0.0;
                double wFace = 0.0;
                if (dim >= 2) vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                if (dim >= 3) wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                double work = T_xx * uFace + T_xy * vFace + T_xz * wFace;

                double area = mesh_faceAreaX(mesh, j, k);

                if (i >= 1) {
                    double coeff = area / mesh_cell_volume(mesh, i - 1, j, k);
                    rhsRhoU[idxL] += coeff * T_xx;
                    if (dim >= 2) rhsRhoV[idxL] += coeff * T_xy;
                    if (dim >= 3) rhsRhoW[idxL] += coeff * T_xz;
                    rhsRhoE[idxL] += coeff * work;
                }

                if (i < mesh->nx) {
                    double coeff = area / mesh_cell_volume(mesh, i, j, k);
                    rhsRhoU[idxR] -= coeff * T_xx;
                    if (dim >= 2) rhsRhoV[idxR] -= coeff * T_xy;
                    if (dim >= 3) rhsRhoW[idxR] -= coeff * T_xz;
                    rhsRhoE[idxR] -= coeff * work;
                }
            }
        }
    }

    /* --- Y-direction faces --- */
    if (dim >= 2) {
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j <= mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idxL = mesh_index(mesh, i, j - 1, k);
                    size_t idxR = mesh_index(mesh, i, j, k);

                    double dyj = 0.5 * (mesh_dy(mesh, j - 1) + mesh_dy(mesh, j));

                    double dady = (alpha[idxR] - alpha[idxL]) / dyj;

                    double dxi = mesh_dx(mesh, i);
                    size_t Lim = mesh_index(mesh, i - 1, j - 1, k);
                    size_t Lip = mesh_index(mesh, i + 1, j - 1, k);
                    size_t Rim = mesh_index(mesh, i - 1, j, k);
                    size_t Rip = mesh_index(mesh, i + 1, j, k);

                    double dadx_L = (alpha[Lip] - alpha[Lim]) / (2.0 * dxi);
                    double dadx_R = (alpha[Rip] - alpha[Rim]) / (2.0 * dxi);
                    double dadx = 0.5 * (dadx_L + dadx_R);

                    double dadz = 0.0;
                    if (dim >= 3) {
                        double dzk = mesh_dz(mesh, k);
                        size_t Lkm = mesh_index(mesh, i, j - 1, k - 1);
                        size_t Lkp = mesh_index(mesh, i, j - 1, k + 1);
                        size_t Rkm = mesh_index(mesh, i, j, k - 1);
                        size_t Rkp = mesh_index(mesh, i, j, k + 1);

                        double dadz_L = (alpha[Lkp] - alpha[Lkm]) / (2.0 * dzk);
                        double dadz_R = (alpha[Rkp] - alpha[Rkm]) / (2.0 * dzk);
                        dadz = 0.5 * (dadz_L + dadz_R);
                    }

                    double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                    double normGradReg = std::max(normGrad, eps);

                    double T_yx = -sigma * dady * dadx / normGradReg;
                    double T_yy = sigma * (dadx * dadx + dadz * dadz) / normGradReg;
                    double T_yz = -sigma * dady * dadz / normGradReg;

                    double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                    double vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                    double wFace = 0.0;
                    if (dim >= 3) wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                    double work = T_yx * uFace + T_yy * vFace + T_yz * wFace;

                    double area = mesh_faceAreaY(mesh, i, k);

                    if (j >= 1) {
                        double coeff = area / mesh_cell_volume(mesh, i, j - 1, k);
                        rhsRhoU[idxL] += coeff * T_yx;
                        rhsRhoV[idxL] += coeff * T_yy;
                        if (dim >= 3) rhsRhoW[idxL] += coeff * T_yz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (j < mesh->ny) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
                        rhsRhoU[idxR] -= coeff * T_yx;
                        rhsRhoV[idxR] -= coeff * T_yy;
                        if (dim >= 3) rhsRhoW[idxR] -= coeff * T_yz;
                        rhsRhoE[idxR] -= coeff * work;
                    }
                }
            }
        }
    }

    /* --- Z-direction faces --- */
    if (dim >= 3) {
        for (int k = 0; k <= mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idxL = mesh_index(mesh, i, j, k - 1);
                    size_t idxR = mesh_index(mesh, i, j, k);

                    double dzk = 0.5 * (mesh_dz(mesh, k - 1) + mesh_dz(mesh, k));

                    double dadz = (alpha[idxR] - alpha[idxL]) / dzk;

                    double dxi = mesh_dx(mesh, i);
                    size_t Lim = mesh_index(mesh, i - 1, j, k - 1);
                    size_t Lip = mesh_index(mesh, i + 1, j, k - 1);
                    size_t Rim = mesh_index(mesh, i - 1, j, k);
                    size_t Rip = mesh_index(mesh, i + 1, j, k);

                    double dadx_L = (alpha[Lip] - alpha[Lim]) / (2.0 * dxi);
                    double dadx_R = (alpha[Rip] - alpha[Rim]) / (2.0 * dxi);
                    double dadx = 0.5 * (dadx_L + dadx_R);

                    double dyj = mesh_dy(mesh, j);
                    size_t Ljm = mesh_index(mesh, i, j - 1, k - 1);
                    size_t Ljp = mesh_index(mesh, i, j + 1, k - 1);
                    size_t Rjm = mesh_index(mesh, i, j - 1, k);
                    size_t Rjp = mesh_index(mesh, i, j + 1, k);

                    double dady_L = (alpha[Ljp] - alpha[Ljm]) / (2.0 * dyj);
                    double dady_R = (alpha[Rjp] - alpha[Rjm]) / (2.0 * dyj);
                    double dady = 0.5 * (dady_L + dady_R);

                    double normGrad = std::sqrt(dadx * dadx + dady * dady + dadz * dadz);
                    double normGradReg = std::max(normGrad, eps);

                    double T_zx = -sigma * dadz * dadx / normGradReg;
                    double T_zy = -sigma * dadz * dady / normGradReg;
                    double T_zz = sigma * (dadx * dadx + dady * dady) / normGradReg;

                    double uFace = 0.5 * (state->velU[idxL] + state->velU[idxR]);
                    double vFace = 0.5 * (state->velV[idxL] + state->velV[idxR]);
                    double wFace = 0.5 * (state->velW[idxL] + state->velW[idxR]);

                    double work = T_zx * uFace + T_zy * vFace + T_zz * wFace;

                    double area = mesh_faceAreaZ(mesh, i, j);

                    if (k >= 1) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k - 1);
                        rhsRhoU[idxL] += coeff * T_zx;
                        rhsRhoV[idxL] += coeff * T_zy;
                        rhsRhoW[idxL] += coeff * T_zz;
                        rhsRhoE[idxL] += coeff * work;
                    }

                    if (k < mesh->nz) {
                        double coeff = area / mesh_cell_volume(mesh, i, j, k);
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
