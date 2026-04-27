#include "RiemannSolver.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#pragma omp declare target
RiemannFlux computeLFFlux(
    const PrimitiveState* left,
    const PrimitiveState* right,
    const double* normal,
    const FluxConfig* fc)
{
    /* No memset on `flux`: every field consumed by the caller in the active
     * dim / nPhases / useIGR path is explicitly written below.  The unused
     * tail (momentumFlux[dim..2], alphaFlux[nPhases..]) is never read. */
    RiemannFlux flux;
    const int dim = fc->dim;
    const int includePressure = fc->includePressure;

    double uL = normalVelocity(left, normal, dim);
    double uR = normalVelocity(right, normal, dim);

    double uLS = 0.0, uRS = 0.0;
    for (int i = 0; i < dim; ++i) {
        uLS += left->u[i] * left->u[i];
        uRS += right->u[i] * right->u[i];
    }
    uLS = std::sqrt(uLS);
    uRS = std::sqrt(uRS);

    double C;
    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        C = std::max(uLS, uRS) + std::max(cL, cR);
    } else {
        C = std::max(uLS, uRS);
    }

    flux.massFlux = 0.5 * (left->rho * uL + right->rho * uR) -
                    0.5 * C * (right->rho - left->rho);

    for (int i = 0; i < dim; ++i) {
        flux.momentumFlux[i] = 0.5 * (left->rho * left->u[i] * uL + right->rho * right->u[i] * uR) -
                               0.5 * C * (right->rho * right->u[i] - left->rho * left->u[i]);
    }

    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);
    flux.energyFlux = 0.5 * (rhoEL * uL + rhoER * uR) - 0.5 * C * (rhoER - rhoEL);

    if (includePressure) {
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] += 0.5 * (left->p + right->p) * normal[i];
        flux.energyFlux += 0.5 * (left->p * uL + right->p * uR);
    }

    if (fc->useIGR) {
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] += 0.5 * (left->sigma + right->sigma) * normal[i];
        flux.energyFlux += 0.5 * (left->sigma * uL + right->sigma * uR);
    }

    flux.faceVelocity = 0.5 * (uL + uR);
    flux.pressureFlux = 0.5 * (left->p * uL + right->p * uR) - 0.5 * C * (right->p - left->p);
    for (int ph = 0; ph < fc->nPhases; ++ph) {
        flux.alphaFlux[ph] = 0.5 * (left->alpha[ph] * uL + right->alpha[ph] * uR)
                           - 0.5 * C * (right->alpha[ph] - left->alpha[ph]);
    }

    return flux;
}

RiemannFlux computeRusanovFlux(
    const PrimitiveState* left,
    const PrimitiveState* right,
    const double* normal,
    const FluxConfig* fc)
{
    RiemannFlux flux;
    const int dim = fc->dim;
    const int includePressure = fc->includePressure;

    double uL = normalVelocity(left, normal, dim);
    double uR = normalVelocity(right, normal, dim);

    double absUL = std::abs(uL);
    double absUR = std::abs(uR);
    double sMax;
    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        sMax = std::max(absUL + cL, absUR + cR);
    } else {
        sMax = std::max(absUL, absUR);
    }

    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);

    double massFluxL = left->rho * uL;
    double momFluxL[3] = {};
    for (int i = 0; i < dim; ++i)
        momFluxL[i] = left->rho * left->u[i] * uL;
    double energyFluxL = rhoEL * uL;

    double massFluxR = right->rho * uR;
    double momFluxR[3] = {};
    for (int i = 0; i < dim; ++i)
        momFluxR[i] = right->rho * right->u[i] * uR;
    double energyFluxR = rhoER * uR;

    if (includePressure) {
        for (int i = 0; i < dim; ++i) {
            momFluxL[i] += left->p * normal[i];
            momFluxR[i] += right->p * normal[i];
        }
        energyFluxL += left->p * uL;
        energyFluxR += right->p * uR;
    }

    flux.massFlux = 0.5 * (massFluxL + massFluxR) - 0.5 * sMax * (right->rho - left->rho);
    for (int i = 0; i < dim; ++i) {
        flux.momentumFlux[i] = 0.5 * (momFluxL[i] + momFluxR[i])
            - 0.5 * sMax * (right->rho * right->u[i] - left->rho * left->u[i]);
    }
    flux.energyFlux = 0.5 * (energyFluxL + energyFluxR) - 0.5 * sMax * (rhoER - rhoEL);

    flux.faceVelocity = 0.5 * (uL + uR);
    flux.pressureFlux = 0.5 * (left->p * uL + right->p * uR) - 0.5 * sMax * (right->p - left->p);
    for (int ph = 0; ph < fc->nPhases; ++ph) {
        flux.alphaFlux[ph] = 0.5 * (left->alpha[ph] * uL + right->alpha[ph] * uR)
                           - 0.5 * sMax * (right->alpha[ph] - left->alpha[ph]);
    }

    return flux;
}

RiemannFlux computeHLLCFlux(
    const PrimitiveState* left,
    const PrimitiveState* right,
    const double* normal,
    const FluxConfig* fc)
{
    RiemannFlux flux;
    const int dim = fc->dim;
    const int includePressure = fc->includePressure;

    /* Hoist all per-side scalar reads into locals so the compiler keeps them
     * in registers instead of issuing repeated global-memory loads through
     * the `left` / `right` pointers across the conditional branches below. */
    const double rhoL = left->rho;
    const double rhoR = right->rho;
    const double pL = left->p;
    const double pR = right->p;
    const double uL_x = left->u[0];
    const double uL_y = left->u[1];
    const double uL_z = left->u[2];
    const double uR_x = right->u[0];
    const double uR_y = right->u[1];
    const double uR_z = right->u[2];

    double uL = uL_x * normal[0];
    double uR = uR_x * normal[0];
    if (dim >= 2) { uL += uL_y * normal[1]; uR += uR_y * normal[1]; }
    if (dim >= 3) { uL += uL_z * normal[2]; uR += uR_z * normal[2]; }

    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);

    double sL, sR, sStar;

    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        sL = std::min(uL - cL, uR - cR);
        sR = std::max(uL + cL, uR + cR);
        sStar = (pR - pL
                 + rhoL * uL * (sL - uL)
                 - rhoR * uR * (sR - uR))
              / (rhoL * (sL - uL) - rhoR * (sR - uR));
    } else {
        sL = std::min(uL, uR);
        sR = std::max(uL, uR);
        sStar = 0.5 * (uL + uR);
    }

    /* Bundle the dim-3 velocity arrays for the loops below. */
    const double uLv[3] = {uL_x, uL_y, uL_z};
    const double uRv[3] = {uR_x, uR_y, uR_z};

    if (sL >= 0) {
        flux.massFlux = rhoL * uL;
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] = rhoL * uLv[i] * uL;
        flux.energyFlux = rhoEL * uL;
        if (includePressure) {
            for (int i = 0; i < dim; ++i)
                flux.momentumFlux[i] += pL * normal[i];
            flux.energyFlux += pL * uL;
        }
    } else if (sR <= 0) {
        flux.massFlux = rhoR * uR;
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] = rhoR * uRv[i] * uR;
        flux.energyFlux = rhoER * uR;
        if (includePressure) {
            for (int i = 0; i < dim; ++i)
                flux.momentumFlux[i] += pR * normal[i];
            flux.energyFlux += pR * uR;
        }
    } else if (sStar >= 0) {
        double rhoStarL = rhoL * (sL - uL) / (sL - sStar);
        flux.massFlux = rhoL * uL + sL * (rhoStarL - rhoL);
        if (includePressure) {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarL = rhoStarL * (uLv[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = rhoL * uLv[i] * uL + pL * normal[i]
                    + sL * (rhoUStarL - rhoL * uLv[i]);
            }
            double eL = rhoEL / rhoL;
            double EStarL = rhoStarL * (eL + (sStar - uL) * (sStar + pL / (rhoL * (sL - uL))));
            flux.energyFlux = (rhoEL + pL) * uL + sL * (EStarL - rhoEL);
        } else {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarL = rhoStarL * (uLv[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = rhoL * uLv[i] * uL
                    + sL * (rhoUStarL - rhoL * uLv[i]);
            }
            double eL = rhoEL / rhoL;
            double EStarL = rhoStarL * (eL + (sStar - uL) * sStar);
            flux.energyFlux = rhoEL * uL + sL * (EStarL - rhoEL);
        }
    } else {
        double rhoStarR = rhoR * (sR - uR) / (sR - sStar);
        flux.massFlux = rhoR * uR + sR * (rhoStarR - rhoR);
        if (includePressure) {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarR = rhoStarR * (uRv[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = rhoR * uRv[i] * uR + pR * normal[i]
                    + sR * (rhoUStarR - rhoR * uRv[i]);
            }
            double eR = rhoER / rhoR;
            double EStarR = rhoStarR * (eR + (sStar - uR) * (sStar + pR / (rhoR * (sR - uR))));
            flux.energyFlux = (rhoER + pR) * uR + sR * (EStarR - rhoER);
        } else {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarR = rhoStarR * (uRv[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = rhoR * uRv[i] * uR
                    + sR * (rhoUStarR - rhoR * uRv[i]);
            }
            double eR = rhoER / rhoR;
            double EStarR = rhoStarR * (eR + (sStar - uR) * sStar);
            flux.energyFlux = rhoER * uR + sR * (EStarR - rhoER);
        }
    }

    if (sL >= 0) {
        flux.faceVelocity = uL;
        flux.pressureFlux = pL * uL;
        for (int ph = 0; ph < fc->nPhases; ++ph)
            flux.alphaFlux[ph] = left->alpha[ph] * uL;
    } else if (sR <= 0) {
        flux.faceVelocity = uR;
        flux.pressureFlux = pR * uR;
        for (int ph = 0; ph < fc->nPhases; ++ph)
            flux.alphaFlux[ph] = right->alpha[ph] * uR;
    } else {
        flux.faceVelocity = sStar;
        if (sStar >= 0) {
            flux.pressureFlux = pL * sStar;
            for (int ph = 0; ph < fc->nPhases; ++ph)
                flux.alphaFlux[ph] = left->alpha[ph] * sStar;
        } else {
            flux.pressureFlux = pR * sStar;
            for (int ph = 0; ph < fc->nPhases; ++ph)
                flux.alphaFlux[ph] = right->alpha[ph] * sStar;
        }
    }

    return flux;
}
#pragma omp end declare target
