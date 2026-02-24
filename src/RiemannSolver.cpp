#include "RiemannSolver.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

RiemannFlux computeLFFlux(
    const PrimitiveState* left,
    const PrimitiveState* right,
    const double* normal,
    const FluxConfig* fc)
{
    RiemannFlux flux;
    memset(&flux, 0, sizeof(flux));
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
    memset(&flux, 0, sizeof(flux));
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
    memset(&flux, 0, sizeof(flux));
    const int dim = fc->dim;
    const int includePressure = fc->includePressure;

    double uL = normalVelocity(left, normal, dim);
    double uR = normalVelocity(right, normal, dim);

    double rhoEL = rhoEFromState(left);
    double rhoER = rhoEFromState(right);

    double sL, sR, sStar;

    if (includePressure) {
        double cL = soundSpeedDirect(left);
        double cR = soundSpeedDirect(right);
        sL = std::min(uL - cL, uR - cR);
        sR = std::max(uL + cL, uR + cR);
        sStar = (right->p - left->p
                 + left->rho * uL * (sL - uL)
                 - right->rho * uR * (sR - uR))
              / (left->rho * (sL - uL) - right->rho * (sR - uR));
    } else {
        sL = std::min(uL, uR);
        sR = std::max(uL, uR);
        sStar = 0.5 * (uL + uR);
    }

    if (sL >= 0) {
        flux.massFlux = left->rho * uL;
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] = left->rho * left->u[i] * uL;
        flux.energyFlux = rhoEL * uL;
        if (includePressure) {
            for (int i = 0; i < dim; ++i)
                flux.momentumFlux[i] += left->p * normal[i];
            flux.energyFlux += left->p * uL;
        }
    } else if (sR <= 0) {
        flux.massFlux = right->rho * uR;
        for (int i = 0; i < dim; ++i)
            flux.momentumFlux[i] = right->rho * right->u[i] * uR;
        flux.energyFlux = rhoER * uR;
        if (includePressure) {
            for (int i = 0; i < dim; ++i)
                flux.momentumFlux[i] += right->p * normal[i];
            flux.energyFlux += right->p * uR;
        }
    } else if (sStar >= 0) {
        double rhoStarL = left->rho * (sL - uL) / (sL - sStar);
        flux.massFlux = left->rho * uL + sL * (rhoStarL - left->rho);
        if (includePressure) {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarL = rhoStarL * (left->u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left->rho * left->u[i] * uL + left->p * normal[i]
                    + sL * (rhoUStarL - left->rho * left->u[i]);
            }
            double eL = rhoEL / left->rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * (sStar + left->p / (left->rho * (sL - uL))));
            flux.energyFlux = (rhoEL + left->p) * uL + sL * (EStarL - rhoEL);
        } else {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarL = rhoStarL * (left->u[i] + (sStar - uL) * normal[i]);
                flux.momentumFlux[i] = left->rho * left->u[i] * uL
                    + sL * (rhoUStarL - left->rho * left->u[i]);
            }
            double eL = rhoEL / left->rho;
            double EStarL = rhoStarL * (eL + (sStar - uL) * sStar);
            flux.energyFlux = rhoEL * uL + sL * (EStarL - rhoEL);
        }
    } else {
        double rhoStarR = right->rho * (sR - uR) / (sR - sStar);
        flux.massFlux = right->rho * uR + sR * (rhoStarR - right->rho);
        if (includePressure) {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarR = rhoStarR * (right->u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right->rho * right->u[i] * uR + right->p * normal[i]
                    + sR * (rhoUStarR - right->rho * right->u[i]);
            }
            double eR = rhoER / right->rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * (sStar + right->p / (right->rho * (sR - uR))));
            flux.energyFlux = (rhoER + right->p) * uR + sR * (EStarR - rhoER);
        } else {
            for (int i = 0; i < dim; ++i) {
                double rhoUStarR = rhoStarR * (right->u[i] + (sStar - uR) * normal[i]);
                flux.momentumFlux[i] = right->rho * right->u[i] * uR
                    + sR * (rhoUStarR - right->rho * right->u[i]);
            }
            double eR = rhoER / right->rho;
            double EStarR = rhoStarR * (eR + (sStar - uR) * sStar);
            flux.energyFlux = rhoER * uR + sR * (EStarR - rhoER);
        }
    }

    if (sL >= 0) {
        flux.faceVelocity = uL;
        flux.pressureFlux = left->p * uL;
        for (int ph = 0; ph < fc->nPhases; ++ph)
            flux.alphaFlux[ph] = left->alpha[ph] * uL;
    } else if (sR <= 0) {
        flux.faceVelocity = uR;
        flux.pressureFlux = right->p * uR;
        for (int ph = 0; ph < fc->nPhases; ++ph)
            flux.alphaFlux[ph] = right->alpha[ph] * uR;
    } else {
        flux.faceVelocity = sStar;
        if (sStar >= 0) {
            flux.pressureFlux = left->p * sStar;
            for (int ph = 0; ph < fc->nPhases; ++ph)
                flux.alphaFlux[ph] = left->alpha[ph] * sStar;
        } else {
            flux.pressureFlux = right->p * sStar;
            for (int ph = 0; ph < fc->nPhases; ++ph)
                flux.alphaFlux[ph] = right->alpha[ph] * sStar;
        }
    }

    return flux;
}
