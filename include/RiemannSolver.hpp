#ifndef RIEMANN_SOLVER_HPP
#define RIEMANN_SOLVER_HPP

#include "State.hpp"
#include "SimulationConfig.hpp"
#include <cmath>

enum RiemannSolverType { RS_LF, RS_RUSANOV, RS_HLLC };

struct FluxConfig {
    int dim;
    int includePressure;
    int useIGR;
    int nPhases;
};

struct RiemannFlux {
    double massFlux;
    double momentumFlux[3];
    double energyFlux;
    double alphaFlux[MAX_PHASES];
    double pressureFlux;
    double faceVelocity;
};

static inline double normalVelocity(const PrimitiveState* W, const double* n, int dim) {
    double vn = 0.0;
    for (int d = 0; d < dim; ++d)
        vn += W->u[d] * n[d];
    return vn;
}

static inline double soundSpeedDirect(const PrimitiveState* W) {
    return std::sqrt(W->gammaEff * std::max(W->p + W->piInfEff, 1e-14) / std::max(W->rho, 1e-14));
}

static inline double rhoEFromState(const PrimitiveState* W) {
    double ke = 0.5 * W->rho * (W->u[0]*W->u[0] + W->u[1]*W->u[1] + W->u[2]*W->u[2]);
    return (W->p + W->gammaEff * W->piInfEff) / (W->gammaEff - 1.0) + ke;
}

RiemannFlux computeLFFlux(const PrimitiveState* left, const PrimitiveState* right,
                          const double* normal, const FluxConfig* fc);

RiemannFlux computeRusanovFlux(const PrimitiveState* left, const PrimitiveState* right,
                               const double* normal, const FluxConfig* fc);

RiemannFlux computeHLLCFlux(const PrimitiveState* left, const PrimitiveState* right,
                            const double* normal, const FluxConfig* fc);

static inline RiemannFlux computeFluxDirect(enum RiemannSolverType type,
                                            const PrimitiveState* left, const PrimitiveState* right,
                                            const double* normal, const FluxConfig* fc) {
    switch (type) {
        case RS_LF:      return computeLFFlux(left, right, normal, fc);
        case RS_RUSANOV:  return computeRusanovFlux(left, right, normal, fc);
        case RS_HLLC:     return computeHLLCFlux(left, right, normal, fc);
    }
    return computeLFFlux(left, right, normal, fc);
}

#endif /* RIEMANN_SOLVER_HPP */
