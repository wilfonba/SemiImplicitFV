#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include "State.hpp"

enum ReconstructionOrder {
    WENO1,
    WENO3,
    WENO5,
    UPWIND1,
    UPWIND3,
    UPWIND5
};

struct ExplicitParams {
    double cfl;
    double constDt;
    double maxDt;
    double minDt;
};

struct SemiImplicitParams {
    double cfl;
    double constDt;
    double maxDt;
    double minDt;
    double maxAcousticCFL;
    int maxPressureIters;
    double pressureTol;
    int singlePressureSolve;
};

struct IGRParams {
    double alphaCoeff;
    int IGRIters;
    int IGRWarmStartIters;
};

struct PhaseEOS {
    double gamma;
    double pInf;
};

struct MultiPhaseParams {
    int nPhases;
    PhaseEOS phases[MAX_PHASES];
    double alphaMin;
};

struct BodyForceParams {
    double a[3];
    double b[3];
    double c[3];
    double d[3];
};

struct ViscousParams {
    double mu;
    double phaseMu[MAX_PHASES];
    int nPhaseMu;
};

struct SurfaceTensionParams {
    double sigma;
    double epsGradAlpha;
};

struct SimulationConfig {
    int dim;
    int nGhost;
    int RKOrder;
    int useIGR;
    int semiImplicit;
    enum ReconstructionOrder reconOrder;
    double wenoEps;
    int step;
    double time;

    struct ExplicitParams explicitParams;
    struct SemiImplicitParams semiImplicitParams;
    struct IGRParams igrParams;
    struct MultiPhaseParams multiPhaseParams;
    struct BodyForceParams bodyForceParams;
    struct ViscousParams viscousParams;
    struct SurfaceTensionParams surfaceTensionParams;
};

SimulationConfig config_defaults(void);
void config_validate(const SimulationConfig* cfg);
int config_is_multi_phase(const SimulationConfig* cfg);
int config_has_viscosity(const SimulationConfig* cfg);
int config_has_surface_tension(const SimulationConfig* cfg);
int config_has_body_force(const SimulationConfig* cfg);
int config_required_ghost_cells(const SimulationConfig* cfg);

#endif /* SIMULATION_CONFIG_HPP */
