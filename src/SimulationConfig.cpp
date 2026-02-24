#include "SimulationConfig.hpp"
#include <mpi.h>
#include <cstring>
#include <stdexcept>
#include <string>

SimulationConfig config_defaults(void) {
    SimulationConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.dim = 3;
    cfg.nGhost = 2;
    cfg.RKOrder = 1;
    cfg.useIGR = 0;
    cfg.semiImplicit = 0;
    cfg.reconOrder = WENO1;
    cfg.wenoEps = 1e-6;
    cfg.step = 0;
    cfg.time = 0.0;

    cfg.explicitParams.cfl = 0.8;
    cfg.explicitParams.constDt = -1.0;
    cfg.explicitParams.maxDt = 1e-2;
    cfg.explicitParams.minDt = 1e-12;

    cfg.semiImplicitParams.cfl = 0.8;
    cfg.semiImplicitParams.constDt = -1.0;
    cfg.semiImplicitParams.maxDt = 1e-2;
    cfg.semiImplicitParams.minDt = 1e-12;
    cfg.semiImplicitParams.maxAcousticCFL = -1.0;
    cfg.semiImplicitParams.maxPressureIters = 100;
    cfg.semiImplicitParams.pressureTol = 1e-8;
    cfg.semiImplicitParams.singlePressureSolve = 0;

    cfg.igrParams.alphaCoeff = 1.0;
    cfg.igrParams.IGRIters = 5;
    cfg.igrParams.IGRWarmStartIters = 50;

    cfg.multiPhaseParams.nPhases = 0;
    cfg.multiPhaseParams.alphaMin = 1e-8;
    for (int i = 0; i < MAX_PHASES; ++i) {
        cfg.multiPhaseParams.phases[i].gamma = 1.4;
        cfg.multiPhaseParams.phases[i].pInf = 0.0;
    }

    cfg.viscousParams.mu = 0.0;
    cfg.viscousParams.nPhaseMu = 0;

    cfg.surfaceTensionParams.sigma = 0.0;
    cfg.surfaceTensionParams.epsGradAlpha = 1e-8;

    return cfg;
}

int config_is_multi_phase(const SimulationConfig* cfg) {
    return cfg->multiPhaseParams.nPhases > 0;
}

int config_has_viscosity(const SimulationConfig* cfg) {
    return cfg->viscousParams.mu > 0.0 || cfg->viscousParams.nPhaseMu > 0;
}

int config_has_surface_tension(const SimulationConfig* cfg) {
    return cfg->surfaceTensionParams.sigma > 0.0;
}

int config_has_body_force(const SimulationConfig* cfg) {
    for (int i = 0; i < 3; ++i)
        if (cfg->bodyForceParams.a[i] != 0.0 || cfg->bodyForceParams.b[i] != 0.0)
            return 1;
    return 0;
}

int config_required_ghost_cells(const SimulationConfig* cfg) {
    switch (cfg->reconOrder) {
        case WENO1:
        case UPWIND1:
            return 1;
        case WENO3:
        case UPWIND3:
            return 2;
        case WENO5:
        case UPWIND5:
            return 3;
    }
    return 1;
}

void config_validate(const SimulationConfig* cfg) {
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (mpiInitialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0) return;
    }

    if (cfg->dim < 1 || cfg->dim > 3)
        throw std::invalid_argument("dim must be 1, 2, or 3 (got " + std::to_string(cfg->dim) + ")");

    if (cfg->RKOrder < 1 || cfg->RKOrder > 3)
        throw std::invalid_argument("RKOrder must be 1, 2, or 3 (got " + std::to_string(cfg->RKOrder) + ")");

    int reqGhost = config_required_ghost_cells(cfg);
    if (cfg->nGhost < reqGhost)
        throw std::invalid_argument("nGhost=" + std::to_string(cfg->nGhost)
            + " is too small for chosen reconOrder (need >= " + std::to_string(reqGhost) + ")");

    if (cfg->semiImplicit) {
        const SemiImplicitParams* p = &cfg->semiImplicitParams;
        if (p->cfl <= 0)
            throw std::invalid_argument("semiImplicitParams.cfl must be > 0");
        if (p->minDt <= 0)
            throw std::invalid_argument("semiImplicitParams.minDt must be > 0");
        if (p->maxDt <= p->minDt)
            throw std::invalid_argument("semiImplicitParams.maxDt must be > minDt");
        if (p->pressureTol <= 0)
            throw std::invalid_argument("semiImplicitParams.pressureTol must be > 0");
        if (p->maxPressureIters <= 0)
            throw std::invalid_argument("semiImplicitParams.maxPressureIters must be > 0");
    } else {
        const ExplicitParams* p = &cfg->explicitParams;
        if (p->cfl <= 0)
            throw std::invalid_argument("explicitParams.cfl must be > 0");
        if (p->minDt <= 0)
            throw std::invalid_argument("explicitParams.minDt must be > 0");
        if (p->maxDt <= p->minDt)
            throw std::invalid_argument("explicitParams.maxDt must be > minDt");
    }

    if (cfg->useIGR) {
        const IGRParams* p = &cfg->igrParams;
        if (p->alphaCoeff <= 0)
            throw std::invalid_argument("igrParams.alphaCoeff must be > 0");
        if (p->IGRIters <= 0)
            throw std::invalid_argument("igrParams.IGRIters must be > 0");
        if (p->IGRWarmStartIters < 0)
            throw std::invalid_argument("igrParams.IGRWarmStartIters must be >= 0");
    }

    if (config_is_multi_phase(cfg)) {
        const MultiPhaseParams* mp = &cfg->multiPhaseParams;
        for (int ph = 0; ph < mp->nPhases; ++ph) {
            if (mp->phases[ph].gamma <= 1.0)
                throw std::invalid_argument("multiPhaseParams.phases[" + std::to_string(ph)
                    + "].gamma must be > 1");
            if (mp->phases[ph].pInf < 0.0)
                throw std::invalid_argument("multiPhaseParams.phases[" + std::to_string(ph)
                    + "].pInf must be >= 0");
        }
        if (mp->alphaMin <= 0.0)
            throw std::invalid_argument("multiPhaseParams.alphaMin must be > 0");
    }

    if (cfg->viscousParams.mu < 0.0)
        throw std::invalid_argument("viscousParams.mu must be >= 0");

    if (cfg->viscousParams.nPhaseMu > 0) {
        if (!config_is_multi_phase(cfg))
            throw std::invalid_argument("viscousParams.phaseMu requires multi-phase (nPhases >= 2)");
        if (cfg->viscousParams.nPhaseMu != cfg->multiPhaseParams.nPhases)
            throw std::invalid_argument("viscousParams.nPhaseMu must equal nPhases");
        for (int ph = 0; ph < cfg->multiPhaseParams.nPhases; ++ph)
            if (cfg->viscousParams.phaseMu[ph] < 0.0)
                throw std::invalid_argument("viscousParams.phaseMu[" + std::to_string(ph)
                    + "] must be >= 0");
    }

    if (cfg->surfaceTensionParams.sigma < 0.0)
        throw std::invalid_argument("surfaceTensionParams.sigma must be >= 0");
    if (cfg->surfaceTensionParams.epsGradAlpha <= 0.0)
        throw std::invalid_argument("surfaceTensionParams.epsGradAlpha must be > 0");
}
