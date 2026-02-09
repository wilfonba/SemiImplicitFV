#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include <stdexcept>
#include <string>

namespace SemiImplicitFV {

enum class ReconstructionOrder {
    WENO1,        // piecewise constant (copies cell values)
    WENO3,        // 3rd order WENO (r=2, needs 2 ghost cells)
    WENO5,        // 5th order WENO (r=3, needs 3 ghost cells)
    UPWIND1,      // piecewise constant (copies cell values)
    UPWIND3,      // 3rd order upwind (WENO without shock capturing)
    UPWIND5,      // 5th order upwind (WENO without shock capturing)
};

struct ExplicitParams {
    double cfl = 0.8;
    double constDt = -1.0;       // if > 0, overrides CFL-based time step
    double maxDt = 1e-2;
    double minDt = 1e-12;
};

struct SemiImplicitParams {
    double cfl = 0.8;            // CFL based on material velocity only
    double maxDt = 1e-2;
    double minDt = 1e-12;
    int maxPressureIters = 100;
    double pressureTol = 1e-8;
};

struct IGRParams {
    double alphaCoeff = 1.0;     // alpha = alphaCoeff * dx^2
    int IGRIters = 5;
    int IGRWarmStartIters = 50;
};

struct SimulationConfig {
    // Global parameters
    int dim = 3;
    int nGhost = 2;
    int RKOrder = 1;
    bool useIGR = false;
    bool semiImplicit = false;
    ReconstructionOrder reconOrder = ReconstructionOrder::WENO1;
    double wenoEps = 1e-6;
    int step = 0;

    // Solver-specific parameters
    ExplicitParams explicitParams;
    SemiImplicitParams semiImplicitParams;
    IGRParams igrParams;

    int requiredGhostCells() const {
        switch (reconOrder) {
            case ReconstructionOrder::WENO1:
            case ReconstructionOrder::UPWIND1:
                return 1;
            case ReconstructionOrder::WENO3:
            case ReconstructionOrder::UPWIND3:
                return 2;
            case ReconstructionOrder::WENO5:
            case ReconstructionOrder::UPWIND5:
                return 3;
        }
        return 1;
    }

    void validate() const {
        if (dim < 1 || dim > 3)
            throw std::invalid_argument("dim must be 1, 2, or 3 (got " + std::to_string(dim) + ")");

        if (RKOrder < 1 || RKOrder > 3)
            throw std::invalid_argument("RKOrder must be 1, 2, or 3 (got " + std::to_string(RKOrder) + ")");

        if (semiImplicit && RKOrder > 1)
            throw std::invalid_argument("semiImplicit requires RKOrder=1 (SSP-RK blending not supported for pressure-split scheme)");

        int reqGhost = requiredGhostCells();
        if (nGhost < reqGhost)
            throw std::invalid_argument("nGhost=" + std::to_string(nGhost)
                + " is too small for chosen reconOrder (need >= " + std::to_string(reqGhost) + ")");

        if (semiImplicit) {
            const auto& p = semiImplicitParams;
            if (p.cfl <= 0)
                throw std::invalid_argument("semiImplicitParams.cfl must be > 0");
            if (p.minDt <= 0)
                throw std::invalid_argument("semiImplicitParams.minDt must be > 0");
            if (p.maxDt <= p.minDt)
                throw std::invalid_argument("semiImplicitParams.maxDt must be > minDt");
            if (p.pressureTol <= 0)
                throw std::invalid_argument("semiImplicitParams.pressureTol must be > 0");
            if (p.maxPressureIters <= 0)
                throw std::invalid_argument("semiImplicitParams.maxPressureIters must be > 0");
        } else {
            const auto& p = explicitParams;
            if (p.cfl <= 0)
                throw std::invalid_argument("explicitParams.cfl must be > 0");
            if (p.minDt <= 0)
                throw std::invalid_argument("explicitParams.minDt must be > 0");
            if (p.maxDt <= p.minDt)
                throw std::invalid_argument("explicitParams.maxDt must be > minDt");
        }

        if (useIGR) {
            const auto& p = igrParams;
            if (p.alphaCoeff <= 0)
                throw std::invalid_argument("igrParams.alphaCoeff must be > 0");
            if (p.IGRIters <= 0)
                throw std::invalid_argument("igrParams.IGRIters must be > 0");
            if (p.IGRWarmStartIters < 0)
                throw std::invalid_argument("igrParams.IGRWarmStartIters must be >= 0");
        }
    }
};

} // namespace SemiImplicitFV

#endif // SIMULATION_CONFIG_HPP
