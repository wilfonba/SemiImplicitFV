#ifndef RIEMANN_SOLVER_HPP
#define RIEMANN_SOLVER_HPP

#include "State.hpp"
#include "EquationOfState.hpp"
#include "SimulationConfig.hpp"
#include <array>
#include <cmath>
#include <memory>
#include <string>

namespace SemiImplicitFV {

// Enum for compile-time-free dispatch of Riemann solver type in flux loops.
// Avoids virtual calls, enabling future OpenACC #pragma acc routine seq.
enum class RiemannSolverType { LF, Rusanov, HLLC };

// Lightweight config passed to free flux functions (no pointers, no STL).
struct FluxConfig {
    int dim;
    bool includePressure;
    bool useIGR;
    int nPhases;  // 0 = single-phase
};

struct RiemannFlux {
    double massFlux;
    std::array<double, 3> momentumFlux;
    double energyFlux;
    double alphaFlux[8];   // volume fraction fluxes (nPhases entries)
    double pressureFlux;   // upwind pressure flux p*u_n (for conservative p advection)
    double faceVelocity;   // normal velocity at face (for div(u) source term)

    RiemannFlux() : massFlux(0.0), momentumFlux{0.0, 0.0, 0.0}, energyFlux(0.0),
                    alphaFlux{}, pressureFlux(0.0), faceVelocity(0.0) {}
};

// Abstract base class for Riemann solvers
class RiemannSolver {
public:
    explicit RiemannSolver(std::shared_ptr<EquationOfState> eos,
                           const SimulationConfig& config = {})
        : eos_(std::move(eos)), config_(config) {}

    virtual ~RiemannSolver() = default;

    virtual std::string name() const = 0;

    const EquationOfState& eos() const { return *eos_; }
    int dim() const { return config_.dim; }

protected:
    std::shared_ptr<EquationOfState> eos_;
    SimulationConfig config_;
};

// Utility used by solver implementations
inline double normalVelocity(const PrimitiveState& W, const std::array<double, 3>& n, int dim = 3) {
    double vn = 0.0;
    for (int d = 0; d < dim; ++d)
        vn += W.u[d] * n[d];
    return vn;
}

// ---- Free (non-virtual) flux functions for GPU-ready dispatch ----
// These require gammaEff > 0 on both left and right states (always true
// after reconstruction populates gammaEff/piInfEff).

// Inline sound speed from gammaEff (no EOS pointer needed)
inline double soundSpeedDirect(const PrimitiveState& W) {
    return std::sqrt(W.gammaEff * std::max(W.p + W.piInfEff, 1e-14) / std::max(W.rho, 1e-14));
}

// Inline rhoE from gammaEff (no EOS pointer needed)
inline double rhoEFromState(const PrimitiveState& W) {
    double ke = 0.5 * W.rho * (W.u[0]*W.u[0] + W.u[1]*W.u[1] + W.u[2]*W.u[2]);
    return (W.p + W.gammaEff * W.piInfEff) / (W.gammaEff - 1.0) + ke;
}

// Per-solver free functions
RiemannFlux computeLFFlux(const PrimitiveState& left, const PrimitiveState& right,
                          const std::array<double, 3>& normal, const FluxConfig& fc);

RiemannFlux computeRusanovFlux(const PrimitiveState& left, const PrimitiveState& right,
                               const std::array<double, 3>& normal, const FluxConfig& fc);

RiemannFlux computeHLLCFlux(const PrimitiveState& left, const PrimitiveState& right,
                             const std::array<double, 3>& normal, const FluxConfig& fc);

// Dispatch by enum (switch-based, branch-predicted after first call)
inline RiemannFlux computeFluxDirect(RiemannSolverType type,
                                     const PrimitiveState& left, const PrimitiveState& right,
                                     const std::array<double, 3>& normal, const FluxConfig& fc) {
    switch (type) {
        case RiemannSolverType::LF:      return computeLFFlux(left, right, normal, fc);
        case RiemannSolverType::Rusanov:  return computeRusanovFlux(left, right, normal, fc);
        case RiemannSolverType::HLLC:     return computeHLLCFlux(left, right, normal, fc);
    }
    return computeLFFlux(left, right, normal, fc); // unreachable
}

} // namespace SemiImplicitFV

#endif // RIEMANN_SOLVER_HPP

