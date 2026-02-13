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

struct RiemannFlux {
    double massFlux;
    std::array<double, 3> momentumFlux;
    double energyFlux;
    double alphaFlux[8];   // volume fraction fluxes (nPhases-1 entries)
    double faceVelocity;   // normal velocity at face (for div(u) source term)

    RiemannFlux() : massFlux(0.0), momentumFlux{0.0, 0.0, 0.0}, energyFlux(0.0),
                    alphaFlux{}, faceVelocity(0.0) {}
};

// Abstract base class for Riemann solvers
class RiemannSolver {
public:
    explicit RiemannSolver(std::shared_ptr<EquationOfState> eos,
                           const SimulationConfig& config = {})
        : eos_(std::move(eos)), config_(config) {}

    virtual ~RiemannSolver() = default;

    // Compute numerical flux at a face
    virtual RiemannFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const = 0;

    // Maximum wave speed for CFL
    // Without pressure: |u| only. With pressure: |u| + c
    virtual double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const = 0;

    virtual std::string name() const = 0;

    const EquationOfState& eos() const { return *eos_; }
    int dim() const { return config_.dim; }

protected:
    std::shared_ptr<EquationOfState> eos_;
    SimulationConfig config_;
};

// Utility: compute sound speed using gammaEff if available (multi-phase),
// otherwise fall back to EOS
inline double soundSpeedFromState(const PrimitiveState& W, const EquationOfState& eos) {
    if (W.gammaEff > 0.0)
        return std::sqrt(W.gammaEff * std::max(W.p + W.piInfEff, 1e-14) / std::max(W.rho, 1e-14));
    return eos.soundSpeed(W);
}

// Utility used by solver implementations
inline double normalVelocity(const PrimitiveState& W, const std::array<double, 3>& n, int dim = 3) {
    double vn = 0.0;
    for (int d = 0; d < dim; ++d)
        vn += W.u[d] * n[d];
    return vn;
}

} // namespace SemiImplicitFV

#endif // RIEMANN_SOLVER_HPP

