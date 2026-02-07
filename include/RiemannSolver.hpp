#ifndef RIEMANN_SOLVER_HPP
#define RIEMANN_SOLVER_HPP

#include "State.hpp"
#include "EquationOfState.hpp"
#include "SimulationConfig.hpp"
#include <array>
#include <memory>
#include <string>

namespace SemiImplicitFV {

struct RiemannFlux {
    double massFlux;
    std::array<double, 3> momentumFlux;
    double energyFlux;

    RiemannFlux() : massFlux(0.0), momentumFlux{0.0, 0.0, 0.0}, energyFlux(0.0) {}
};

// Abstract base class for Riemann solvers
// By default (includePressure=false), solves the advective (pressure-free) part:
//   F_adv = [ρu, ρu², ρuv, ρuw, Eu]^T
// When includePressure is set, solves the full Euler flux:
//   F = [ρu, ρu² + p, ρuv, ρuw, (E+p)u]^T
class RiemannSolver {
public:
    explicit RiemannSolver(std::shared_ptr<EquationOfState> eos,
                           bool includePressure = false,
                           const SimulationConfig& config = {})
        : eos_(std::move(eos)), includePressure_(includePressure), config_(config) {}

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

    bool includePressure() const { return includePressure_; }
    const EquationOfState& eos() const { return *eos_; }
    int dim() const { return config_.dim; }

protected:
    std::shared_ptr<EquationOfState> eos_;
    bool includePressure_;
    SimulationConfig config_;
};

// Utility used by solver implementations
inline double normalVelocity(const PrimitiveState& W, const std::array<double, 3>& n, int dim = 3) {
    double vn = 0.0;
    for (int d = 0; d < dim; ++d)
        vn += W.u[d] * n[d];
    return vn;
}

} // namespace SemiImplicitFV

#endif // RIEMANN_SOLVER_HPP

