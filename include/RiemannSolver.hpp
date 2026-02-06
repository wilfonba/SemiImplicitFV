#ifndef RIEMANN_SOLVER_HPP
#define RIEMANN_SOLVER_HPP

#include "State.hpp"
#include "EquationOfState.hpp"
#include <array>
#include <memory>
#include <string>

namespace SemiImplicitFV {

// Flux structure for advective (pressure-free) terms
// Used in the explicit advection step of semi-implicit method
struct RiemannFlux {
    double massFlux;                   // ρu_n
    std::array<double, 3> momentumFlux; // ρu⊗u_n (no pressure!)
    double energyFlux;                 // Eu_n (no pu term!)

    RiemannFlux() : massFlux(0.0), momentumFlux{0.0, 0.0, 0.0}, energyFlux(0.0) {}
};

// Abstract base class for Riemann solvers
// By default (includePressure=false), solves the advective (pressure-free) part:
//   F_adv = [ρu, ρu², ρuv, ρuw, Eu]^T
// When includePressure is set, solves the full Euler flux:
//   F = [ρu, ρu² + p, ρuv, ρuw, (E+p)u]^T
class RiemannSolver {
public:
    explicit RiemannSolver(std::shared_ptr<EquationOfStateBase> eos,
                                    bool includePressure = false)
        : eos_(std::move(eos)), includePressure_(includePressure) {}

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
    const EquationOfStateBase& eos() const { return *eos_; }

protected:
    std::shared_ptr<EquationOfStateBase> eos_;
    bool includePressure_;
};

// Upwind solver for advective flux
class UpwindSolver : public RiemannSolver {
public:
    explicit UpwindSolver(std::shared_ptr<EquationOfStateBase> eos,
                                   bool includePressure = false)
        : RiemannSolver(std::move(eos), includePressure) {}

    RiemannFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override {
        return includePressure_ ? "Upwind" : "UpwindAdvective";
    }
};

// Rusanov/LLF solver for advective flux
class RusanovSolver : public RiemannSolver {
public:
    explicit RusanovSolver(std::shared_ptr<EquationOfStateBase> eos,
                                    bool includePressure = false)
        : RiemannSolver(std::move(eos), includePressure) {}

    RiemannFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override {
        return includePressure_ ? "Rusanov" : "RusanovAdvective";
    }
};

// HLLC-type solver for advective flux
// When includePressure is false, all eigenvalues are u so this reduces to a simpler form
class HLLCSolver : public RiemannSolver {
public:
    explicit HLLCSolver(std::shared_ptr<EquationOfStateBase> eos,
                                  bool includePressure = false)
        : RiemannSolver(std::move(eos), includePressure) {}

    RiemannFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override {
        return includePressure_ ? "HLLC" : "HLLCAdvective";
    }
};

} // namespace SemiImplicitFV

#endif // RIEMANN_SOLVER_HPP
