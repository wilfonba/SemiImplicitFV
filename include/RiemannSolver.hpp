#ifndef RIEMANN_SOLVER_HPP
#define RIEMANN_SOLVER_HPP

#include "State.hpp"
#include <array>
#include <string>

namespace SemiImplicitFV {

// Flux structure for advective (pressure-free) terms
// Used in the explicit advection step of semi-implicit method
struct AdvectiveFlux {
    double massFlux;                   // ρu_n
    std::array<double, 3> momentumFlux; // ρu⊗u_n (no pressure!)
    double energyFlux;                 // Eu_n (no pu term!)

    AdvectiveFlux() : massFlux(0.0), momentumFlux{0.0, 0.0, 0.0}, energyFlux(0.0) {}
};

// Abstract base class for pressure-free Riemann solvers
// These solve the advective part of the split Euler equations:
//   F_adv = [ρu, ρu², ρuv, ρuw, Eu]^T
// Eigenvalues are all u (not u±c), allowing larger time steps
class AdvectiveRiemannSolver {
public:
    virtual ~AdvectiveRiemannSolver() = default;

    // Compute pressure-free numerical flux at a face
    virtual AdvectiveFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const = 0;

    // Maximum wave speed for CFL (just material velocity, no sound speed)
    virtual double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const = 0;

    virtual std::string name() const = 0;
};

// Upwind solver for advective flux
class UpwindAdvectiveSolver : public AdvectiveRiemannSolver {
public:
    AdvectiveFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override { return "UpwindAdvective"; }
};

// Rusanov/LLF solver for advective flux
class RusanovAdvectiveSolver : public AdvectiveRiemannSolver {
public:
    AdvectiveFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override { return "RusanovAdvective"; }
};

// HLLC-type solver for advective flux
// Since all eigenvalues are u, this reduces to a simpler form
class HLLCAdvectiveSolver : public AdvectiveRiemannSolver {
public:
    AdvectiveFlux computeFlux(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    double maxWaveSpeed(
        const PrimitiveState& left,
        const PrimitiveState& right,
        const std::array<double, 3>& normal
    ) const override;

    std::string name() const override { return "HLLCAdvective"; }
};

} // namespace SemiImplicitFV

#endif // RIEMANN_SOLVER_HPP
