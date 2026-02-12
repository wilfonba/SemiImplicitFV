#ifndef SURFACE_TENSION_HPP
#define SURFACE_TENSION_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <vector>

namespace SemiImplicitFV {

/// Add conservative surface tension (capillary stress tensor) contributions to
/// the momentum and energy RHS arrays.  Following Schmidmayer et al. (2017),
/// the capillary stress tensor for an interface with volume fraction alpha is:
///
///   T_cap = sigma * ( |grad(alpha)| I  -  grad(alpha) x grad(alpha) / |grad(alpha)| )
///
/// Its divergence recovers the classical CSF force: div(T_cap) = sigma * kappa * grad(alpha).
///
/// Gradients of alpha are computed at cell faces using central differences
/// (normal direction) and averaged cell-center differences (transverse),
/// identically to the viscous flux discretization.
void addSurfaceTensionFluxes(
    const SimulationConfig& config,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    double sigma,
    std::vector<double>& rhsRhoU,
    std::vector<double>& rhsRhoV,
    std::vector<double>& rhsRhoW,
    std::vector<double>& rhsRhoE);

} // namespace SemiImplicitFV

#endif // SURFACE_TENSION_HPP
