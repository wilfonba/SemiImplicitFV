#ifndef VISCOUS_FLUX_HPP
#define VISCOUS_FLUX_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <vector>

namespace SemiImplicitFV {

/// Add viscous stress contributions (Newtonian fluid, Stokes hypothesis) to the
/// momentum and energy RHS arrays.  The viscous stress tensor is:
///
///   tau_ij = mu * (du_i/dx_j + du_j/dx_i) - (2/3) * mu * div(u) * delta_ij
///
/// Velocity gradients are computed at cell faces using central differences
/// (normal direction) and averaged cell-center differences (transverse).
/// Viscous work (tau . u) is added to the energy RHS.
void addViscousFluxes(
    const SimulationConfig& config,
    const RectilinearMesh& mesh,
    const SolutionState& state,
    double mu,
    std::vector<double>& rhsRhoU,
    std::vector<double>& rhsRhoV,
    std::vector<double>& rhsRhoW,
    std::vector<double>& rhsRhoE);

} // namespace SemiImplicitFV

#endif // VISCOUS_FLUX_HPP
