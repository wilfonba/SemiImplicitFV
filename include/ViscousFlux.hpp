#ifndef VISCOUS_FLUX_HPP
#define VISCOUS_FLUX_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

/* Add viscous stress contributions (Newtonian fluid, Stokes hypothesis) to the
   momentum and energy RHS arrays.  The viscous stress tensor is:

     tau_ij = mu * (du_i/dx_j + du_j/dx_i) - (2/3) * mu * div(u) * delta_ij

   Velocity gradients are computed at cell faces using central differences
   (normal direction) and averaged cell-center differences (transverse).
   Viscous work (tau . u) is added to the energy RHS. */
void add_viscous_fluxes(
    const struct SimulationConfig* config,
    const struct RectilinearMesh* mesh,
    const struct SolutionState* state,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE);

/* GPU-offload variant.  All field pointers (state->velU/V/W, state->alpha,
 * rhsRho*) must be device-resident.  Uses a cell-based parallel pattern:
 * each cell computes viscous flux contributions from both of its faces in
 * each direction, so interior faces are computed twice.  This matches the
 * Riemann flux pattern in explicit_compute_rhs and avoids atomics. */
void add_viscous_fluxes_device(
    const struct SimulationConfig* config,
    const struct RectilinearMesh* mesh,
    const struct SolutionState* state,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE);

#endif /* VISCOUS_FLUX_HPP */
