#ifndef SURFACE_TENSION_HPP
#define SURFACE_TENSION_HPP

#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;

/* Add conservative surface tension (capillary stress tensor) contributions to
   the momentum and energy RHS arrays.  Following Schmidmayer et al. (2017),
   the capillary stress tensor for an interface with volume fraction alpha is:

     T_cap = sigma * ( |grad(alpha)| I  -  grad(alpha) x grad(alpha) / |grad(alpha)| )

   Its divergence recovers the classical CSF force: div(T_cap) = sigma * kappa * grad(alpha). */
void add_surface_tension_fluxes(
    const struct SimulationConfig* config,
    const struct RectilinearMesh* mesh,
    const struct SolutionState* state,
    double sigma,
    double* rhsRhoU,
    double* rhsRhoV,
    double* rhsRhoW,
    double* rhsRhoE);

#endif /* SURFACE_TENSION_HPP */
