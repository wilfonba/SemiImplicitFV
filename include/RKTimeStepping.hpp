#ifndef RK_TIME_STEPPING_HPP
#define RK_TIME_STEPPING_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "EquationOfState.hpp"

namespace SemiImplicitFV {

// Advective time step: CFL based on material velocity only (no sound speed).
// Used by the semi-implicit solver where pressure is handled implicitly.
double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt);

// Acoustic time step: CFL based on |u| + c (velocity + sound speed).
// Used by the explicit solver.
double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt);

} // namespace SemiImplicitFV

#endif // RK_TIME_STEPPING_HPP
