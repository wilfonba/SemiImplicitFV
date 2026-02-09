#ifndef RK_TIME_STEPPING_HPP
#define RK_TIME_STEPPING_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "EquationOfState.hpp"

#include <functional>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace SemiImplicitFV {

class Runtime;
class VTKSession;

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

#ifdef ENABLE_MPI
// MPI-aware advective time step with global reduction.
double computeAdvectiveTimeStep(const RectilinearMesh& mesh,
                                const SolutionState& state,
                                double cfl, double maxDt,
                                MPI_Comm comm);

// MPI-aware acoustic time step with global reduction.
double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               double cfl, double maxDt,
                               MPI_Comm comm);
#endif

// ---- Time loop ----

struct TimeLoopParams {
    double endTime;
    double outputInterval;
    int printInterval = 1;
};

void runTimeLoop(
    Runtime& rt,
    SimulationConfig& config,
    const RectilinearMesh& mesh,
    SolutionState& state,
    VTKSession& vtk,
    const std::function<double(double)>& stepFn,
    const TimeLoopParams& params);

} // namespace SemiImplicitFV

#endif // RK_TIME_STEPPING_HPP
