#ifndef RK_TIME_STEPPING_HPP
#define RK_TIME_STEPPING_HPP

#include "SimulationConfig.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "EquationOfState.hpp"

#include <functional>
#include <mpi.h>

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

// Acoustic time step with config for multi-phase sound speed via Wood's formula.
double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               const SimulationConfig& config,
                               double cfl, double maxDt);

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

// MPI-aware acoustic time step with config for multi-phase.
double computeAcousticTimeStep(const RectilinearMesh& mesh,
                               const SolutionState& state,
                               const EquationOfState& eos,
                               const SimulationConfig& config,
                               double cfl, double maxDt,
                               MPI_Comm comm);

// Viscous time step: dt <= cfl * dx_min^2 / (2 * dim * nu), nu = mu/rho.
double computeViscousDt(const RectilinearMesh& mesh,
                        const SolutionState& state,
                        double mu, double cfl, double maxDt);

double computeViscousDt(const RectilinearMesh& mesh,
                        const SolutionState& state,
                        double mu, double cfl, double maxDt,
                        MPI_Comm comm);

// Capillary time step: dt <= cfl * sqrt(rho * dx_min^3 / sigma).
double computeCapillaryDt(const RectilinearMesh& mesh,
                          const SolutionState& state,
                          double sigma, double cfl, double maxDt);

double computeCapillaryDt(const RectilinearMesh& mesh,
                          const SolutionState& state,
                          double sigma, double cfl, double maxDt,
                          MPI_Comm comm);

// ---- Time loop ----

struct TimeLoopParams {
    double endTime;
    double outputInterval;
    int printInterval = 1;
    bool checkNaN = true;           // Abort if NaNs detected at I/O steps
    // If set, prints acoustic CFL = dt / acousticDtFn() at each print step.
    // The callback should return the acoustic time step with CFL=1.
    std::function<double()> acousticDtFn;
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
