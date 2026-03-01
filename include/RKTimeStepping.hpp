#ifndef RK_TIME_STEPPING_HPP
#define RK_TIME_STEPPING_HPP

#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"
#include <mpi.h>
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;
struct Runtime;
struct VTKSession;

/* Advective time step: CFL based on material velocity only (no sound speed). */
double computeAdvectiveTimeStep(const struct RectilinearMesh* mesh,
                                const struct SolutionState* state,
                                double cfl, double maxDt);

/* Acoustic time step: CFL based on |u| + c. */
double computeAcousticTimeStep(const struct RectilinearMesh* mesh,
                               const struct SolutionState* state,
                               const struct EOSData* eos,
                               double cfl, double maxDt);

/* Acoustic time step with config for multi-phase sound speed via Wood's formula. */
double computeAcousticTimeStep_config(const struct RectilinearMesh* mesh,
                                      const struct SolutionState* state,
                                      const struct EOSData* eos,
                                      const struct SimulationConfig* config,
                                      double cfl, double maxDt);

/* MPI-aware variants (global reduction). */
double computeAdvectiveTimeStep_mpi(const struct RectilinearMesh* mesh,
                                    const struct SolutionState* state,
                                    double cfl, double maxDt,
                                    MPI_Comm comm);

double computeAcousticTimeStep_mpi(const struct RectilinearMesh* mesh,
                                   const struct SolutionState* state,
                                   const struct EOSData* eos,
                                   double cfl, double maxDt,
                                   MPI_Comm comm);

double computeAcousticTimeStep_config_mpi(const struct RectilinearMesh* mesh,
                                          const struct SolutionState* state,
                                          const struct EOSData* eos,
                                          const struct SimulationConfig* config,
                                          double cfl, double maxDt,
                                          MPI_Comm comm);

/* Viscous time step */
double computeViscousDt(const struct RectilinearMesh* mesh,
                        const struct SolutionState* state,
                        double mu, double cfl, double maxDt);

double computeViscousDt_mpi(const struct RectilinearMesh* mesh,
                            const struct SolutionState* state,
                            double mu, double cfl, double maxDt,
                            MPI_Comm comm);

double computeViscousDt_config(const struct RectilinearMesh* mesh,
                               const struct SolutionState* state,
                               const struct SimulationConfig* config,
                               double cfl, double maxDt);

double computeViscousDt_config_mpi(const struct RectilinearMesh* mesh,
                                   const struct SolutionState* state,
                                   const struct SimulationConfig* config,
                                   double cfl, double maxDt,
                                   MPI_Comm comm);

/* Capillary time step */
double computeCapillaryDt(const struct RectilinearMesh* mesh,
                          const struct SolutionState* state,
                          double sigma, double cfl, double maxDt);

double computeCapillaryDt_mpi(const struct RectilinearMesh* mesh,
                              const struct SolutionState* state,
                              double sigma, double cfl, double maxDt,
                              MPI_Comm comm);

/* ---- Time loop ---- */

/* Step function: takes targetDt and context, returns actual dt used. */
typedef double (*StepFn)(double targetDt, void* ctx);

/* Acoustic Dt callback: returns acoustic dt with CFL=1. */
typedef double (*AcousticDtFn)(void* ctx);

struct TimeLoopParams {
    double endTime;
    double outputInterval;
    int printInterval;
    int checkNaN;
    AcousticDtFn acousticDtFn;
    void* acousticDtCtx;
    int checkpoint;
    double startTime;
    int restarting;
};

struct TimeLoopParams time_loop_params_defaults(void);

void run_time_loop(
    struct Runtime* rt,
    struct SimulationConfig* config,
    const struct RectilinearMesh* mesh,
    struct SolutionState* state,
    struct VTKSession* vtk,
    StepFn stepFn,
    void* stepCtx,
    const struct TimeLoopParams* params);

#endif /* RK_TIME_STEPPING_HPP */
