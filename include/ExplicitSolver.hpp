#ifndef EXPLICIT_SOLVER_HPP
#define EXPLICIT_SOLVER_HPP

#include "SimulationConfig.hpp"
#include "RiemannSolver.hpp"
#include "Reconstruction.hpp"
#include "EquationOfState.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;
struct HaloExchange;

/* Forward declare IGR types (being converted in parallel) */
struct IGRSolver;
typedef double GradientTensor[3][3];

struct ExplicitSolverWork {
    struct EOSData eos;
    struct ExplicitParams params;
    struct ReconstructorData reconstructor;
    enum RiemannSolverType solverType;
    struct FluxConfig fluxConfig;

    struct HaloExchange* halo;

    /* IGR solver (may be NULL) */
    struct IGRSolver* igrSolver;

    /* RHS storage */
    double* rhsRho;
    double* rhsRhoU;
    double* rhsRhoV;
    double* rhsRhoW;
    double* rhsRhoE;

    /* Multi-phase RHS storage: flat arrays, nPhases * totalCells each */
    double* rhsAlphaRho;
    double* rhsAlpha;

    /* Scratch for velocity divergence */
    double* divU;

    /* Velocity gradient tensor array (for IGR) */
    GradientTensor* gradU;

    size_t totalCells;
    int dim;
    int nPhases;

    /* One-shot flag: 0 until the first explicit_step has primed CUDA-aware
     * MPI Allreduce + the dt reduction kernel.  The very first call would
     * otherwise pay multi-second JIT/handshake costs on the timing path. */
    int firstStepDone;
};

void explicit_solver_init(struct ExplicitSolverWork* w,
                          const struct RectilinearMesh* mesh,
                          const struct EOSData* eos,
                          enum RiemannSolverType solverType,
                          struct IGRSolver* igrSolver,
                          const struct SimulationConfig* config);

void explicit_solver_free(struct ExplicitSolverWork* w);

double explicit_step(struct ExplicitSolverWork* w,
                     const struct SimulationConfig* config,
                     const struct RectilinearMesh* mesh,
                     struct SolutionState* state,
                     double targetDt);

#endif /* EXPLICIT_SOLVER_HPP */
