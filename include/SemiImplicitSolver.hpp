#ifndef SEMI_IMPLICIT_SOLVER_HPP
#define SEMI_IMPLICIT_SOLVER_HPP

#include "SimulationConfig.hpp"
#include "RiemannSolver.hpp"
#include "Reconstruction.hpp"
#include "EquationOfState.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct SolutionState;
struct HaloExchange;
struct PressureSolverData;
struct IGRSolver;
typedef double GradientTensor[3][3];

struct SemiImplicitSolverWork {
    struct EOSData eos;
    struct SemiImplicitParams params;
    struct ReconstructorData reconstructor;
    enum RiemannSolverType solverType;
    struct FluxConfig fluxConfig;
    double gamma;
    double pInf;

    struct HaloExchange* halo;
    struct PressureSolverData* pressureSolver;
    struct IGRSolver* igrSolver;

    int lastPressureIters;

    /* Pressure solver workspace */
    double* pressureRhs;
    double* pressure;
    double* divUstar;

    /* RHS storage */
    double* rhsRho;
    double* rhsRhoU;
    double* rhsRhoV;
    double* rhsRhoW;
    double* rhsRhoE;
    double* rhsPadvected;

    /* Multi-phase RHS: flat arrays, nPhases * totalCells each */
    double* rhsAlphaRho;
    double* rhsAlpha;

    /* Scratch for velocity divergence */
    double* divU;

    /* Pre-allocated scratch arrays for multi-phase per-cell operations */
    double scratchAlphas[MAX_PHASES];
    double scratchAlphaRhos[MAX_PHASES];

    /* Velocity gradient tensor array (for IGR) */
    GradientTensor* gradU;

    size_t totalCells;
    int dim;
    int nPhases;
};

void semi_implicit_solver_init(struct SemiImplicitSolverWork* w,
                               const struct RectilinearMesh* mesh,
                               const struct EOSData* eos,
                               enum RiemannSolverType solverType,
                               struct PressureSolverData* pressureSolver,
                               struct IGRSolver* igrSolver,
                               const struct SimulationConfig* config);

void semi_implicit_solver_free(struct SemiImplicitSolverWork* w);

double semi_implicit_step(struct SemiImplicitSolverWork* w,
                          const struct SimulationConfig* config,
                          const struct RectilinearMesh* mesh,
                          struct SolutionState* state,
                          double targetDt);

#endif /* SEMI_IMPLICIT_SOLVER_HPP */
