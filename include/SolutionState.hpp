#ifndef SOLUTION_STATE_HPP
#define SOLUTION_STATE_HPP

#include "State.hpp"
#include "SimulationConfig.hpp"
#include <stddef.h>

struct RectilinearMesh;
struct HaloExchange;

enum VarSet {
    VARSET_CONS,
    VARSET_PRIM,
    VARSET_SIGMA
};

struct SolutionState {
    int dim;
    size_t totalCells;
    int nPhases;

    /* Conservative variables */
    double* rho;
    double* rhoU;
    double* rhoV;   /* NULL if dim < 2 */
    double* rhoW;   /* NULL if dim < 3 */
    double* rhoE;

    /* Primitive variables */
    double* velU;
    double* velV;    /* NULL if dim < 2 */
    double* velW;    /* NULL if dim < 3 */
    double* pres;
    double* sigma;

    /* Backup conservative variables (NULL if RKOrder == 1) */
    double* rho0;
    double* rhoU0;
    double* rhoV0;   /* NULL if dim < 2 or RKOrder == 1 */
    double* rhoW0;   /* NULL if dim < 3 or RKOrder == 1 */
    double* rhoE0;
    double* pres0;

    /* Semi-implicit star states (NULL if !semiImplicit) */
    double* rhoUStar;
    double* rhoVStar;  /* NULL if dim < 2 or !semiImplicit */
    double* rhoWStar;  /* NULL if dim < 3 or !semiImplicit */
    double* rhoEstar;
    double* pAdvected;
    double* rhoc2;
    double* divUStar;

    /* Multi-phase fields: flat arrays, size nPhases * totalCells
       Access as alpha[phase * totalCells + cell]
       NULL when single-phase (nPhases < 2) */
    double* alpha;
    double* alphaRho;

    /* Multi-phase RK backup (NULL if nPhases < 2 or RKOrder == 1) */
    double* alpha0;
    double* alphaRho0;

    /* Auxiliary variable */
    double* aux;
};

void solution_state_init(struct SolutionState* s, size_t totalCells,
                         const struct SimulationConfig* config);
void solution_state_free(struct SolutionState* s);

/* Gather / scatter state bundles */
ConservativeState state_get_conservative(const struct SolutionState* s, size_t idx);
void state_set_conservative(struct SolutionState* s, size_t idx,
                            const ConservativeState* U);
PrimitiveState state_get_primitive(const struct SolutionState* s, size_t idx);
void state_set_primitive(struct SolutionState* s, size_t idx,
                         const PrimitiveState* W);

/* Copy cell data between flat indices with velocity sign multipliers */
void state_copy_cell(struct SolutionState* s, size_t dst, size_t src,
                     double sU, double sV, double sW);
void state_copy_cell_P(struct SolutionState* s, size_t dst, size_t src,
                       double sU, double sV, double sW);
void state_copy_cell_C(struct SolutionState* s, size_t dst, size_t src,
                       double sU, double sV, double sW);

/* Save conservative variables for a single cell to backup storage */
void state_save_conservative_cell(struct SolutionState* s, size_t idx);

/* Convert between conservative and primitive for entire physical domain */
void state_cons_to_prim(struct SolutionState* s,
                        const struct RectilinearMesh* mesh,
                        const struct EOSData* eos);
void state_prim_to_cons(struct SolutionState* s,
                        const struct RectilinearMesh* mesh,
                        const struct EOSData* eos);

/* Smooth all fields using explicit heat equation iterations */
void state_smooth(struct SolutionState* s,
                  const struct RectilinearMesh* mesh,
                  int nIterations);

/* MPI-aware smoothing */
void state_smooth_mpi(struct SolutionState* s,
                      const struct RectilinearMesh* mesh,
                      int nIterations,
                      struct HaloExchange* halo);

/* Config-aware smoothing (multi-phase reconciliation) */
void state_smooth_config(struct SolutionState* s,
                         const struct RectilinearMesh* mesh,
                         int nIterations,
                         const struct SimulationConfig* config);

/* Config-aware + MPI-aware smoothing */
void state_smooth_config_mpi(struct SolutionState* s,
                             const struct RectilinearMesh* mesh,
                             int nIterations,
                             struct HaloExchange* halo,
                             const struct SimulationConfig* config);

#endif /* SOLUTION_STATE_HPP */
