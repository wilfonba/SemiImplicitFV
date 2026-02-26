#include "SolutionState.hpp"
#include "RectilinearMesh.hpp"
#include "EquationOfState.hpp"
#include "MixtureEOS.hpp"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
   Helper: allocate a zero-filled double array, or return NULL if n == 0
   --------------------------------------------------------------------------- */

static double* alloc_field(size_t n)
{
    if (n == 0) return NULL;
    double* p = (double*)calloc(n, sizeof(double));
    return p;
}

static void free_field(double** p)
{
    free(*p);
    *p = NULL;
}

/* ---------------------------------------------------------------------------
   Init / Free
   --------------------------------------------------------------------------- */

void solution_state_init(struct SolutionState* s, size_t totalCells,
                         const struct SimulationConfig* config)
{
    memset(s, 0, sizeof(*s));
    s->dim = config->dim;
    s->totalCells = totalCells;
    s->nPhases = config->multiPhaseParams.nPhases;

    int isMultiPhase = config_is_multi_phase(config);

    /* Conservative variables */
    s->rho  = alloc_field(totalCells);
    s->rhoU = alloc_field(totalCells);
    s->rhoV = (s->dim >= 2) ? alloc_field(totalCells) : NULL;
    s->rhoW = (s->dim >= 3) ? alloc_field(totalCells) : NULL;
    s->rhoE = alloc_field(totalCells);

    /* Primitive variables */
    s->velU  = alloc_field(totalCells);
    s->velV  = (s->dim >= 2) ? alloc_field(totalCells) : NULL;
    s->velW  = (s->dim >= 3) ? alloc_field(totalCells) : NULL;
    s->pres  = alloc_field(totalCells);
    s->sigma = alloc_field(totalCells);

    /* Semi-implicit star states */
    if (config->semiImplicit) {
        s->rhoUStar = alloc_field(totalCells);
        s->rhoVStar = (s->dim >= 2) ? alloc_field(totalCells) : NULL;
        s->rhoWStar = (s->dim >= 3) ? alloc_field(totalCells) : NULL;
        s->rhoEstar  = alloc_field(totalCells);
        s->pAdvected = alloc_field(totalCells);
        s->rhoc2     = alloc_field(totalCells);
        s->divUStar  = alloc_field(totalCells);
    }

    /* Auxiliary variable */
    s->aux = alloc_field(totalCells);

    /* Multi-phase fields: flat arrays, nPhases * totalCells */
    if (isMultiPhase) {
        int nPh = s->nPhases;
        s->alpha    = alloc_field((size_t)nPh * totalCells);
        s->alphaRho = alloc_field((size_t)nPh * totalCells);
    }

    /* Backup arrays for multi-stage time stepping */
    if (config->RKOrder > 1) {
        s->rho0  = alloc_field(totalCells);
        s->rhoU0 = alloc_field(totalCells);
        s->rhoV0 = (s->dim >= 2) ? alloc_field(totalCells) : NULL;
        s->rhoW0 = (s->dim >= 3) ? alloc_field(totalCells) : NULL;
        s->rhoE0 = alloc_field(totalCells);
        s->pres0 = alloc_field(totalCells);

        if (isMultiPhase) {
            int nPh = s->nPhases;
            s->alpha0    = alloc_field((size_t)nPh * totalCells);
            s->alphaRho0 = alloc_field((size_t)nPh * totalCells);
        }
    }
}

void solution_state_free(struct SolutionState* s)
{
    free_field(&s->rho);
    free_field(&s->rhoU);
    free_field(&s->rhoV);
    free_field(&s->rhoW);
    free_field(&s->rhoE);

    free_field(&s->velU);
    free_field(&s->velV);
    free_field(&s->velW);
    free_field(&s->pres);
    free_field(&s->sigma);

    free_field(&s->rho0);
    free_field(&s->rhoU0);
    free_field(&s->rhoV0);
    free_field(&s->rhoW0);
    free_field(&s->rhoE0);
    free_field(&s->pres0);

    free_field(&s->rhoUStar);
    free_field(&s->rhoVStar);
    free_field(&s->rhoWStar);
    free_field(&s->rhoEstar);
    free_field(&s->pAdvected);
    free_field(&s->rhoc2);
    free_field(&s->divUStar);

    free_field(&s->alpha);
    free_field(&s->alphaRho);
    free_field(&s->alpha0);
    free_field(&s->alphaRho0);

    free_field(&s->aux);

    s->totalCells = 0;
}

/* ---------------------------------------------------------------------------
   Copy cell helpers
   --------------------------------------------------------------------------- */

void state_copy_cell_P(struct SolutionState* s, size_t dst, size_t src,
                       double sU, double sV, double sW)
{
    size_t tc = s->totalCells;

    s->rho[dst]  = s->rho[src];
    s->velU[dst] = sU * s->velU[src];

    if (s->dim >= 2) s->velV[dst] = sV * s->velV[src];
    if (s->dim >= 3) s->velW[dst] = sW * s->velW[src];

    s->pres[dst]  = s->pres[src];
    s->sigma[dst] = s->sigma[src];
    s->aux[dst]   = s->aux[src];

    if (s->alpha) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            s->alphaRho[ph * tc + dst] = s->alphaRho[ph * tc + src];
            s->alpha[ph * tc + dst]    = s->alpha[ph * tc + src];
        }
    }
}

void state_copy_cell_C(struct SolutionState* s, size_t dst, size_t src,
                       double sU, double sV, double sW)
{
    size_t tc = s->totalCells;

    s->rho[dst]  = s->rho[src];
    s->rhoU[dst] = sU * s->rhoU[src];

    if (s->dim >= 2) s->rhoV[dst] = sV * s->rhoV[src];
    if (s->dim >= 3) s->rhoW[dst] = sW * s->rhoW[src];

    s->rhoE[dst]  = s->rhoE[src];
    s->sigma[dst] = s->sigma[src];
    s->aux[dst]   = s->aux[src];

    if (s->alpha) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            s->alphaRho[ph * tc + dst] = s->alphaRho[ph * tc + src];
            s->alpha[ph * tc + dst]    = s->alpha[ph * tc + src];
        }
    }
}

void state_copy_cell(struct SolutionState* s, size_t dst, size_t src,
                     double sU, double sV, double sW)
{
    size_t tc = s->totalCells;

    s->rho[dst]  = s->rho[src];
    s->rhoU[dst] = sU * s->rhoU[src];
    s->velU[dst] = sU * s->velU[src];

    if (s->dim >= 2) {
        s->rhoV[dst] = sV * s->rhoV[src];
        s->velV[dst] = sV * s->velV[src];
    }

    if (s->dim >= 3) {
        s->rhoW[dst] = sW * s->rhoW[src];
        s->velW[dst] = sW * s->velW[src];
    }

    s->rhoE[dst]  = s->rhoE[src];
    s->pres[dst]  = s->pres[src];
    s->sigma[dst] = s->sigma[src];
    s->aux[dst]   = s->aux[src];

    if (s->alpha) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            s->alphaRho[ph * tc + dst] = s->alphaRho[ph * tc + src];
            s->alpha[ph * tc + dst]    = s->alpha[ph * tc + src];
        }
    }
}

/* ---------------------------------------------------------------------------
   Gather / Scatter state bundles
   --------------------------------------------------------------------------- */

ConservativeState state_get_conservative(const struct SolutionState* s, size_t idx)
{
    ConservativeState U;
    U.rho = s->rho[idx];
    U.rhoU[0] = s->rhoU[idx];
    U.rhoU[1] = (s->dim >= 2) ? s->rhoV[idx] : 0.0;
    U.rhoU[2] = (s->dim >= 3) ? s->rhoW[idx] : 0.0;
    U.rhoE = s->rhoE[idx];
    return U;
}

void state_set_conservative(struct SolutionState* s, size_t idx,
                            const ConservativeState* U)
{
    s->rho[idx]  = U->rho;
    s->rhoU[idx] = U->rhoU[0];
    if (s->dim >= 2) s->rhoV[idx] = U->rhoU[1];
    if (s->dim >= 3) s->rhoW[idx] = U->rhoU[2];
    s->rhoE[idx] = U->rhoE;
}

PrimitiveState state_get_primitive(const struct SolutionState* s, size_t idx)
{
    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = s->rho[idx];
    W.u[0] = s->velU[idx];
    W.u[1] = (s->dim >= 2) ? s->velV[idx] : 0.0;
    W.u[2] = (s->dim >= 3) ? s->velW[idx] : 0.0;
    W.p = s->pres[idx];
    W.sigma = s->sigma[idx];
    return W;
}

void state_set_primitive(struct SolutionState* s, size_t idx,
                         const PrimitiveState* W)
{
    s->rho[idx]  = W->rho;
    s->velU[idx] = W->u[0];
    if (s->dim >= 2) s->velV[idx] = W->u[1];
    if (s->dim >= 3) s->velW[idx] = W->u[2];
    s->pres[idx]  = W->p;
    s->sigma[idx] = W->sigma;
}

/* ---------------------------------------------------------------------------
   Conservative <-> Primitive conversion (whole physical domain)
   --------------------------------------------------------------------------- */

void state_cons_to_prim(struct SolutionState* s,
                        const struct RectilinearMesh* mesh,
                        const struct EOSData* eos)
{
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                ConservativeState U = state_get_conservative(s, idx);
                PrimitiveState W = eos_to_primitive(eos, &U);
                s->velU[idx] = W.u[0];
                if (s->dim >= 2) s->velV[idx] = W.u[1];
                if (s->dim >= 3) s->velW[idx] = W.u[2];
                s->pres[idx] = W.p;
            }
        }
    }
}

void state_prim_to_cons(struct SolutionState* s,
                        const struct RectilinearMesh* mesh,
                        const struct EOSData* eos)
{
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                PrimitiveState W = state_get_primitive(s, idx);
                ConservativeState U = eos_to_conservative(eos, &W);
                s->rho[idx]  = U.rho;
                s->rhoU[idx] = U.rhoU[0];
                if (s->dim >= 2) s->rhoV[idx] = U.rhoU[1];
                if (s->dim >= 3) s->rhoW[idx] = U.rhoU[2];
                s->rhoE[idx] = U.rhoE;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   Save conservative cell to backup storage
   --------------------------------------------------------------------------- */

void state_save_conservative_cell(struct SolutionState* s, size_t idx)
{
    size_t tc = s->totalCells;

    s->rho0[idx]  = s->rho[idx];
    s->rhoU0[idx] = s->rhoU[idx];
    if (s->dim >= 2) s->rhoV0[idx] = s->rhoV[idx];
    if (s->dim >= 3) s->rhoW0[idx] = s->rhoW[idx];
    s->rhoE0[idx] = s->rhoE[idx];
    s->pres0[idx] = s->pres[idx];

    if (s->alphaRho0) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            s->alphaRho0[ph * tc + idx] = s->alphaRho[ph * tc + idx];
        }
    }
    if (s->alpha0) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            s->alpha0[ph * tc + idx] = s->alpha[ph * tc + idx];
        }
    }
}

/* ---------------------------------------------------------------------------
   Smoothing: explicit heat equation iterations
   --------------------------------------------------------------------------- */

static void smooth_one_field(double* field, double* scratch,
                             const struct RectilinearMesh* mesh,
                             int dim, double nu, int nIterations,
                             void (*fill_ghosts)(const struct RectilinearMesh*, double*),
                             const struct RectilinearMesh* ghost_mesh)
{
    for (int iter = 0; iter < nIterations; ++iter) {
        fill_ghosts(ghost_mesh, field);
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double lap = 0.0;

                    size_t xm = mesh_index(mesh, i - 1, j, k);
                    size_t xp = mesh_index(mesh, i + 1, j, k);
                    lap += field[xm] + 2.0 * field[idx] + field[xp];

                    if (dim >= 2) {
                        size_t ym = mesh_index(mesh, i, j - 1, k);
                        size_t yp = mesh_index(mesh, i, j + 1, k);
                        lap += field[ym] + 2.0 * field[idx] + field[yp];
                    }
                    if (dim >= 3) {
                        size_t zm = mesh_index(mesh, i, j, k - 1);
                        size_t zp = mesh_index(mesh, i, j, k + 1);
                        lap += field[zm] + 2.0 * field[idx] + field[zp];
                    }

                    scratch[idx] = nu * lap;
                }
            }
        }
        /* Copy result back */
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    field[idx] = scratch[idx];
                }
            }
        }
    }
}

void state_smooth(struct SolutionState* s,
                  const struct RectilinearMesh* mesh,
                  int nIterations)
{
    int dim = s->dim;
    double nu = 1.0 / (4.0 * dim);
    size_t tc = s->totalCells;

    /* Use aux as scratch */
    #define SMOOTH(fld) smooth_one_field(fld, s->aux, mesh, dim, nu, \
                                         nIterations, mesh_fill_scalar_ghosts, mesh)

    /* Smooth conservative variables */
    SMOOTH(s->rho);
    SMOOTH(s->rhoU);
    if (dim >= 2) SMOOTH(s->rhoV);
    if (dim >= 3) SMOOTH(s->rhoW);
    SMOOTH(s->rhoE);

    /* Smooth primitive variables to keep them consistent */
    SMOOTH(s->velU);
    if (dim >= 2) SMOOTH(s->velV);
    if (dim >= 3) SMOOTH(s->velW);
    SMOOTH(s->pres);

    /* Smooth multi-phase fields */
    if (s->alpha) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            SMOOTH(s->alphaRho + ph * tc);
            SMOOTH(s->alpha + ph * tc);
        }
    }

    #undef SMOOTH
}

/* ---------------------------------------------------------------------------
   MPI-aware smoothing
   --------------------------------------------------------------------------- */

#include "HaloExchange.hpp"

/* We need a wrapper that matches the fill_ghosts function pointer signature
   but uses MPI.  Since the function-pointer approach cannot carry extra state
   for the halo object, we use a different implementation approach for MPI
   smoothing: inline the loop directly. */

static void smooth_field_inline(double* field, double* scratch,
                                const struct RectilinearMesh* mesh,
                                int dim, double nu, int nIterations,
                                struct HaloExchange* halo)
{
    for (int iter = 0; iter < nIterations; ++iter) {
        mesh_fill_scalar_ghosts_mpi(mesh, field, halo);
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    double lap = 0.0;

                    size_t xm = mesh_index(mesh, i - 1, j, k);
                    size_t xp = mesh_index(mesh, i + 1, j, k);
                    lap += field[xm] + 2.0 * field[idx] + field[xp];

                    if (dim >= 2) {
                        size_t ym = mesh_index(mesh, i, j - 1, k);
                        size_t yp = mesh_index(mesh, i, j + 1, k);
                        lap += field[ym] + 2.0 * field[idx] + field[yp];
                    }
                    if (dim >= 3) {
                        size_t zm = mesh_index(mesh, i, j, k - 1);
                        size_t zp = mesh_index(mesh, i, j, k + 1);
                        lap += field[zm] + 2.0 * field[idx] + field[zp];
                    }

                    scratch[idx] = nu * lap;
                }
            }
        }
        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    size_t idx = mesh_index(mesh, i, j, k);
                    field[idx] = scratch[idx];
                }
            }
        }
    }
}

void state_smooth_mpi(struct SolutionState* s,
                      const struct RectilinearMesh* mesh,
                      int nIterations,
                      struct HaloExchange* halo)
{
    int dim = s->dim;
    double nu = 1.0 / (4.0 * dim);
    size_t tc = s->totalCells;

    #define SMOOTH_MPI(fld) smooth_field_inline(fld, s->aux, mesh, dim, nu, \
                                                nIterations, halo)

    SMOOTH_MPI(s->rho);
    SMOOTH_MPI(s->rhoU);
    if (dim >= 2) SMOOTH_MPI(s->rhoV);
    if (dim >= 3) SMOOTH_MPI(s->rhoW);
    SMOOTH_MPI(s->rhoE);

    SMOOTH_MPI(s->velU);
    if (dim >= 2) SMOOTH_MPI(s->velV);
    if (dim >= 3) SMOOTH_MPI(s->velW);
    SMOOTH_MPI(s->pres);

    if (s->alpha) {
        for (int ph = 0; ph < s->nPhases; ++ph) {
            SMOOTH_MPI(s->alphaRho + ph * tc);
            SMOOTH_MPI(s->alpha + ph * tc);
        }
    }

    #undef SMOOTH_MPI
}

/* ---------------------------------------------------------------------------
   Reconcile multi-phase thermodynamics after smoothing
   --------------------------------------------------------------------------- */

static void reconcile_multi_phase(struct SolutionState* s,
                                  const struct RectilinearMesh* mesh,
                                  const struct SimulationConfig* config)
{
    const struct MultiPhaseParams* mp = &config->multiPhaseParams;
    int nPhases = mp->nPhases;
    int dim = config->dim;
    size_t tc = s->totalCells;

    double alphas[MAX_PHASES];

    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);

                /* Clamp and normalize alpha */
                double alphaSum = 0.0;
                for (int ph = 0; ph < nPhases; ++ph) {
                    double a = s->alpha[ph * tc + idx];
                    if (a < mp->alphaMin) a = mp->alphaMin;
                    s->alpha[ph * tc + idx] = a;
                    alphaSum += a;
                }
                for (int ph = 0; ph < nPhases; ++ph)
                    s->alpha[ph * tc + idx] /= alphaSum;

                /* Recompute rho from smoothed alphaRho */
                double rhoNew = 0.0;
                for (int ph = 0; ph < nPhases; ++ph) {
                    double ar = s->alphaRho[ph * tc + idx];
                    if (ar < 1e-14) ar = 1e-14;
                    s->alphaRho[ph * tc + idx] = ar;
                    rhoNew += ar;
                }
                s->rho[idx] = rhoNew;

                /* Recompute momentum from smoothed velocity */
                s->rhoU[idx] = rhoNew * s->velU[idx];
                if (dim >= 2) s->rhoV[idx] = rhoNew * s->velV[idx];
                if (dim >= 3) s->rhoW[idx] = rhoNew * s->velW[idx];

                /* Gather alphas for this cell */
                for (int ph = 0; ph < nPhases; ++ph)
                    alphas[ph] = s->alpha[ph * tc + idx];

                /* Recompute rhoE from smoothed pressure and alpha via MixtureEOS */
                double ke = 0.5 * rhoNew * s->velU[idx] * s->velU[idx];
                if (dim >= 2) ke += 0.5 * rhoNew * s->velV[idx] * s->velV[idx];
                if (dim >= 3) ke += 0.5 * rhoNew * s->velW[idx] * s->velW[idx];

                s->rhoE[idx] = mixture_total_energy(
                    rhoNew, s->pres[idx], alphas,
                    nPhases, ke, mp->phases);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   Config-aware smoothing
   --------------------------------------------------------------------------- */

void state_smooth_config(struct SolutionState* s,
                         const struct RectilinearMesh* mesh,
                         int nIterations,
                         const struct SimulationConfig* config)
{
    if (!config_is_multi_phase(config)) {
        state_smooth(s, mesh, nIterations);
        return;
    }

    int dim = s->dim;
    double nu = 1.0 / (4.0 * dim);
    size_t tc = s->totalCells;

    #define SMOOTH(fld) smooth_one_field(fld, s->aux, mesh, dim, nu, \
                                         nIterations, mesh_fill_scalar_ghosts, mesh)

    /* Smooth only the independent primitive fields */
    for (int ph = 0; ph < s->nPhases; ++ph) SMOOTH(s->alpha + ph * tc);
    for (int ph = 0; ph < s->nPhases; ++ph) SMOOTH(s->alphaRho + ph * tc);
    SMOOTH(s->pres);
    SMOOTH(s->velU);
    if (dim >= 2) SMOOTH(s->velV);
    if (dim >= 3) SMOOTH(s->velW);

    #undef SMOOTH

    /* Recompute conservative variables from smoothed primitives */
    reconcile_multi_phase(s, mesh, config);
}

void state_smooth_config_mpi(struct SolutionState* s,
                             const struct RectilinearMesh* mesh,
                             int nIterations,
                             struct HaloExchange* halo,
                             const struct SimulationConfig* config)
{
    if (!config_is_multi_phase(config)) {
        state_smooth_mpi(s, mesh, nIterations, halo);
        return;
    }

    int dim = s->dim;
    double nu = 1.0 / (4.0 * dim);
    size_t tc = s->totalCells;

    #define SMOOTH_MPI(fld) smooth_field_inline(fld, s->aux, mesh, dim, nu, \
                                                nIterations, halo)

    /* Smooth only the independent primitive fields */
    for (int ph = 0; ph < s->nPhases; ++ph) SMOOTH_MPI(s->alpha + ph * tc);
    for (int ph = 0; ph < s->nPhases; ++ph) SMOOTH_MPI(s->alphaRho + ph * tc);
    SMOOTH_MPI(s->pres);
    SMOOTH_MPI(s->velU);
    if (dim >= 2) SMOOTH_MPI(s->velV);
    if (dim >= 3) SMOOTH_MPI(s->velW);

    #undef SMOOTH_MPI

    /* Recompute conservative variables from smoothed primitives */
    reconcile_multi_phase(s, mesh, config);
}
