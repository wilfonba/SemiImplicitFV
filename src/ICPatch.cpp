#include "ICPatch.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include "MixtureEOS.hpp"
#include "ExpressionEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

int ic_geometry_contains(const ICGeometry* geom, double x, double y, double z) {
    switch (geom->type) {
        case IC_GEOM_BOX:
            return x >= geom->box.lo[0] && x <= geom->box.hi[0] &&
                   y >= geom->box.lo[1] && y <= geom->box.hi[1] &&
                   z >= geom->box.lo[2] && z <= geom->box.hi[2];
        case IC_GEOM_SPHERE: {
            double dx = x - geom->sphere.center[0];
            double dy = y - geom->sphere.center[1];
            double dz = z - geom->sphere.center[2];
            return dx*dx + dy*dy + dz*dz <= geom->sphere.radius * geom->sphere.radius;
        }
        case IC_GEOM_PLANE: {
            double dot = (x - geom->plane.point[0]) * geom->plane.normal[0]
                       + (y - geom->plane.point[1]) * geom->plane.normal[1]
                       + (z - geom->plane.point[2]) * geom->plane.normal[2];
            return dot >= 0.0;
        }
        case IC_GEOM_ANALYTIC:
            if (geom->subRegion)
                return ic_geometry_contains(geom->subRegion, x, y, z);
            return 1; /* no sub-region means all cells */
    }
    return 0;
}

ICState ic_state_defaults(void) {
    ICState st;
    memset(&st, 0, sizeof(st));
    st.rho = 1.0;
    st.p   = 1.0;
    return st;
}

void ic_patch_init(ICPatch* patch) {
    memset(patch, 0, sizeof(*patch));
}

void ic_patch_free(ICPatch* patch) {
    if (patch->exprNames) {
        for (int i = 0; i < patch->nExpressions; ++i) {
            free(patch->exprNames[i]);
            free(patch->exprStrings[i]);
        }
        free(patch->exprNames);
        free(patch->exprStrings);
    }
    if (patch->geometry.subRegion) {
        free(patch->geometry.subRegion);
        patch->geometry.subRegion = NULL;
    }
    patch->exprNames = NULL;
    patch->exprStrings = NULL;
    patch->nExpressions = 0;
}

/* Helper: apply an ICState to a single cell */
static void applyCellState(
    size_t idx,
    const ICState* st,
    SolutionState* state,
    const EOSData* eos,
    const SimulationConfig* config)
{
    PrimitiveState W;
    memset(&W, 0, sizeof(W));
    W.rho = st->rho;
    W.u[0] = st->u;
    W.u[1] = st->v;
    W.u[2] = st->w;
    W.p = st->p;
    W.sigma = st->sigma;
    W.T = eos_temperature(eos, &W);

    state_set_primitive(state, idx, &W);

    if (config_is_multi_phase(config) && st->nAlpha > 0) {
        int nPhases = config->multiPhaseParams.nPhases;
        size_t tc = state->totalCells;
        double rho = 0.0;
        for (int ph = 0; ph < nPhases; ++ph) {
            state->alpha[ph * tc + idx] = st->alpha[ph];
            if (st->nAlphaRho > 0) {
                state->alphaRho[ph * tc + idx] = st->alphaRho[ph];
            } else {
                state->alphaRho[ph * tc + idx] = st->alpha[ph] * st->rho;
            }
            rho += state->alphaRho[ph * tc + idx];
        }
        state->rho[idx] = rho;

        double ke = 0.5 * rho * (st->u*st->u + st->v*st->v + st->w*st->w);
        state->rhoE[idx] = mixture_total_energy(
            rho, st->p, st->alpha, nPhases, ke,
            config->multiPhaseParams.phases);
        state->rhoU[idx] = rho * st->u;
        if (config->dim >= 2) state->rhoV[idx] = rho * st->v;
        if (config->dim >= 3) state->rhoW[idx] = rho * st->w;
    } else {
        ConservativeState U = eos_to_conservative(eos, &W);
        state_set_conservative(state, idx, &U);
    }
}

void apply_initial_conditions(
    const RectilinearMesh* mesh,
    SolutionState* state,
    const EOSData* eos,
    const SimulationConfig* config,
    const ICState* defaultState,
    const ICPatch* patches,
    int nPatches)
{
    int dim = config->dim;

    /* Step 1: Fill all cells with default state */
    for (int k = 0; k < mesh->nz; ++k) {
        for (int j = 0; j < mesh->ny; ++j) {
            for (int i = 0; i < mesh->nx; ++i) {
                size_t idx = mesh_index(mesh, i, j, k);
                applyCellState(idx, defaultState, state, eos, config);
            }
        }
    }

    /* Step 2: Apply patches in order */
    for (int p = 0; p < nPatches; ++p) {
        const ICPatch* patch = &patches[p];
        int isAnalytic = (patch->nExpressions > 0);

        /* Create expression evaluator if needed */
        ExpressionEvaluator* evaluator = NULL;
        if (isAnalytic) {
            evaluator = new ExpressionEvaluator();
            for (int e = 0; e < patch->nExpressions; ++e) {
                evaluator->addExpression(patch->exprNames[e], patch->exprStrings[e]);
            }
        }

        for (int k = 0; k < mesh->nz; ++k) {
            for (int j = 0; j < mesh->ny; ++j) {
                for (int i = 0; i < mesh->nx; ++i) {
                    double x = mesh_cellCentroidX(mesh, i);
                    double y = (dim >= 2) ? mesh_cellCentroidY(mesh, j) : 0.0;
                    double z = (dim >= 3) ? mesh_cellCentroidZ(mesh, k) : 0.0;

                    if (ic_geometry_contains(&patch->geometry, x, y, z)) {
                        size_t idx = mesh_index(mesh, i, j, k);

                        if (isAnalytic) {
                            ICState evalState = patch->state;
                            evaluator->setCoordinates(x, y, z);

                            for (int e = 0; e < patch->nExpressions; ++e) {
                                const char* name = patch->exprNames[e];
                                double val = evaluator->evaluate(name);
                                if (strcmp(name, "rho") == 0) evalState.rho = val;
                                else if (strcmp(name, "u") == 0) evalState.u = val;
                                else if (strcmp(name, "v") == 0) evalState.v = val;
                                else if (strcmp(name, "w") == 0) evalState.w = val;
                                else if (strcmp(name, "p") == 0) evalState.p = val;
                                else if (strcmp(name, "sigma") == 0) evalState.sigma = val;
                                else if (strncmp(name, "alpha_", 6) == 0) {
                                    int ph = atoi(name + 6);
                                    if (ph >= 0 && ph < evalState.nAlpha)
                                        evalState.alpha[ph] = val;
                                }
                                else if (strncmp(name, "alphaRho_", 9) == 0) {
                                    int ph = atoi(name + 9);
                                    if (ph >= 0 && ph < evalState.nAlphaRho)
                                        evalState.alphaRho[ph] = val;
                                }
                            }
                            applyCellState(idx, &evalState, state, eos, config);
                        } else {
                            applyCellState(idx, &patch->state, state, eos, config);
                        }
                    }
                }
            }
        }

        delete evaluator;
    }
}
