#ifndef IC_PATCH_HPP
#define IC_PATCH_HPP

#include "SimulationConfig.hpp"
#include "EquationOfState.hpp"
#include <cmath>
#include <cstring>

/* Forward declarations - these headers will be included by the .cpp */
struct RectilinearMesh;
struct SolutionState;

enum ICGeometryType {
    IC_GEOM_BOX,
    IC_GEOM_SPHERE,
    IC_GEOM_PLANE,
    IC_GEOM_ANALYTIC
};

struct ICGeometryBox {
    double lo[3];
    double hi[3];
};

struct ICGeometrySphere {
    double center[3];
    double radius;
};

struct ICGeometryPlane {
    double point[3];
    double normal[3];
};

struct ICGeometry {
    enum ICGeometryType type;
    union {
        ICGeometryBox box;
        ICGeometrySphere sphere;
        ICGeometryPlane plane;
    };
    /* For analytic type: sub-geometry for region check. NULL means all cells. */
    ICGeometry* subRegion;
};

int ic_geometry_contains(const ICGeometry* geom, double x, double y, double z);

struct ICState {
    double rho;
    double u, v, w;
    double p;
    double sigma;
    double alpha[MAX_PHASES];
    double alphaRho[MAX_PHASES];
    int nAlpha;
    int nAlphaRho;
};

ICState ic_state_defaults(void);

struct ICPatch {
    ICGeometry geometry;
    ICState state;
    /* Expression strings for analytic patches: parallel arrays */
    char** exprNames;
    char** exprStrings;
    int nExpressions;
};

void ic_patch_init(ICPatch* patch);
void ic_patch_free(ICPatch* patch);

void apply_initial_conditions(
    const RectilinearMesh* mesh,
    SolutionState* state,
    const EOSData* eos,
    const SimulationConfig* config,
    const ICState* defaultState,
    const ICPatch* patches,
    int nPatches);

#endif /* IC_PATCH_HPP */
