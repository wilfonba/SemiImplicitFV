#ifndef RECTILINEAR_MESH_HPP
#define RECTILINEAR_MESH_HPP

#include "SolutionState.hpp"
#include "SimulationConfig.hpp"
#include <stddef.h>

struct HaloExchange;

enum BoundaryCondition {
    BC_SYMMETRY,
    BC_OUTFLOW,
    BC_PERIODIC,
    BC_SLIP_WALL,
    BC_NO_SLIP_WALL
};

#define XLOW  0
#define XHIGH 1
#define YLOW  2
#define YHIGH 3
#define ZLOW  4
#define ZHIGH 5

struct RectilinearMesh {
    int dim;
    int nx, ny, nz;
    int nGhost;
    int ngx, ngy, ngz;

    double* xNodesExt;
    int xNodesExtSize;
    double* yNodesExt;
    int yNodesExtSize;
    double* zNodesExt;
    int zNodesExtSize;

    enum BoundaryCondition bc[6];
};

/* Construct from node arrays.  xNodes has (nx+1) entries, etc. */
void mesh_init(struct RectilinearMesh* m,
               const struct SimulationConfig* config,
               const double* xNodes, int nxNodes,
               const double* yNodes, int nyNodes,
               const double* zNodes, int nzNodes);

/* Convenience: uniform grid */
void mesh_init_uniform(struct RectilinearMesh* m,
                       const struct SimulationConfig* config,
                       int nx, double xMin, double xMax,
                       int ny, double yMin, double yMax,
                       int nz, double zMin, double zMax);

void mesh_free(struct RectilinearMesh* m);

void mesh_set_bc(struct RectilinearMesh* m, int face, enum BoundaryCondition bc);

/* ---------------------------------------------------------------------------
   Inline geometry accessors
   --------------------------------------------------------------------------- */

static inline int mesh_nxTotal(const struct RectilinearMesh* m) {
    return m->nx + 2 * m->ngx;
}
static inline int mesh_nyTotal(const struct RectilinearMesh* m) {
    return m->ny + 2 * m->ngy;
}
static inline int mesh_nzTotal(const struct RectilinearMesh* m) {
    return m->nz + 2 * m->ngz;
}
static inline size_t mesh_total_cells(const struct RectilinearMesh* m) {
    return (size_t)mesh_nxTotal(m) * mesh_nyTotal(m) * mesh_nzTotal(m);
}
static inline size_t mesh_index(const struct RectilinearMesh* m,
                                int i, int j, int k) {
    return (size_t)((i + m->ngx) +
                    mesh_nxTotal(m) * ((j + m->ngy) +
                                       mesh_nyTotal(m) * (k + m->ngz)));
}
static inline double mesh_dx(const struct RectilinearMesh* m, int i) {
    return m->xNodesExt[i + m->ngx + 1] - m->xNodesExt[i + m->ngx];
}
static inline double mesh_dy(const struct RectilinearMesh* m, int j) {
    return m->yNodesExt[j + m->ngy + 1] - m->yNodesExt[j + m->ngy];
}
static inline double mesh_dz(const struct RectilinearMesh* m, int k) {
    return m->zNodesExt[k + m->ngz + 1] - m->zNodesExt[k + m->ngz];
}
static inline double mesh_cell_volume(const struct RectilinearMesh* m,
                                      int i, int j, int k) {
    return mesh_dx(m, i) * mesh_dy(m, j) * mesh_dz(m, k);
}
static inline double mesh_cellCentroidX(const struct RectilinearMesh* m, int i) {
    return 0.5 * (m->xNodesExt[i + m->ngx] + m->xNodesExt[i + m->ngx + 1]);
}
static inline double mesh_cellCentroidY(const struct RectilinearMesh* m, int j) {
    return 0.5 * (m->yNodesExt[j + m->ngy] + m->yNodesExt[j + m->ngy + 1]);
}
static inline double mesh_cellCentroidZ(const struct RectilinearMesh* m, int k) {
    return 0.5 * (m->zNodesExt[k + m->ngz] + m->zNodesExt[k + m->ngz + 1]);
}
static inline double mesh_nodeX(const struct RectilinearMesh* m, int i) {
    return m->xNodesExt[i + m->ngx];
}
static inline double mesh_nodeY(const struct RectilinearMesh* m, int j) {
    return m->yNodesExt[j + m->ngy];
}
static inline double mesh_nodeZ(const struct RectilinearMesh* m, int k) {
    return m->zNodesExt[k + m->ngz];
}
static inline double mesh_faceAreaX(const struct RectilinearMesh* m,
                                    int j, int k) {
    return mesh_dy(m, j) * mesh_dz(m, k);
}
static inline double mesh_faceAreaY(const struct RectilinearMesh* m,
                                    int i, int k) {
    return mesh_dx(m, i) * mesh_dz(m, k);
}
static inline double mesh_faceAreaZ(const struct RectilinearMesh* m,
                                    int i, int j) {
    return mesh_dx(m, i) * mesh_dy(m, j);
}

/* ---------------------------------------------------------------------------
   Ghost fill functions
   --------------------------------------------------------------------------- */

/* Fill ghost cells for all active dimensions based on current BCs. */
void mesh_apply_bcs(const struct RectilinearMesh* m,
                    struct SolutionState* state,
                    enum VarSet varSet);

/* MPI-aware ghost fill: halo exchange + physical BCs on boundary faces only. */
void mesh_apply_bcs_mpi(const struct RectilinearMesh* m,
                        struct SolutionState* state,
                        enum VarSet varSet,
                        struct HaloExchange* halo);

/* Fill ghost cells for a single scalar field (size totalCells). */
void mesh_fill_scalar_ghosts(const struct RectilinearMesh* m,
                             double* field);

/* MPI-aware scalar ghost fill. */
void mesh_fill_scalar_ghosts_mpi(const struct RectilinearMesh* m,
                                 double* field,
                                 struct HaloExchange* halo);

#endif /* RECTILINEAR_MESH_HPP */
