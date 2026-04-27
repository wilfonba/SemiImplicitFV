#include "RectilinearMesh.hpp"
#include "HaloExchange.hpp"
#include "MPIContext.hpp"
#include "SolutionState.hpp"
#include "NvtxRange.hpp"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
   Build an extended node array from physical nodes by mirroring cell
   widths into the ghost region on each side.
   Returns a malloc'd array of size (nCells + 2*ng + 1).
   --------------------------------------------------------------------------- */

static double* build_extended_nodes(const double* physNodes, int nCells, int ng,
                                    int* outSize)
{
    int nExt = nCells + 2 * ng + 1;
    double* ext = (double*)calloc(nExt, sizeof(double));

    for (int i = 0; i <= nCells; ++i) {
        ext[ng + i] = physNodes[i];
    }

    for (int g = 1; g <= ng; ++g) {
        int mirror = g - 1;
        if (mirror > nCells - 1) mirror = nCells - 1;
        double width = physNodes[mirror + 1] - physNodes[mirror];
        ext[ng - g] = ext[ng - g + 1] - width;
    }

    for (int g = 1; g <= ng; ++g) {
        int mirror = nCells - g;
        if (mirror < 0) mirror = 0;
        double width = physNodes[mirror + 1] - physNodes[mirror];
        ext[ng + nCells + g] = ext[ng + nCells + g - 1] + width;
    }

    *outSize = nExt;
    return ext;
}

/* ---------------------------------------------------------------------------
   Construction
   --------------------------------------------------------------------------- */

void mesh_init(struct RectilinearMesh* m,
               const struct SimulationConfig* config,
               const double* xNodes, int nxNodes,
               const double* yNodes, int nyNodes,
               const double* zNodes, int nzNodes)
{
    m->dim = config->dim;
    m->nGhost = config->nGhost;

    m->nx = nxNodes - 1;
    m->ny = nyNodes - 1;
    m->nz = nzNodes - 1;

    m->ngx = m->nGhost;
    m->ngy = (m->dim >= 2) ? m->nGhost : 0;
    m->ngz = (m->dim >= 3) ? m->nGhost : 0;

    m->xNodesExt = build_extended_nodes(xNodes, m->nx, m->ngx, &m->xNodesExtSize);
    m->yNodesExt = build_extended_nodes(yNodes, m->ny, m->ngy, &m->yNodesExtSize);
    m->zNodesExt = build_extended_nodes(zNodes, m->nz, m->ngz, &m->zNodesExtSize);

    /* Default all boundaries to outflow */
    for (int f = 0; f < 6; ++f)
        m->bc[f] = BC_OUTFLOW;

    /* Mirror node arrays to the device.  They are read by inline accessors
     * (mesh_dx, mesh_cell_volume, ...) inside every flux kernel and are
     * immutable after mesh_init, so `map(to:...)` both allocates and copies
     * in one shot. */
    double* xExt = m->xNodesExt; int xN = m->xNodesExtSize;
    double* yExt = m->yNodesExt; int yN = m->yNodesExtSize;
    double* zExt = m->zNodesExt; int zN = m->zNodesExtSize;
    #pragma omp target enter data map(to: xExt[0:xN], yExt[0:yN], zExt[0:zN])
}

void mesh_init_uniform(struct RectilinearMesh* m,
                       const struct SimulationConfig* config,
                       int nx, double xMin, double xMax,
                       int ny, double yMin, double yMax,
                       int nz, double zMin, double zMax)
{
    /* Build temporary linspace node arrays */
    double* xN = (double*)malloc((nx + 1) * sizeof(double));
    double* yN = (double*)malloc((ny + 1) * sizeof(double));
    double* zN = (double*)malloc((nz + 1) * sizeof(double));

    double hx = (xMax - xMin) / nx;
    for (int i = 0; i <= nx; ++i) xN[i] = xMin + i * hx;

    double hy = (yMax - yMin) / ny;
    for (int j = 0; j <= ny; ++j) yN[j] = yMin + j * hy;

    double hz = (zMax - zMin) / nz;
    for (int k = 0; k <= nz; ++k) zN[k] = zMin + k * hz;

    mesh_init(m, config, xN, nx + 1, yN, ny + 1, zN, nz + 1);

    free(xN);
    free(yN);
    free(zN);
}

void mesh_free(struct RectilinearMesh* m)
{
    double* xExt = m->xNodesExt; int xN = m->xNodesExtSize;
    double* yExt = m->yNodesExt; int yN = m->yNodesExtSize;
    double* zExt = m->zNodesExt; int zN = m->zNodesExtSize;
    if (xExt) {
        #pragma omp target exit data map(delete: xExt[0:xN])
    }
    if (yExt) {
        #pragma omp target exit data map(delete: yExt[0:yN])
    }
    if (zExt) {
        #pragma omp target exit data map(delete: zExt[0:zN])
    }

    free(m->xNodesExt);  m->xNodesExt = NULL;
    free(m->yNodesExt);  m->yNodesExt = NULL;
    free(m->zNodesExt);  m->zNodesExt = NULL;
    m->xNodesExtSize = 0;
    m->yNodesExtSize = 0;
    m->zNodesExtSize = 0;
}

void mesh_set_bc(struct RectilinearMesh* m, int face, enum BoundaryCondition bc)
{
    m->bc[face] = bc;
}

/* ---------------------------------------------------------------------------
   Ghost fill helpers (static, direction-specific)
   --------------------------------------------------------------------------- */

static void fill_ghost_cell(struct SolutionState* state, enum VarSet varSet,
                            size_t ghost, size_t src,
                            double sU, double sV, double sW)
{
    switch (varSet) {
    case VARSET_PRIM:
        state_copy_cell_P(state, ghost, src, sU, sV, sW);
        break;
    case VARSET_CONS:
        state_copy_cell_C(state, ghost, src, sU, sV, sW);
        break;
    default:
        state_copy_cell(state, ghost, src, sU, sV, sW);
        break;
    }
}

static void fill_ghost_x(const struct RectilinearMesh* m,
                          struct SolutionState* state,
                          enum VarSet varSet,
                          int skipLow, int skipHigh)
{
    for (int k = 0; k < m->nz; ++k) {
        for (int j = 0; j < m->ny; ++j) {
            if (!skipLow) {
                for (int g = 1; g <= m->ngx; ++g) {
                    size_t ghost = mesh_index(m, -g, j, k);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[XLOW]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, g - 1, j, k);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, m->nx - g, j, k);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, g - 1, j, k);
                        sU = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, g - 1, j, k);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, 0, j, k);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }

            if (!skipHigh) {
                for (int g = 1; g <= m->ngx; ++g) {
                    size_t ghost = mesh_index(m, m->nx - 1 + g, j, k);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[XHIGH]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, m->nx - g, j, k);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, g - 1, j, k);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, m->nx - g, j, k);
                        sU = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, m->nx - g, j, k);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, m->nx - 1, j, k);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }
        }
    }
}

static void fill_ghost_y(const struct RectilinearMesh* m,
                          struct SolutionState* state,
                          enum VarSet varSet,
                          int skipLow, int skipHigh)
{
    int iLo = -m->ngx;
    int iHi = m->nx + m->ngx;

    for (int k = 0; k < m->nz; ++k) {
        for (int i = iLo; i < iHi; ++i) {
            if (!skipLow) {
                for (int g = 1; g <= m->ngy; ++g) {
                    size_t ghost = mesh_index(m, i, -g, k);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[YLOW]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, i, g - 1, k);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, i, m->ny - g, k);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, i, g - 1, k);
                        sV = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, i, g - 1, k);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, i, 0, k);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }

            if (!skipHigh) {
                for (int g = 1; g <= m->ngy; ++g) {
                    size_t ghost = mesh_index(m, i, m->ny - 1 + g, k);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[YHIGH]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, i, m->ny - g, k);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, i, g - 1, k);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, i, m->ny - g, k);
                        sV = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, i, m->ny - g, k);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, i, m->ny - 1, k);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }
        }
    }
}

static void fill_ghost_z(const struct RectilinearMesh* m,
                          struct SolutionState* state,
                          enum VarSet varSet,
                          int skipLow, int skipHigh)
{
    int iLo = -m->ngx;
    int iHi = m->nx + m->ngx;
    int jLo = -m->ngy;
    int jHi = m->ny + m->ngy;

    for (int j = jLo; j < jHi; ++j) {
        for (int i = iLo; i < iHi; ++i) {
            if (!skipLow) {
                for (int g = 1; g <= m->ngz; ++g) {
                    size_t ghost = mesh_index(m, i, j, -g);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[ZLOW]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, i, j, g - 1);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, i, j, m->nz - g);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, i, j, g - 1);
                        sW = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, i, j, g - 1);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, i, j, 0);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }

            if (!skipHigh) {
                for (int g = 1; g <= m->ngz; ++g) {
                    size_t ghost = mesh_index(m, i, j, m->nz - 1 + g);
                    size_t src;
                    double sU = 1.0, sV = 1.0, sW = 1.0;

                    switch (m->bc[ZHIGH]) {
                    case BC_SYMMETRY:
                        src = mesh_index(m, i, j, m->nz - g);
                        break;
                    case BC_PERIODIC:
                        src = mesh_index(m, i, j, g - 1);
                        break;
                    case BC_SLIP_WALL:
                        src = mesh_index(m, i, j, m->nz - g);
                        sW = -1.0;
                        break;
                    case BC_NO_SLIP_WALL:
                        src = mesh_index(m, i, j, m->nz - g);
                        sU = -1.0; sV = -1.0; sW = -1.0;
                        break;
                    case BC_OUTFLOW:
                    default:
                        src = mesh_index(m, i, j, m->nz - 1);
                        break;
                    }

                    fill_ghost_cell(state, varSet, ghost, src, sU, sV, sW);
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   Public ghost fill: single-process
   --------------------------------------------------------------------------- */

void mesh_apply_bcs(const struct RectilinearMesh* m,
                    struct SolutionState* state,
                    enum VarSet varSet)
{
    fill_ghost_x(m, state, varSet, 0, 0);
    if (m->dim >= 2) fill_ghost_y(m, state, varSet, 0, 0);
    if (m->dim >= 3) fill_ghost_z(m, state, varSet, 0, 0);
}

/* ---------------------------------------------------------------------------
   Scalar ghost fill: single-process
   --------------------------------------------------------------------------- */

void mesh_fill_scalar_ghosts(const struct RectilinearMesh* m, double* field)
{
    /* X-direction (physical j,k only) */
    for (int k = 0; k < m->nz; ++k) {
        for (int j = 0; j < m->ny; ++j) {
            for (int g = 1; g <= m->ngx; ++g) {
                size_t ghost = mesh_index(m, -g, j, k);
                if (m->bc[XLOW] == BC_PERIODIC) {
                    field[ghost] = field[mesh_index(m, m->nx - g, j, k)];
                } else {
                    field[ghost] = field[mesh_index(m, 0, j, k)];
                }
                ghost = mesh_index(m, m->nx - 1 + g, j, k);
                if (m->bc[XHIGH] == BC_PERIODIC) {
                    field[ghost] = field[mesh_index(m, g - 1, j, k)];
                } else {
                    field[ghost] = field[mesh_index(m, m->nx - 1, j, k)];
                }
            }
        }
    }

    /* Y-direction */
    if (m->dim >= 2) {
        int iLo = -m->ngx;
        int iHi = m->nx + m->ngx;
        for (int k = 0; k < m->nz; ++k) {
            for (int i = iLo; i < iHi; ++i) {
                for (int g = 1; g <= m->ngy; ++g) {
                    size_t ghost = mesh_index(m, i, -g, k);
                    if (m->bc[YLOW] == BC_PERIODIC) {
                        field[ghost] = field[mesh_index(m, i, m->ny - g, k)];
                    } else {
                        field[ghost] = field[mesh_index(m, i, 0, k)];
                    }
                    ghost = mesh_index(m, i, m->ny - 1 + g, k);
                    if (m->bc[YHIGH] == BC_PERIODIC) {
                        field[ghost] = field[mesh_index(m, i, g - 1, k)];
                    } else {
                        field[ghost] = field[mesh_index(m, i, m->ny - 1, k)];
                    }
                }
            }
        }
    }

    /* Z-direction */
    if (m->dim >= 3) {
        int iLo = -m->ngx;
        int iHi = m->nx + m->ngx;
        int jLo = -m->ngy;
        int jHi = m->ny + m->ngy;
        for (int j = jLo; j < jHi; ++j) {
            for (int i = iLo; i < iHi; ++i) {
                for (int g = 1; g <= m->ngz; ++g) {
                    size_t ghost = mesh_index(m, i, j, -g);
                    if (m->bc[ZLOW] == BC_PERIODIC) {
                        field[ghost] = field[mesh_index(m, i, j, m->nz - g)];
                    } else {
                        field[ghost] = field[mesh_index(m, i, j, 0)];
                    }
                    ghost = mesh_index(m, i, j, m->nz - 1 + g);
                    if (m->bc[ZHIGH] == BC_PERIODIC) {
                        field[ghost] = field[mesh_index(m, i, j, g - 1)];
                    } else {
                        field[ghost] = field[mesh_index(m, i, j, m->nz - 1)];
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   MPI-aware boundary conditions
   --------------------------------------------------------------------------- */

void mesh_apply_bcs_mpi(const struct RectilinearMesh* m,
                        struct SolutionState* state,
                        enum VarSet varSet,
                        struct HaloExchange* halo)
{
    /* Onion-peel: for each direction, first exchange halos, then apply
       physical BCs only on faces that are true boundaries. */

    /* X-direction */
    halo_exchange_state_direction(halo, state, varSet, 0);
    fill_ghost_x(m, state, varSet,
                 !mpi_is_physical_boundary(halo->mpi, XLOW),
                 !mpi_is_physical_boundary(halo->mpi, XHIGH));

    /* Y-direction */
    if (m->dim >= 2) {
        halo_exchange_state_direction(halo, state, varSet, 1);
        fill_ghost_y(m, state, varSet,
                     !mpi_is_physical_boundary(halo->mpi, YLOW),
                     !mpi_is_physical_boundary(halo->mpi, YHIGH));
    }

    /* Z-direction */
    if (m->dim >= 3) {
        halo_exchange_state_direction(halo, state, varSet, 2);
        fill_ghost_z(m, state, varSet,
                     !mpi_is_physical_boundary(halo->mpi, ZLOW),
                     !mpi_is_physical_boundary(halo->mpi, ZHIGH));
    }
}

void mesh_fill_scalar_ghosts_mpi(const struct RectilinearMesh* m,
                                 double* field,
                                 struct HaloExchange* halo)
{
    /* X-direction halo exchange */
    halo_exchange_scalar_direction(halo, field, 0);

    if (mpi_is_physical_boundary(halo->mpi, XLOW) || mpi_is_physical_boundary(halo->mpi, XHIGH)) {
        for (int k = 0; k < m->nz; ++k) {
            for (int j = 0; j < m->ny; ++j) {
                if (mpi_is_physical_boundary(halo->mpi, XLOW)) {
                    for (int g = 1; g <= m->ngx; ++g) {
                        size_t ghost = mesh_index(m, -g, j, k);
                        if (m->bc[XLOW] == BC_PERIODIC) {
                            field[ghost] = field[mesh_index(m, m->nx - g, j, k)];
                        } else {
                            field[ghost] = field[mesh_index(m, 0, j, k)];
                        }
                    }
                }
                if (mpi_is_physical_boundary(halo->mpi, XHIGH)) {
                    for (int g = 1; g <= m->ngx; ++g) {
                        size_t ghost = mesh_index(m, m->nx - 1 + g, j, k);
                        if (m->bc[XHIGH] == BC_PERIODIC) {
                            field[ghost] = field[mesh_index(m, g - 1, j, k)];
                        } else {
                            field[ghost] = field[mesh_index(m, m->nx - 1, j, k)];
                        }
                    }
                }
            }
        }
    }

    /* Y-direction */
    if (m->dim >= 2) {
        halo_exchange_scalar_direction(halo, field, 1);

        if (mpi_is_physical_boundary(halo->mpi, YLOW) || mpi_is_physical_boundary(halo->mpi, YHIGH)) {
            int iLo = -m->ngx;
            int iHi = m->nx + m->ngx;
            for (int k = 0; k < m->nz; ++k) {
                for (int i = iLo; i < iHi; ++i) {
                    if (mpi_is_physical_boundary(halo->mpi, YLOW)) {
                        for (int g = 1; g <= m->ngy; ++g) {
                            size_t ghost = mesh_index(m, i, -g, k);
                            if (m->bc[YLOW] == BC_PERIODIC) {
                                field[ghost] = field[mesh_index(m, i, m->ny - g, k)];
                            } else {
                                field[ghost] = field[mesh_index(m, i, 0, k)];
                            }
                        }
                    }
                    if (mpi_is_physical_boundary(halo->mpi, YHIGH)) {
                        for (int g = 1; g <= m->ngy; ++g) {
                            size_t ghost = mesh_index(m, i, m->ny - 1 + g, k);
                            if (m->bc[YHIGH] == BC_PERIODIC) {
                                field[ghost] = field[mesh_index(m, i, g - 1, k)];
                            } else {
                                field[ghost] = field[mesh_index(m, i, m->ny - 1, k)];
                            }
                        }
                    }
                }
            }
        }
    }

    /* Z-direction */
    if (m->dim >= 3) {
        halo_exchange_scalar_direction(halo, field, 2);

        if (mpi_is_physical_boundary(halo->mpi, ZLOW) || mpi_is_physical_boundary(halo->mpi, ZHIGH)) {
            int iLo = -m->ngx;
            int iHi = m->nx + m->ngx;
            int jLo = -m->ngy;
            int jHi = m->ny + m->ngy;
            for (int j = jLo; j < jHi; ++j) {
                for (int i = iLo; i < iHi; ++i) {
                    if (mpi_is_physical_boundary(halo->mpi, ZLOW)) {
                        for (int g = 1; g <= m->ngz; ++g) {
                            size_t ghost = mesh_index(m, i, j, -g);
                            if (m->bc[ZLOW] == BC_PERIODIC) {
                                field[ghost] = field[mesh_index(m, i, j, m->nz - g)];
                            } else {
                                field[ghost] = field[mesh_index(m, i, j, 0)];
                            }
                        }
                    }
                    if (mpi_is_physical_boundary(halo->mpi, ZHIGH)) {
                        for (int g = 1; g <= m->ngz; ++g) {
                            size_t ghost = mesh_index(m, i, j, m->nz - 1 + g);
                            if (m->bc[ZHIGH] == BC_PERIODIC) {
                                field[ghost] = field[mesh_index(m, i, j, g - 1)];
                            } else {
                                field[ghost] = field[mesh_index(m, i, j, m->nz - 1)];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
   Device physical BC fill (VARSET_PRIM, single-phase).  Uses fields already
   resident on the GPU; the BC type and velocity-sign factors are uniform for
   all threads on a given face, so the inner switch does not diverge.
   --------------------------------------------------------------------------- */

static inline void sign_factors_for_bc(int bc, int normalAxis,
                                       double* sU, double* sV, double* sW) {
    *sU = 1.0; *sV = 1.0; *sW = 1.0;
    if (bc == BC_SLIP_WALL) {
        if (normalAxis == 0) *sU = -1.0;
        else if (normalAxis == 1) *sV = -1.0;
        else *sW = -1.0;
    } else if (bc == BC_NO_SLIP_WALL) {
        *sU = -1.0; *sV = -1.0; *sW = -1.0;
    }
}

/* Field layout for ghost copy:
 *   VARSET_PRIM: rho, velU, [velV], [velW], pres, sigma
 *   VARSET_CONS: rho, rhoU, [rhoV], [rhoW], rhoE, sigma
 * sigma is copied in both varSets (matches state_copy_cell_P/_C).  The
 * isCons flag is uniform per launch, so the branch does not diverge. */

static void fill_ghost_x_device(const RectilinearMesh* m, SolutionState* state,
                                enum VarSet varSet, int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    if (ngx == 0) return;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = m->dim;
    const int isCons = (varSet == VARSET_CONS);
    const int nPhases = state->nPhases;
    const size_t tc = state->totalCells;
    double* rho   = state->rho;
    double* f1    = isCons ? state->rhoU : state->velU;
    double* f2    = isCons ? state->rhoV : state->velV;
    double* f3    = isCons ? state->rhoW : state->velW;
    double* flast = isCons ? state->rhoE : state->pres;
    double* sig   = state->sigma;
    double* aRho  = state->alphaRho;
    double* a     = state->alpha;

    /* Two faces fused into one launch.  Each (face, k, j, g) thread handles
     * one ghost cell.  `skipLow` / `skipHigh` are uniform per launch, so the
     * `if (skip*) continue;` test does not introduce warp divergence. */
    const int bcLow  = m->bc[XLOW];
    const int bcHigh = m->bc[XHIGH];
    double sULo, sVLo, sWLo, sUHi, sVHi, sWHi;
    sign_factors_for_bc(bcLow,  0, &sULo, &sVLo, &sWLo);
    sign_factors_for_bc(bcHigh, 0, &sUHi, &sVHi, &sWHi);

    #pragma omp target teams distribute parallel for collapse(4)
    for (int face = 0; face < 2; ++face) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int g = 1; g <= ngx; ++g) {
                    if (face == 0 ? skipLow : skipHigh) continue;
                    int iGhostExt, iSrcExt;
                    double sU, sV, sW;
                    int bc;
                    if (face == 0) {
                        bc = bcLow;
                        sU = sULo; sV = sVLo; sW = sWLo;
                        iGhostExt = ngx - g;
                        switch (bc) {
                        case BC_PERIODIC:     iSrcExt = ngx + nx - g; break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: iSrcExt = ngx + g - 1;  break;
                        case BC_OUTFLOW:
                        default:              iSrcExt = ngx;          break;
                        }
                    } else {
                        bc = bcHigh;
                        sU = sUHi; sV = sVHi; sW = sWHi;
                        iGhostExt = ngx + nx - 1 + g;
                        switch (bc) {
                        case BC_PERIODIC:     iSrcExt = ngx + g - 1;  break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: iSrcExt = ngx + nx - g; break;
                        case BC_OUTFLOW:
                        default:              iSrcExt = ngx + nx - 1; break;
                        }
                    }
                    size_t row = (size_t)(j + ngy) + (size_t)nyTot * (size_t)(k + ngz);
                    size_t ghostIdx = (size_t)iGhostExt + (size_t)nxTot * row;
                    size_t srcIdx   = (size_t)iSrcExt   + (size_t)nxTot * row;
                    rho[ghostIdx]   = rho[srcIdx];
                    f1[ghostIdx]    = sU * f1[srcIdx];
                    if (dim >= 2) f2[ghostIdx] = sV * f2[srcIdx];
                    if (dim >= 3) f3[ghostIdx] = sW * f3[srcIdx];
                    flast[ghostIdx] = flast[srcIdx];
                    sig[ghostIdx]   = sig[srcIdx];
                    for (int ph = 0; ph < nPhases; ++ph) {
                        size_t off = (size_t)ph * tc;
                        aRho[off + ghostIdx] = aRho[off + srcIdx];
                        a[off + ghostIdx]    = a[off + srcIdx];
                    }
                }
            }
        }
    }
}

static void fill_ghost_y_device(const RectilinearMesh* m, SolutionState* state,
                                enum VarSet varSet, int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iSpan = nx + 2 * ngx;
    const int dim = m->dim;
    const int isCons = (varSet == VARSET_CONS);
    const int nPhases = state->nPhases;
    const size_t tc = state->totalCells;
    double* rho   = state->rho;
    double* f1    = isCons ? state->rhoU : state->velU;
    double* f2    = isCons ? state->rhoV : state->velV;
    double* f3    = isCons ? state->rhoW : state->velW;
    double* flast = isCons ? state->rhoE : state->pres;
    double* sig   = state->sigma;
    double* aRho  = state->alphaRho;
    double* a     = state->alpha;

    if (ngy == 0) return;
    const int bcLow  = m->bc[YLOW];
    const int bcHigh = m->bc[YHIGH];
    double sULo, sVLo, sWLo, sUHi, sVHi, sWHi;
    sign_factors_for_bc(bcLow,  1, &sULo, &sVLo, &sWLo);
    sign_factors_for_bc(bcHigh, 1, &sUHi, &sVHi, &sWHi);

    #pragma omp target teams distribute parallel for collapse(4)
    for (int face = 0; face < 2; ++face) {
        for (int k = 0; k < nz; ++k) {
            for (int g = 1; g <= ngy; ++g) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    if (face == 0 ? skipLow : skipHigh) continue;
                    int jGhostExt, jSrcExt;
                    double sU, sV, sW;
                    int bc;
                    if (face == 0) {
                        bc = bcLow;
                        sU = sULo; sV = sVLo; sW = sWLo;
                        jGhostExt = ngy - g;
                        switch (bc) {
                        case BC_PERIODIC:     jSrcExt = ngy + ny - g; break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: jSrcExt = ngy + g - 1;  break;
                        case BC_OUTFLOW:
                        default:              jSrcExt = ngy;          break;
                        }
                    } else {
                        bc = bcHigh;
                        sU = sUHi; sV = sVHi; sW = sWHi;
                        jGhostExt = ngy + ny - 1 + g;
                        switch (bc) {
                        case BC_PERIODIC:     jSrcExt = ngy + g - 1;  break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: jSrcExt = ngy + ny - g; break;
                        case BC_OUTFLOW:
                        default:              jSrcExt = ngy + ny - 1; break;
                        }
                    }
                    int iExt = ii;
                    size_t ghostIdx = (size_t)iExt +
                                      (size_t)nxTot * ((size_t)jGhostExt +
                                                       (size_t)nyTot * (size_t)(k + ngz));
                    size_t srcIdx   = (size_t)iExt +
                                      (size_t)nxTot * ((size_t)jSrcExt +
                                                       (size_t)nyTot * (size_t)(k + ngz));
                    rho[ghostIdx]   = rho[srcIdx];
                    f1[ghostIdx]    = sU * f1[srcIdx];
                    if (dim >= 2) f2[ghostIdx] = sV * f2[srcIdx];
                    if (dim >= 3) f3[ghostIdx] = sW * f3[srcIdx];
                    flast[ghostIdx] = flast[srcIdx];
                    sig[ghostIdx]   = sig[srcIdx];
                    for (int ph = 0; ph < nPhases; ++ph) {
                        size_t off = (size_t)ph * tc;
                        aRho[off + ghostIdx] = aRho[off + srcIdx];
                        a[off + ghostIdx]    = a[off + srcIdx];
                    }
                }
            }
        }
    }
}

static void fill_ghost_z_device(const RectilinearMesh* m, SolutionState* state,
                                enum VarSet varSet, int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iSpan = nx + 2 * ngx;
    const int jSpan = ny + 2 * ngy;
    const int dim = m->dim;
    const int isCons = (varSet == VARSET_CONS);
    const int nPhases = state->nPhases;
    const size_t tc = state->totalCells;
    double* rho   = state->rho;
    double* f1    = isCons ? state->rhoU : state->velU;
    double* f2    = isCons ? state->rhoV : state->velV;
    double* f3    = isCons ? state->rhoW : state->velW;
    double* flast = isCons ? state->rhoE : state->pres;
    double* sig   = state->sigma;
    double* aRho  = state->alphaRho;
    double* a     = state->alpha;

    if (ngz == 0) return;
    const int bcLow  = m->bc[ZLOW];
    const int bcHigh = m->bc[ZHIGH];
    double sULo, sVLo, sWLo, sUHi, sVHi, sWHi;
    sign_factors_for_bc(bcLow,  2, &sULo, &sVLo, &sWLo);
    sign_factors_for_bc(bcHigh, 2, &sUHi, &sVHi, &sWHi);

    #pragma omp target teams distribute parallel for collapse(4)
    for (int face = 0; face < 2; ++face) {
        for (int g = 1; g <= ngz; ++g) {
            for (int jj = 0; jj < jSpan; ++jj) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    if (face == 0 ? skipLow : skipHigh) continue;
                    int kGhostExt, kSrcExt;
                    double sU, sV, sW;
                    int bc;
                    if (face == 0) {
                        bc = bcLow;
                        sU = sULo; sV = sVLo; sW = sWLo;
                        kGhostExt = ngz - g;
                        switch (bc) {
                        case BC_PERIODIC:     kSrcExt = ngz + nz - g; break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: kSrcExt = ngz + g - 1;  break;
                        case BC_OUTFLOW:
                        default:              kSrcExt = ngz;          break;
                        }
                    } else {
                        bc = bcHigh;
                        sU = sUHi; sV = sVHi; sW = sWHi;
                        kGhostExt = ngz + nz - 1 + g;
                        switch (bc) {
                        case BC_PERIODIC:     kSrcExt = ngz + g - 1;  break;
                        case BC_SYMMETRY:
                        case BC_SLIP_WALL:
                        case BC_NO_SLIP_WALL: kSrcExt = ngz + nz - g; break;
                        case BC_OUTFLOW:
                        default:              kSrcExt = ngz + nz - 1; break;
                        }
                    }
                    int iExt = ii;
                    int jExt = jj;
                    size_t ghostIdx = (size_t)iExt +
                                      (size_t)nxTot * ((size_t)jExt +
                                                       (size_t)nyTot * (size_t)kGhostExt);
                    size_t srcIdx   = (size_t)iExt +
                                      (size_t)nxTot * ((size_t)jExt +
                                                       (size_t)nyTot * (size_t)kSrcExt);
                    rho[ghostIdx]   = rho[srcIdx];
                    f1[ghostIdx]    = sU * f1[srcIdx];
                    if (dim >= 2) f2[ghostIdx] = sV * f2[srcIdx];
                    if (dim >= 3) f3[ghostIdx] = sW * f3[srcIdx];
                    flast[ghostIdx] = flast[srcIdx];
                    sig[ghostIdx]   = sig[srcIdx];
                    for (int ph = 0; ph < nPhases; ++ph) {
                        size_t off = (size_t)ph * tc;
                        aRho[off + ghostIdx] = aRho[off + srcIdx];
                        a[off + ghostIdx]    = a[off + srcIdx];
                    }
                }
            }
        }
    }
}

void mesh_apply_bcs_mpi_device(const struct RectilinearMesh* m,
                               struct SolutionState* state,
                               enum VarSet varSet,
                               struct HaloExchange* halo)
{
    /* VARSET_PRIM and VARSET_CONS supported on device for any nPhases. */
    if (varSet != VARSET_PRIM && varSet != VARSET_CONS) {
        solution_state_update_prim_from_device(state);
        mesh_apply_bcs_mpi(m, state, varSet, halo);
        solution_state_update_prim_to_device(state);
        return;
    }

    /* `skip*` semantics: skip the device ghost fill iff the MPI exchange
     * already wrote to this ghost slab.  That only happens when the
     * neighbour is a different rank.  Self-rank wraps (single-rank periodic)
     * and physical boundaries both leave the ghost cells for the device fill
     * to handle (it switches on m->bc[*] for periodic/symmetry/wall/outflow). */

    /* X-direction. */
    /* Short-circuit halo dispatch when neither face is a remote neighbour:
     * skip the function call, the `num_fields()` work, the buffer-capacity
     * check, and the per-direction switch entirely.  The device ghost-fill
     * kernel below handles self-rank periodic wrap and physical boundaries. */
    const int xRemote = mpi_neighbor_is_remote(halo->mpi, XLOW) ||
                        mpi_neighbor_is_remote(halo->mpi, XHIGH);
    const int yRemote = (m->dim >= 2) &&
                        (mpi_neighbor_is_remote(halo->mpi, YLOW) ||
                         mpi_neighbor_is_remote(halo->mpi, YHIGH));
    const int zRemote = (m->dim >= 3) &&
                        (mpi_neighbor_is_remote(halo->mpi, ZLOW) ||
                         mpi_neighbor_is_remote(halo->mpi, ZHIGH));

    NVTX_PUSH("BC::X");
    if (xRemote) halo_exchange_state_direction_device(halo, state, varSet, 0);
    fill_ghost_x_device(m, state, varSet,
                        mpi_neighbor_is_remote(halo->mpi, XLOW),
                        mpi_neighbor_is_remote(halo->mpi, XHIGH));
    NVTX_POP();

    if (m->dim >= 2) {
        NVTX_PUSH("BC::Y");
        if (yRemote) halo_exchange_state_direction_device(halo, state, varSet, 1);
        fill_ghost_y_device(m, state, varSet,
                            mpi_neighbor_is_remote(halo->mpi, YLOW),
                            mpi_neighbor_is_remote(halo->mpi, YHIGH));
        NVTX_POP();
    }

    if (m->dim >= 3) {
        NVTX_PUSH("BC::Z");
        if (zRemote) halo_exchange_state_direction_device(halo, state, varSet, 2);
        fill_ghost_z_device(m, state, varSet,
                            mpi_neighbor_is_remote(halo->mpi, ZLOW),
                            mpi_neighbor_is_remote(halo->mpi, ZHIGH));
        NVTX_POP();
    }
}

/* ---------------------------------------------------------------------------
   Device scalar ghost fill (used by IGR sigma)
   --------------------------------------------------------------------------- */

static void fill_scalar_ghost_x_device(const RectilinearMesh* m, double* field,
                                       int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;

    if (!skipLow && ngx > 0) {
        const int bc = m->bc[XLOW];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int g = 1; g <= ngx; ++g) {
                    int iGhostExt = ngx - g;
                    int iSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     iSrcExt = ngx + nx - g; break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: iSrcExt = ngx + g - 1;  break;
                    case BC_OUTFLOW:
                    default:              iSrcExt = ngx;          break;
                    }
                    size_t row = (size_t)(j + ngy) + (size_t)nyTot * (size_t)(k + ngz);
                    field[(size_t)iGhostExt + (size_t)nxTot * row] =
                        field[(size_t)iSrcExt + (size_t)nxTot * row];
                }
            }
        }
    }

    if (!skipHigh && ngx > 0) {
        const int bc = m->bc[XHIGH];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int g = 1; g <= ngx; ++g) {
                    int iGhostExt = ngx + nx - 1 + g;
                    int iSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     iSrcExt = ngx + g - 1;     break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: iSrcExt = ngx + nx - g;    break;
                    case BC_OUTFLOW:
                    default:              iSrcExt = ngx + nx - 1;    break;
                    }
                    size_t row = (size_t)(j + ngy) + (size_t)nyTot * (size_t)(k + ngz);
                    field[(size_t)iGhostExt + (size_t)nxTot * row] =
                        field[(size_t)iSrcExt + (size_t)nxTot * row];
                }
            }
        }
    }
}

static void fill_scalar_ghost_y_device(const RectilinearMesh* m, double* field,
                                       int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iSpan = nx + 2 * ngx;

    if (!skipLow && ngy > 0) {
        const int bc = m->bc[YLOW];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int g = 1; g <= ngy; ++g) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    int jGhostExt = ngy - g;
                    int jSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     jSrcExt = ngy + ny - g; break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: jSrcExt = ngy + g - 1;  break;
                    case BC_OUTFLOW:
                    default:              jSrcExt = ngy;          break;
                    }
                    size_t ghostIdx = (size_t)ii +
                                      (size_t)nxTot * ((size_t)jGhostExt +
                                                       (size_t)nyTot * (size_t)(k + ngz));
                    size_t srcIdx = (size_t)ii +
                                    (size_t)nxTot * ((size_t)jSrcExt +
                                                     (size_t)nyTot * (size_t)(k + ngz));
                    field[ghostIdx] = field[srcIdx];
                }
            }
        }
    }

    if (!skipHigh && ngy > 0) {
        const int bc = m->bc[YHIGH];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int g = 1; g <= ngy; ++g) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    int jGhostExt = ngy + ny - 1 + g;
                    int jSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     jSrcExt = ngy + g - 1;     break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: jSrcExt = ngy + ny - g;    break;
                    case BC_OUTFLOW:
                    default:              jSrcExt = ngy + ny - 1;    break;
                    }
                    size_t ghostIdx = (size_t)ii +
                                      (size_t)nxTot * ((size_t)jGhostExt +
                                                       (size_t)nyTot * (size_t)(k + ngz));
                    size_t srcIdx = (size_t)ii +
                                    (size_t)nxTot * ((size_t)jSrcExt +
                                                     (size_t)nyTot * (size_t)(k + ngz));
                    field[ghostIdx] = field[srcIdx];
                }
            }
        }
    }
}

static void fill_scalar_ghost_z_device(const RectilinearMesh* m, double* field,
                                       int skipLow, int skipHigh) {
    const int nx = m->nx, ny = m->ny, nz = m->nz;
    const int ngx = m->ngx, ngy = m->ngy, ngz = m->ngz;
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iSpan = nx + 2 * ngx;
    const int jSpan = ny + 2 * ngy;

    if (!skipLow && ngz > 0) {
        const int bc = m->bc[ZLOW];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int g = 1; g <= ngz; ++g) {
            for (int jj = 0; jj < jSpan; ++jj) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    int kGhostExt = ngz - g;
                    int kSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     kSrcExt = ngz + nz - g; break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: kSrcExt = ngz + g - 1;  break;
                    case BC_OUTFLOW:
                    default:              kSrcExt = ngz;          break;
                    }
                    size_t ghostIdx = (size_t)ii +
                                      (size_t)nxTot * ((size_t)jj +
                                                       (size_t)nyTot * (size_t)kGhostExt);
                    size_t srcIdx = (size_t)ii +
                                    (size_t)nxTot * ((size_t)jj +
                                                     (size_t)nyTot * (size_t)kSrcExt);
                    field[ghostIdx] = field[srcIdx];
                }
            }
        }
    }

    if (!skipHigh && ngz > 0) {
        const int bc = m->bc[ZHIGH];
        #pragma omp target teams distribute parallel for collapse(3)
        for (int g = 1; g <= ngz; ++g) {
            for (int jj = 0; jj < jSpan; ++jj) {
                for (int ii = 0; ii < iSpan; ++ii) {
                    int kGhostExt = ngz + nz - 1 + g;
                    int kSrcExt;
                    switch (bc) {
                    case BC_PERIODIC:     kSrcExt = ngz + g - 1;     break;
                    case BC_SYMMETRY:
                    case BC_SLIP_WALL:
                    case BC_NO_SLIP_WALL: kSrcExt = ngz + nz - g;    break;
                    case BC_OUTFLOW:
                    default:              kSrcExt = ngz + nz - 1;    break;
                    }
                    size_t ghostIdx = (size_t)ii +
                                      (size_t)nxTot * ((size_t)jj +
                                                       (size_t)nyTot * (size_t)kGhostExt);
                    size_t srcIdx = (size_t)ii +
                                    (size_t)nxTot * ((size_t)jj +
                                                     (size_t)nyTot * (size_t)kSrcExt);
                    field[ghostIdx] = field[srcIdx];
                }
            }
        }
    }
}

void mesh_fill_scalar_ghosts_mpi_device(const struct RectilinearMesh* m,
                                        double* field,
                                        struct HaloExchange* halo)
{
    halo_exchange_scalar_direction_device(halo, field, 0);
    fill_scalar_ghost_x_device(m, field,
                               mpi_neighbor_is_remote(halo->mpi, XLOW),
                               mpi_neighbor_is_remote(halo->mpi, XHIGH));

    if (m->dim >= 2) {
        halo_exchange_scalar_direction_device(halo, field, 1);
        fill_scalar_ghost_y_device(m, field,
                                   mpi_neighbor_is_remote(halo->mpi, YLOW),
                                   mpi_neighbor_is_remote(halo->mpi, YHIGH));
    }

    if (m->dim >= 3) {
        halo_exchange_scalar_direction_device(halo, field, 2);
        fill_scalar_ghost_z_device(m, field,
                                   mpi_neighbor_is_remote(halo->mpi, ZLOW),
                                   mpi_neighbor_is_remote(halo->mpi, ZHIGH));
    }
}
