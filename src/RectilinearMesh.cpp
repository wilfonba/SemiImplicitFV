#include "RectilinearMesh.hpp"
#include "HaloExchange.hpp"
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
