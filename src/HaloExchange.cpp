#include "HaloExchange.hpp"
#include "RectilinearMesh.hpp"
#include "SolutionState.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>

void halo_init(HaloExchange* h,
               const MPIContext* mpi,
               const RectilinearMesh* mesh) {
    h->mpi = mpi;
    h->mesh = mesh;
    h->nGhost = mesh->nGhost;
    h->nx = mesh->nx;
    h->ny = mesh->ny;
    h->nz = mesh->nz;
    h->ngx = mesh->ngx;
    h->ngy = mesh->ngy;
    h->ngz = mesh->ngz;
    h->dim = mesh->dim;
    for (int f = 0; f < 6; ++f) {
        h->sendBuf[f] = NULL;
        h->recvBuf[f] = NULL;
        h->sendBufCapacity[f] = 0;
        h->recvBufCapacity[f] = 0;
    }
}

void halo_free(HaloExchange* h) {
    for (int f = 0; f < 6; ++f) {
        std::free(h->sendBuf[f]);
        h->sendBuf[f] = NULL;
        std::free(h->recvBuf[f]);
        h->recvBuf[f] = NULL;
        h->sendBufCapacity[f] = 0;
        h->recvBufCapacity[f] = 0;
    }
}

/* Ensure buffer is at least 'size' doubles. */
static void ensure_buf(double** buf, size_t* capacity, size_t size) {
    if (*capacity < size) {
        std::free(*buf);
        *buf = (double*)std::malloc(size * sizeof(double));
        *capacity = size;
    }
}

static size_t slab_size(const HaloExchange* h, int face, int nFields) {
    size_t cells = 0;
    switch (face) {
        case XLOW:
        case XHIGH:
            cells = (size_t)h->nGhost * h->ny * h->nz;
            break;
        case YLOW:
        case YHIGH:
            cells = (size_t)(h->nx + 2 * h->ngx) * h->nGhost * h->nz;
            break;
        case ZLOW:
        case ZHIGH:
            cells = (size_t)(h->nx + 2 * h->ngx) * (h->ny + 2 * h->ngy) * h->nGhost;
            break;
    }
    return cells * (size_t)nFields;
}

/* Number of scalar fields for a given VarSet and dimension */
static int num_fields(enum VarSet varSet, int dim, const SolutionState* state) {
    int nMultiPhase = 0;
    if (state->nPhases > 0) {
        nMultiPhase = 2 * state->nPhases;
    }

    switch (varSet) {
        case VARSET_PRIM:
            return 3 + (dim >= 2 ? 1 : 0) + (dim >= 3 ? 1 : 0) + 1 + nMultiPhase;
        case VARSET_CONS:
            return 2 + (dim >= 2 ? 1 : 0) + (dim >= 3 ? 1 : 0) + 1 + nMultiPhase;
        default:
            return 5 + (dim >= 2 ? 2 : 0) + (dim >= 3 ? 2 : 0) + 2 + 1 + nMultiPhase;
    }
}

/* ---------------------------------------------------------------------------
   Scalar pack/unpack
   --------------------------------------------------------------------------- */

static void pack_scalar(const HaloExchange* h, const double* field, int face, double* buf) {
    const RectilinearMesh* m = h->mesh;
    size_t pos = 0;

    switch (face) {
        case XLOW:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        buf[pos++] = field[mesh_index(m, g, j, k)];
            break;
        case XHIGH:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        buf[pos++] = field[mesh_index(m, h->nx - h->nGhost + g, j, k)];
            break;
        case YLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_index(m, i, g, k)];
            break;
        }
        case YHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_index(m, i, h->ny - h->nGhost + g, k)];
            break;
        }
        case ZLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_index(m, i, j, g)];
            break;
        }
        case ZHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_index(m, i, j, h->nz - h->nGhost + g)];
            break;
        }
    }
}

static void unpack_scalar(const HaloExchange* h, double* field, int face, const double* buf) {
    const RectilinearMesh* m = h->mesh;
    size_t pos = 0;

    switch (face) {
        case XLOW:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        field[mesh_index(m, -h->nGhost + g, j, k)] = buf[pos++];
            break;
        case XHIGH:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        field[mesh_index(m, h->nx + g, j, k)] = buf[pos++];
            break;
        case YLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_index(m, i, -h->nGhost + g, k)] = buf[pos++];
            break;
        }
        case YHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_index(m, i, h->ny + g, k)] = buf[pos++];
            break;
        }
        case ZLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_index(m, i, j, -h->nGhost + g)] = buf[pos++];
            break;
        }
        case ZHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_index(m, i, j, h->nz + g)] = buf[pos++];
            break;
        }
    }
}

/* ---------------------------------------------------------------------------
   State pack/unpack (multi-field)
   --------------------------------------------------------------------------- */

static void pack_state_cell(const SolutionState* state, enum VarSet varSet, size_t idx,
                            double* buf, size_t* pos) {
    int dim = state->dim;
    int nPhases = state->nPhases;
    size_t tc = state->totalCells;

    switch (varSet) {
        case VARSET_PRIM:
            buf[(*pos)++] = state->rho[idx];
            buf[(*pos)++] = state->velU[idx];
            if (dim >= 2) buf[(*pos)++] = state->velV[idx];
            if (dim >= 3) buf[(*pos)++] = state->velW[idx];
            buf[(*pos)++] = state->pres[idx];
            buf[(*pos)++] = state->sigma[idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alphaRho[ph * tc + idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alpha[ph * tc + idx];
            break;
        case VARSET_CONS:
            buf[(*pos)++] = state->rho[idx];
            buf[(*pos)++] = state->rhoU[idx];
            if (dim >= 2) buf[(*pos)++] = state->rhoV[idx];
            if (dim >= 3) buf[(*pos)++] = state->rhoW[idx];
            buf[(*pos)++] = state->rhoE[idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alphaRho[ph * tc + idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alpha[ph * tc + idx];
            break;
        default:
            buf[(*pos)++] = state->rho[idx];
            buf[(*pos)++] = state->rhoU[idx];
            if (dim >= 2) buf[(*pos)++] = state->rhoV[idx];
            if (dim >= 3) buf[(*pos)++] = state->rhoW[idx];
            buf[(*pos)++] = state->rhoE[idx];
            buf[(*pos)++] = state->velU[idx];
            if (dim >= 2) buf[(*pos)++] = state->velV[idx];
            if (dim >= 3) buf[(*pos)++] = state->velW[idx];
            buf[(*pos)++] = state->pres[idx];
            buf[(*pos)++] = state->sigma[idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alphaRho[ph * tc + idx];
            for (int ph = 0; ph < nPhases; ++ph) buf[(*pos)++] = state->alpha[ph * tc + idx];
            break;
    }
}

static void unpack_state_cell(SolutionState* state, enum VarSet varSet, size_t idx,
                              const double* buf, size_t* pos) {
    int dim = state->dim;
    int nPhases = state->nPhases;
    size_t tc = state->totalCells;

    switch (varSet) {
        case VARSET_PRIM:
            state->rho[idx]   = buf[(*pos)++];
            state->velU[idx]  = buf[(*pos)++];
            if (dim >= 2) state->velV[idx] = buf[(*pos)++];
            if (dim >= 3) state->velW[idx] = buf[(*pos)++];
            state->pres[idx]  = buf[(*pos)++];
            state->sigma[idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alphaRho[ph * tc + idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alpha[ph * tc + idx] = buf[(*pos)++];
            break;
        case VARSET_CONS:
            state->rho[idx]  = buf[(*pos)++];
            state->rhoU[idx] = buf[(*pos)++];
            if (dim >= 2) state->rhoV[idx] = buf[(*pos)++];
            if (dim >= 3) state->rhoW[idx] = buf[(*pos)++];
            state->rhoE[idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alphaRho[ph * tc + idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alpha[ph * tc + idx] = buf[(*pos)++];
            break;
        default:
            state->rho[idx]   = buf[(*pos)++];
            state->rhoU[idx]  = buf[(*pos)++];
            if (dim >= 2) state->rhoV[idx] = buf[(*pos)++];
            if (dim >= 3) state->rhoW[idx] = buf[(*pos)++];
            state->rhoE[idx]  = buf[(*pos)++];
            state->velU[idx]  = buf[(*pos)++];
            if (dim >= 2) state->velV[idx] = buf[(*pos)++];
            if (dim >= 3) state->velW[idx] = buf[(*pos)++];
            state->pres[idx]  = buf[(*pos)++];
            state->sigma[idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alphaRho[ph * tc + idx] = buf[(*pos)++];
            for (int ph = 0; ph < nPhases; ++ph) state->alpha[ph * tc + idx] = buf[(*pos)++];
            break;
    }
}

/* Pack state into sendBuf for a given face */
static void pack_state(const HaloExchange* h, SolutionState* state, enum VarSet varSet,
                       int face, double* buf) {
    const RectilinearMesh* m = h->mesh;
    size_t pos = 0;

    switch (face) {
        case XLOW:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        pack_state_cell(state, varSet, mesh_index(m, g, j, k), buf, &pos);
            break;
        case XHIGH:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        pack_state_cell(state, varSet, mesh_index(m, h->nx - h->nGhost + g, j, k), buf, &pos);
            break;
        case YLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        pack_state_cell(state, varSet, mesh_index(m, i, g, k), buf, &pos);
            break;
        }
        case YHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        pack_state_cell(state, varSet, mesh_index(m, i, h->ny - h->nGhost + g, k), buf, &pos);
            break;
        }
        case ZLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        pack_state_cell(state, varSet, mesh_index(m, i, j, g), buf, &pos);
            break;
        }
        case ZHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        pack_state_cell(state, varSet, mesh_index(m, i, j, h->nz - h->nGhost + g), buf, &pos);
            break;
        }
    }
}

static void unpack_state(const HaloExchange* h, SolutionState* state, enum VarSet varSet,
                         int face, const double* buf) {
    const RectilinearMesh* m = h->mesh;
    size_t pos = 0;

    switch (face) {
        case XLOW:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        unpack_state_cell(state, varSet, mesh_index(m, -h->nGhost + g, j, k), buf, &pos);
            break;
        case XHIGH:
            for (int k = 0; k < h->nz; ++k)
                for (int j = 0; j < h->ny; ++j)
                    for (int g = 0; g < h->nGhost; ++g)
                        unpack_state_cell(state, varSet, mesh_index(m, h->nx + g, j, k), buf, &pos);
            break;
        case YLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        unpack_state_cell(state, varSet, mesh_index(m, i, -h->nGhost + g, k), buf, &pos);
            break;
        }
        case YHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            for (int k = 0; k < h->nz; ++k)
                for (int g = 0; g < h->nGhost; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        unpack_state_cell(state, varSet, mesh_index(m, i, h->ny + g, k), buf, &pos);
            break;
        }
        case ZLOW: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        unpack_state_cell(state, varSet, mesh_index(m, i, j, -h->nGhost + g), buf, &pos);
            break;
        }
        case ZHIGH: {
            int iLo = -h->ngx, iHi = h->nx + h->ngx;
            int jLo = -h->ngy, jHi = h->ny + h->ngy;
            for (int g = 0; g < h->nGhost; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        unpack_state_cell(state, varSet, mesh_index(m, i, j, h->nz + g), buf, &pos);
            break;
        }
    }
}

/* ---------------------------------------------------------------------------
   MPI exchange
   --------------------------------------------------------------------------- */

static void do_exchange(HaloExchange* h, int lowFace, int highFace, size_t bufSize) {
    MPI_Request reqs[4];
    int nReqs = 0;

    int lowNeighbor  = h->mpi->neighbors[lowFace];
    int highNeighbor = h->mpi->neighbors[highFace];
    MPI_Comm comm = h->mpi->cartComm;

    MPI_Isend(h->sendBuf[lowFace], (int)bufSize, MPI_DOUBLE,
              lowNeighbor, 0, comm, &reqs[nReqs++]);
    MPI_Irecv(h->recvBuf[highFace], (int)bufSize, MPI_DOUBLE,
              highNeighbor, 0, comm, &reqs[nReqs++]);

    MPI_Isend(h->sendBuf[highFace], (int)bufSize, MPI_DOUBLE,
              highNeighbor, 1, comm, &reqs[nReqs++]);
    MPI_Irecv(h->recvBuf[lowFace], (int)bufSize, MPI_DOUBLE,
              lowNeighbor, 1, comm, &reqs[nReqs++]);

    MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);
}

/* ---------------------------------------------------------------------------
   Public API
   --------------------------------------------------------------------------- */

void halo_exchange_scalar_direction(HaloExchange* h, double* field, int direction) {
    int lowFace  = direction * 2;
    int highFace = direction * 2 + 1;

    size_t bufSize = slab_size(h, lowFace, 1);
    ensure_buf(&h->sendBuf[lowFace],  &h->sendBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->sendBuf[highFace], &h->sendBufCapacity[highFace], bufSize);
    ensure_buf(&h->recvBuf[lowFace],  &h->recvBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->recvBuf[highFace], &h->recvBufCapacity[highFace], bufSize);

    pack_scalar(h, field, lowFace, h->sendBuf[lowFace]);
    pack_scalar(h, field, highFace, h->sendBuf[highFace]);

    do_exchange(h, lowFace, highFace, bufSize);

    if (!mpi_is_physical_boundary(h->mpi, lowFace))
        unpack_scalar(h, field, lowFace, h->recvBuf[lowFace]);
    if (!mpi_is_physical_boundary(h->mpi, highFace))
        unpack_scalar(h, field, highFace, h->recvBuf[highFace]);
}

void halo_exchange_scalar(HaloExchange* h, double* field) {
    halo_exchange_scalar_direction(h, field, 0);
    if (h->dim >= 2) halo_exchange_scalar_direction(h, field, 1);
    if (h->dim >= 3) halo_exchange_scalar_direction(h, field, 2);
}

void halo_exchange_state_direction(HaloExchange* h, SolutionState* state,
                                   enum VarSet varSet, int direction) {
    int dim = state->dim;
    int nf = num_fields(varSet, dim, state);

    int lowFace  = direction * 2;
    int highFace = direction * 2 + 1;

    size_t bufSize = slab_size(h, lowFace, nf);
    ensure_buf(&h->sendBuf[lowFace],  &h->sendBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->sendBuf[highFace], &h->sendBufCapacity[highFace], bufSize);
    ensure_buf(&h->recvBuf[lowFace],  &h->recvBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->recvBuf[highFace], &h->recvBufCapacity[highFace], bufSize);

    pack_state(h, state, varSet, lowFace, h->sendBuf[lowFace]);
    pack_state(h, state, varSet, highFace, h->sendBuf[highFace]);

    do_exchange(h, lowFace, highFace, bufSize);

    if (!mpi_is_physical_boundary(h->mpi, lowFace))
        unpack_state(h, state, varSet, lowFace, h->recvBuf[lowFace]);
    if (!mpi_is_physical_boundary(h->mpi, highFace))
        unpack_state(h, state, varSet, highFace, h->recvBuf[highFace]);
}

void halo_exchange_state(HaloExchange* h, SolutionState* state, enum VarSet varSet) {
    halo_exchange_state_direction(h, state, varSet, 0);
    if (h->dim >= 2) halo_exchange_state_direction(h, state, varSet, 1);
    if (h->dim >= 3) halo_exchange_state_direction(h, state, varSet, 2);
}
