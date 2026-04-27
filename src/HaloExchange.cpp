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
        if (h->sendBuf[f]) {
            double* sb = h->sendBuf[f];
            size_t sc = h->sendBufCapacity[f];
            #pragma omp target exit data map(delete: sb[0:sc])
        }
        if (h->recvBuf[f]) {
            double* rb = h->recvBuf[f];
            size_t rc = h->recvBufCapacity[f];
            #pragma omp target exit data map(delete: rb[0:rc])
        }
        std::free(h->sendBuf[f]);
        h->sendBuf[f] = NULL;
        std::free(h->recvBuf[f]);
        h->recvBuf[f] = NULL;
        h->sendBufCapacity[f] = 0;
        h->recvBufCapacity[f] = 0;
    }
}

/* Ensure buffer is at least 'size' doubles.  The allocation is mirrored on
 * the device via OpenMP map(alloc:), so device-side pack/unpack kernels can
 * write into and read from the same buffer without any additional bookkeeping
 * — target update from/to moves just the packed contents across PCIe. */
static void ensure_buf(double** buf, size_t* capacity, size_t size) {
    if (*capacity < size) {
        if (*buf) {
            double* old = *buf;
            size_t oldCap = *capacity;
            #pragma omp target exit data map(delete: old[0:oldCap])
            std::free(*buf);
        }
        *buf = (double*)std::malloc(size * sizeof(double));
        *capacity = size;
        double* nb = *buf;
        size_t nc = size;
        #pragma omp target enter data map(alloc: nb[0:nc])
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

/* ---------------------------------------------------------------------------
   Device pack/unpack (Phase 1 scope: VARSET_PRIM, single-phase)
   ---------------------------------------------------------------------------

   Layout decisions:
   - A "slab" at face F is the set of ngF cells one ghost-layer thick that
     either sources (for pack) or sinks (for unpack) at that face.
   - Cells are packed in cell-major order matching the host pack_state
     ordering, and fields within a cell follow VARSET_PRIM ordering: rho,
     velU, [velV], [velW], pres, sigma.
   - No std::switch inside the kernel on face type — we launch one of six
     specialised kernels, each with the slab indexing inlined.

   The kernels assume field pointers are already device-resident (attached in
   solution_state_init) and that the buffer pointer is device-resident too
   (attached in ensure_buf). */

/* PRIM layout: rho, velU, [velV], [velW], pres, sigma, then
 *              alphaRho[0..nPhases-1], alpha[0..nPhases-1]
 * CONS layout: rho, rhoU, [rhoV], [rhoW], rhoE, then
 *              alphaRho[0..nPhases-1], alpha[0..nPhases-1]
 * The isCons flag is uniform per launch, so the branch causes no divergence. */

static void pack_state_device_x(SolutionState* state, enum VarSet varSet, int face,
                                int nx, int ny, int nz,
                                int ngx, int ngy, int ngz,
                                int nGhost, int nf,
                                double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = state->dim;
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
    const int iOffset = (face == XLOW) ? ngx : (ngx + nx - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int g = 0; g < nGhost; ++g) {
                size_t cellIdx = (size_t)((iOffset + g) +
                                           nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t linear = (size_t)(g + nGhost * (j + (size_t)ny * k));
                size_t pos = linear * (size_t)nf;
                buf[pos++] = rho[cellIdx];
                buf[pos++] = f1[cellIdx];
                if (dim >= 2) buf[pos++] = f2[cellIdx];
                if (dim >= 3) buf[pos++] = f3[cellIdx];
                buf[pos++] = flast[cellIdx];
                if (!isCons) buf[pos++] = sig[cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = aRho[(size_t)ph * tc + cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = a[(size_t)ph * tc + cellIdx];
            }
        }
    }
}

static void unpack_state_device_x(SolutionState* state, enum VarSet varSet, int face,
                                  int nx, int ny, int nz,
                                  int ngx, int ngy, int ngz,
                                  int nGhost, int nf,
                                  const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int dim = state->dim;
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
    const int iOffset = (face == XLOW) ? 0 : (ngx + nx);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int g = 0; g < nGhost; ++g) {
                size_t cellIdx = (size_t)((iOffset + g) +
                                           nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t linear = (size_t)(g + nGhost * (j + (size_t)ny * k));
                size_t pos = linear * (size_t)nf;
                rho[cellIdx] = buf[pos++];
                f1[cellIdx]  = buf[pos++];
                if (dim >= 2) f2[cellIdx] = buf[pos++];
                if (dim >= 3) f3[cellIdx] = buf[pos++];
                flast[cellIdx] = buf[pos++];
                if (!isCons) sig[cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    aRho[(size_t)ph * tc + cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    a[(size_t)ph * tc + cellIdx] = buf[pos++];
            }
        }
    }
}

static void pack_state_device_y(SolutionState* state, enum VarSet varSet, int face,
                                int nx, int ny, int nz,
                                int ngx, int ngy, int ngz,
                                int nGhost, int nf,
                                double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int iSpan = iHi - iLo;
    const int dim = state->dim;
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
    const int jOffset = (face == YLOW) ? ngy : (ngy + ny - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int g = 0; g < nGhost; ++g) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                size_t cellIdx = (size_t)((i + ngx) +
                                           nxTot * ((jOffset + g) + nyTot * (k + ngz)));
                size_t linear = (size_t)(ii + iSpan * (g + nGhost * (size_t)k));
                size_t pos = linear * (size_t)nf;
                buf[pos++] = rho[cellIdx];
                buf[pos++] = f1[cellIdx];
                if (dim >= 2) buf[pos++] = f2[cellIdx];
                if (dim >= 3) buf[pos++] = f3[cellIdx];
                buf[pos++] = flast[cellIdx];
                if (!isCons) buf[pos++] = sig[cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = aRho[(size_t)ph * tc + cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = a[(size_t)ph * tc + cellIdx];
            }
        }
    }
}

static void unpack_state_device_y(SolutionState* state, enum VarSet varSet, int face,
                                  int nx, int ny, int nz,
                                  int ngx, int ngy, int ngz,
                                  int nGhost, int nf,
                                  const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int iSpan = iHi - iLo;
    const int dim = state->dim;
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
    const int jOffset = (face == YLOW) ? 0 : (ngy + ny);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int g = 0; g < nGhost; ++g) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                size_t cellIdx = (size_t)((i + ngx) +
                                           nxTot * ((jOffset + g) + nyTot * (k + ngz)));
                size_t linear = (size_t)(ii + iSpan * (g + nGhost * (size_t)k));
                size_t pos = linear * (size_t)nf;
                rho[cellIdx] = buf[pos++];
                f1[cellIdx]  = buf[pos++];
                if (dim >= 2) f2[cellIdx] = buf[pos++];
                if (dim >= 3) f3[cellIdx] = buf[pos++];
                flast[cellIdx] = buf[pos++];
                if (!isCons) sig[cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    aRho[(size_t)ph * tc + cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    a[(size_t)ph * tc + cellIdx] = buf[pos++];
            }
        }
    }
}

static void pack_state_device_z(SolutionState* state, enum VarSet varSet, int face,
                                int nx, int ny, int nz,
                                int ngx, int ngy, int ngz,
                                int nGhost, int nf,
                                double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int jLo = -ngy, jHi = ny + ngy;
    const int iSpan = iHi - iLo;
    const int jSpan = jHi - jLo;
    const int dim = state->dim;
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
    const int kOffset = (face == ZLOW) ? ngz : (ngz + nz - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int g = 0; g < nGhost; ++g) {
        for (int jj = 0; jj < jSpan; ++jj) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                int j = jLo + jj;
                size_t cellIdx = (size_t)((i + ngx) +
                                           nxTot * ((j + ngy) + nyTot * (kOffset + g)));
                size_t linear = (size_t)(ii + iSpan * (jj + (size_t)jSpan * g));
                size_t pos = linear * (size_t)nf;
                buf[pos++] = rho[cellIdx];
                buf[pos++] = f1[cellIdx];
                if (dim >= 2) buf[pos++] = f2[cellIdx];
                if (dim >= 3) buf[pos++] = f3[cellIdx];
                buf[pos++] = flast[cellIdx];
                if (!isCons) buf[pos++] = sig[cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = aRho[(size_t)ph * tc + cellIdx];
                for (int ph = 0; ph < nPhases; ++ph)
                    buf[pos++] = a[(size_t)ph * tc + cellIdx];
            }
        }
    }
}

static void unpack_state_device_z(SolutionState* state, enum VarSet varSet, int face,
                                  int nx, int ny, int nz,
                                  int ngx, int ngy, int ngz,
                                  int nGhost, int nf,
                                  const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int jLo = -ngy, jHi = ny + ngy;
    const int iSpan = iHi - iLo;
    const int jSpan = jHi - jLo;
    const int dim = state->dim;
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
    const int kOffset = (face == ZLOW) ? 0 : (ngz + nz);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int g = 0; g < nGhost; ++g) {
        for (int jj = 0; jj < jSpan; ++jj) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                int j = jLo + jj;
                size_t cellIdx = (size_t)((i + ngx) +
                                           nxTot * ((j + ngy) + nyTot * (kOffset + g)));
                size_t linear = (size_t)(ii + iSpan * (jj + (size_t)jSpan * g));
                size_t pos = linear * (size_t)nf;
                rho[cellIdx] = buf[pos++];
                f1[cellIdx]  = buf[pos++];
                if (dim >= 2) f2[cellIdx] = buf[pos++];
                if (dim >= 3) f3[cellIdx] = buf[pos++];
                flast[cellIdx] = buf[pos++];
                if (!isCons) sig[cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    aRho[(size_t)ph * tc + cellIdx] = buf[pos++];
                for (int ph = 0; ph < nPhases; ++ph)
                    a[(size_t)ph * tc + cellIdx] = buf[pos++];
            }
        }
    }
}

void halo_exchange_state_direction_device(HaloExchange* h,
                                          SolutionState* state,
                                          enum VarSet varSet,
                                          int direction) {
    /* VARSET_PRIM and VARSET_CONS with any nPhases run on GPU.  Anything
     * else falls back (currently only the "default" state-dump VarSet). */
    if (varSet != VARSET_PRIM && varSet != VARSET_CONS) {
        halo_exchange_state_direction(h, state, varSet, direction);
        return;
    }

    int dim = state->dim;
    int nf = num_fields(varSet, dim, state);
    int nGhost = h->nGhost;

    int lowFace  = direction * 2;
    int highFace = direction * 2 + 1;

    /* Skip pack / PCIe / MPI / unpack entirely when neither face has a remote
     * neighbour.  Self-rank wrap (single-rank periodic) and physical
     * boundaries are handled by the device ghost-fill kernels instead — this
     * avoids 4 PCIe transfers + 4 MPI ops + 4 implicit syncs per call. */
    if (!mpi_neighbor_is_remote(h->mpi, lowFace) &&
        !mpi_neighbor_is_remote(h->mpi, highFace)) {
        return;
    }

    size_t bufSize = slab_size(h, lowFace, nf);
    ensure_buf(&h->sendBuf[lowFace],  &h->sendBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->sendBuf[highFace], &h->sendBufCapacity[highFace], bufSize);
    ensure_buf(&h->recvBuf[lowFace],  &h->recvBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->recvBuf[highFace], &h->recvBufCapacity[highFace], bufSize);

    double* sbLow  = h->sendBuf[lowFace];
    double* sbHigh = h->sendBuf[highFace];
    double* rbLow  = h->recvBuf[lowFace];
    double* rbHigh = h->recvBuf[highFace];

    /* Device-side pack: writes directly into mapped sendBuf on GPU. */
    if (direction == 0) {
        pack_state_device_x(state, varSet, lowFace,  h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbLow);
        pack_state_device_x(state, varSet, highFace, h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbHigh);
    } else if (direction == 1) {
        pack_state_device_y(state, varSet, lowFace,  h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbLow);
        pack_state_device_y(state, varSet, highFace, h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbHigh);
    } else {
        pack_state_device_z(state, varSet, lowFace,  h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbLow);
        pack_state_device_z(state, varSet, highFace, h->nx, h->ny, h->nz,
                            h->ngx, h->ngy, h->ngz, nGhost, nf, sbHigh);
    }

    /* PCIe-only transfers: the packed buffers, not the full state. */
    #pragma omp target update from(sbLow[0:bufSize])
    #pragma omp target update from(sbHigh[0:bufSize])

    do_exchange(h, lowFace, highFace, bufSize);

    #pragma omp target update to(rbLow[0:bufSize])
    #pragma omp target update to(rbHigh[0:bufSize])

    /* Device-side unpack only on faces that actually received from a
     * neighbour; physical-boundary faces are left alone for the physical
     * BC kernel to fill. */
    if (!mpi_is_physical_boundary(h->mpi, lowFace)) {
        if (direction == 0)
            unpack_state_device_x(state, varSet, lowFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbLow);
        else if (direction == 1)
            unpack_state_device_y(state, varSet, lowFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbLow);
        else
            unpack_state_device_z(state, varSet, lowFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbLow);
    }
    if (!mpi_is_physical_boundary(h->mpi, highFace)) {
        if (direction == 0)
            unpack_state_device_x(state, varSet, highFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbHigh);
        else if (direction == 1)
            unpack_state_device_y(state, varSet, highFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbHigh);
        else
            unpack_state_device_z(state, varSet, highFace, h->nx, h->ny, h->nz,
                                  h->ngx, h->ngy, h->ngz, nGhost, nf, rbHigh);
    }
}

/* ---------------------------------------------------------------------------
   Device pack/unpack for scalar fields (used by IGR sigma halo)
   --------------------------------------------------------------------------- */

static void pack_scalar_device_x(double* field, int face,
                                 int nx, int ny, int nz,
                                 int ngx, int ngy, int ngz,
                                 int nGhost, double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iOffset = (face == XLOW) ? ngx : (ngx + nx - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int g = 0; g < nGhost; ++g) {
                size_t cellIdx = (size_t)((iOffset + g) +
                                          nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t linear = (size_t)(g + nGhost * (j + (size_t)ny * k));
                buf[linear] = field[cellIdx];
            }
        }
    }
}

static void unpack_scalar_device_x(double* field, int face,
                                   int nx, int ny, int nz,
                                   int ngx, int ngy, int ngz,
                                   int nGhost, const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iOffset = (face == XLOW) ? 0 : (ngx + nx);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int g = 0; g < nGhost; ++g) {
                size_t cellIdx = (size_t)((iOffset + g) +
                                          nxTot * ((j + ngy) + nyTot * (k + ngz)));
                size_t linear = (size_t)(g + nGhost * (j + (size_t)ny * k));
                field[cellIdx] = buf[linear];
            }
        }
    }
}

static void pack_scalar_device_y(double* field, int face,
                                 int nx, int ny, int nz,
                                 int ngx, int ngy, int ngz,
                                 int nGhost, double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int iSpan = iHi - iLo;
    const int jOffset = (face == YLOW) ? ngy : (ngy + ny - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int g = 0; g < nGhost; ++g) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                size_t cellIdx = (size_t)((i + ngx) +
                                          nxTot * ((jOffset + g) + nyTot * (k + ngz)));
                size_t linear = (size_t)(ii + iSpan * (g + nGhost * (size_t)k));
                buf[linear] = field[cellIdx];
            }
        }
    }
}

static void unpack_scalar_device_y(double* field, int face,
                                   int nx, int ny, int nz,
                                   int ngx, int ngy, int ngz,
                                   int nGhost, const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int iSpan = iHi - iLo;
    const int jOffset = (face == YLOW) ? 0 : (ngy + ny);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 0; k < nz; ++k) {
        for (int g = 0; g < nGhost; ++g) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                size_t cellIdx = (size_t)((i + ngx) +
                                          nxTot * ((jOffset + g) + nyTot * (k + ngz)));
                size_t linear = (size_t)(ii + iSpan * (g + nGhost * (size_t)k));
                field[cellIdx] = buf[linear];
            }
        }
    }
}

static void pack_scalar_device_z(double* field, int face,
                                 int nx, int ny, int nz,
                                 int ngx, int ngy, int ngz,
                                 int nGhost, double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int jLo = -ngy, jHi = ny + ngy;
    const int iSpan = iHi - iLo;
    const int jSpan = jHi - jLo;
    const int kOffset = (face == ZLOW) ? ngz : (ngz + nz - nGhost);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int g = 0; g < nGhost; ++g) {
        for (int jj = 0; jj < jSpan; ++jj) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                int j = jLo + jj;
                size_t cellIdx = (size_t)((i + ngx) +
                                          nxTot * ((j + ngy) + nyTot * (kOffset + g)));
                size_t linear = (size_t)(ii + iSpan * (jj + (size_t)jSpan * g));
                buf[linear] = field[cellIdx];
            }
        }
    }
}

static void unpack_scalar_device_z(double* field, int face,
                                   int nx, int ny, int nz,
                                   int ngx, int ngy, int ngz,
                                   int nGhost, const double* buf) {
    const int nxTot = nx + 2 * ngx;
    const int nyTot = ny + 2 * ngy;
    const int iLo = -ngx, iHi = nx + ngx;
    const int jLo = -ngy, jHi = ny + ngy;
    const int iSpan = iHi - iLo;
    const int jSpan = jHi - jLo;
    const int kOffset = (face == ZLOW) ? 0 : (ngz + nz);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int g = 0; g < nGhost; ++g) {
        for (int jj = 0; jj < jSpan; ++jj) {
            for (int ii = 0; ii < iSpan; ++ii) {
                int i = iLo + ii;
                int j = jLo + jj;
                size_t cellIdx = (size_t)((i + ngx) +
                                          nxTot * ((j + ngy) + nyTot * (kOffset + g)));
                size_t linear = (size_t)(ii + iSpan * (jj + (size_t)jSpan * g));
                field[cellIdx] = buf[linear];
            }
        }
    }
}

void halo_exchange_scalar_direction_device(HaloExchange* h,
                                           double* field,
                                           int direction) {
    int lowFace  = direction * 2;
    int highFace = direction * 2 + 1;
    int nGhost = h->nGhost;

    /* Same self-rank / physical-boundary fast-path as the state variant. */
    if (!mpi_neighbor_is_remote(h->mpi, lowFace) &&
        !mpi_neighbor_is_remote(h->mpi, highFace)) {
        return;
    }

    size_t bufSize = slab_size(h, lowFace, 1);
    ensure_buf(&h->sendBuf[lowFace],  &h->sendBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->sendBuf[highFace], &h->sendBufCapacity[highFace], bufSize);
    ensure_buf(&h->recvBuf[lowFace],  &h->recvBufCapacity[lowFace],  bufSize);
    ensure_buf(&h->recvBuf[highFace], &h->recvBufCapacity[highFace], bufSize);

    double* sbLow  = h->sendBuf[lowFace];
    double* sbHigh = h->sendBuf[highFace];
    double* rbLow  = h->recvBuf[lowFace];
    double* rbHigh = h->recvBuf[highFace];

    if (direction == 0) {
        pack_scalar_device_x(field, lowFace,  h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbLow);
        pack_scalar_device_x(field, highFace, h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbHigh);
    } else if (direction == 1) {
        pack_scalar_device_y(field, lowFace,  h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbLow);
        pack_scalar_device_y(field, highFace, h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbHigh);
    } else {
        pack_scalar_device_z(field, lowFace,  h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbLow);
        pack_scalar_device_z(field, highFace, h->nx, h->ny, h->nz,
                             h->ngx, h->ngy, h->ngz, nGhost, sbHigh);
    }

    #pragma omp target update from(sbLow[0:bufSize])
    #pragma omp target update from(sbHigh[0:bufSize])

    do_exchange(h, lowFace, highFace, bufSize);

    #pragma omp target update to(rbLow[0:bufSize])
    #pragma omp target update to(rbHigh[0:bufSize])

    if (!mpi_is_physical_boundary(h->mpi, lowFace)) {
        if (direction == 0)
            unpack_scalar_device_x(field, lowFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbLow);
        else if (direction == 1)
            unpack_scalar_device_y(field, lowFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbLow);
        else
            unpack_scalar_device_z(field, lowFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbLow);
    }
    if (!mpi_is_physical_boundary(h->mpi, highFace)) {
        if (direction == 0)
            unpack_scalar_device_x(field, highFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbHigh);
        else if (direction == 1)
            unpack_scalar_device_y(field, highFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbHigh);
        else
            unpack_scalar_device_z(field, highFace, h->nx, h->ny, h->nz,
                                   h->ngx, h->ngy, h->ngz, nGhost, rbHigh);
    }
}
