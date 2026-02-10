#include "HaloExchange.hpp"
#include <algorithm>

namespace SemiImplicitFV {

HaloExchange::HaloExchange(const MPIContext& mpi, const RectilinearMesh& mesh)
    : mpi_(mpi)
    , mesh_(mesh)
    , nGhost_(mesh.nGhost())
    , nx_(mesh.nx())
    , ny_(mesh.ny())
    , nz_(mesh.nz())
    , ngx_(mesh.ngx())
    , ngy_(mesh.ngy())
    , ngz_(mesh.ngz())
{}

std::size_t HaloExchange::slabSize(int face, int nFields) const {
    // Slab dimensions: nGhost cells deep in the face-normal direction,
    // full extent (including previously exchanged ghosts) in tangential directions.
    std::size_t cells = 0;
    switch (face) {
        case MPIContext::XLow:
        case MPIContext::XHigh: {
            // X-face slab: nGhost x ny x nz (physical cells only in y,z at this stage)
            int jExtent = ny_;
            int kExtent = nz_;
            cells = static_cast<std::size_t>(nGhost_) * jExtent * kExtent;
            break;
        }
        case MPIContext::YLow:
        case MPIContext::YHigh: {
            // Y-face slab: (nx + 2*ngx) x nGhost x nz
            // After x-exchange, we include x-ghost cells in the slab
            int iExtent = nx_ + 2 * ngx_;
            int kExtent = nz_;
            cells = static_cast<std::size_t>(iExtent) * nGhost_ * kExtent;
            break;
        }
        case MPIContext::ZLow:
        case MPIContext::ZHigh: {
            // Z-face slab: (nx + 2*ngx) x (ny + 2*ngy) x nGhost
            int iExtent = nx_ + 2 * ngx_;
            int jExtent = ny_ + 2 * ngy_;
            cells = static_cast<std::size_t>(iExtent) * jExtent * nGhost_;
            break;
        }
    }
    return cells * static_cast<std::size_t>(nFields);
}

// ---------------------------------------------------------------------------
// Scalar pack/unpack
// ---------------------------------------------------------------------------

void HaloExchange::packScalar(const std::vector<double>& field, int face) {
    std::size_t pos = 0;
    auto& buf = sendBuf_[face];

    switch (face) {
        case MPIContext::XLow:
            // Send cells i in [0, nGhost) to left neighbor
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        buf[pos++] = field[mesh_.index(g, j, k)];
            break;
        case MPIContext::XHigh:
            // Send cells i in [nx-nGhost, nx) to right neighbor
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        buf[pos++] = field[mesh_.index(nx_ - nGhost_ + g, j, k)];
            break;
        case MPIContext::YLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_.index(i, g, k)];
            break;
        }
        case MPIContext::YHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_.index(i, ny_ - nGhost_ + g, k)];
            break;
        }
        case MPIContext::ZLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_.index(i, j, g)];
            break;
        }
        case MPIContext::ZHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        buf[pos++] = field[mesh_.index(i, j, nz_ - nGhost_ + g)];
            break;
        }
    }
}

void HaloExchange::unpackScalar(std::vector<double>& field, int face) {
    std::size_t pos = 0;
    auto& buf = recvBuf_[face];

    switch (face) {
        case MPIContext::XLow:
            // Receive into ghost cells i in [-nGhost, 0)
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        field[mesh_.index(-nGhost_ + g, j, k)] = buf[pos++];
            break;
        case MPIContext::XHigh:
            // Receive into ghost cells i in [nx, nx+nGhost)
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        field[mesh_.index(nx_ + g, j, k)] = buf[pos++];
            break;
        case MPIContext::YLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_.index(i, -nGhost_ + g, k)] = buf[pos++];
            break;
        }
        case MPIContext::YHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_.index(i, ny_ + g, k)] = buf[pos++];
            break;
        }
        case MPIContext::ZLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_.index(i, j, -nGhost_ + g)] = buf[pos++];
            break;
        }
        case MPIContext::ZHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        field[mesh_.index(i, j, nz_ + g)] = buf[pos++];
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// State pack/unpack (multi-field)
// ---------------------------------------------------------------------------

// Helper: number of scalar fields for a given VarSet and dimension
static int numFields(VarSet varSet, int dim) {
    switch (varSet) {
        case VarSet::PRIM:
            // rho, velU, velV (if dim>=2), velW (if dim>=3), pres, temp, sigma
            return 4 + (dim >= 2 ? 1 : 0) + (dim >= 3 ? 1 : 0) + 1;
        case VarSet::CONS:
            // rho, rhoU, rhoV (if dim>=2), rhoW (if dim>=3), rhoE
            return 2 + (dim >= 2 ? 1 : 0) + (dim >= 3 ? 1 : 0) + 1;
        default:
            // All fields: conservative + primitive + sigma
            return 5 + (dim >= 2 ? 2 : 0) + (dim >= 3 ? 2 : 0) + 3 + 1;
    }
}

void HaloExchange::packState(SolutionState& state, VarSet varSet, int face) {
    // Pack all relevant fields interleaved per cell.
    // We iterate cells in the same order as scalar, but pack multiple values per cell.
    int dim = state.dim();
    std::size_t pos = 0;
    auto& buf = sendBuf_[face];

    // Macro for cell packing
    auto packCell = [&](std::size_t idx) {
        switch (varSet) {
            case VarSet::PRIM:
                buf[pos++] = state.rho[idx];
                buf[pos++] = state.velU[idx];
                if (dim >= 2) buf[pos++] = state.velV[idx];
                if (dim >= 3) buf[pos++] = state.velW[idx];
                buf[pos++] = state.pres[idx];
                buf[pos++] = state.temp[idx];
                buf[pos++] = state.sigma[idx];
                break;
            case VarSet::CONS:
                buf[pos++] = state.rho[idx];
                buf[pos++] = state.rhoU[idx];
                if (dim >= 2) buf[pos++] = state.rhoV[idx];
                if (dim >= 3) buf[pos++] = state.rhoW[idx];
                buf[pos++] = state.rhoE[idx];
                break;
            default:
                buf[pos++] = state.rho[idx];
                buf[pos++] = state.rhoU[idx];
                if (dim >= 2) buf[pos++] = state.rhoV[idx];
                if (dim >= 3) buf[pos++] = state.rhoW[idx];
                buf[pos++] = state.rhoE[idx];
                buf[pos++] = state.velU[idx];
                if (dim >= 2) buf[pos++] = state.velV[idx];
                if (dim >= 3) buf[pos++] = state.velW[idx];
                buf[pos++] = state.pres[idx];
                buf[pos++] = state.temp[idx];
                buf[pos++] = state.sigma[idx];
                break;
        }
    };

    switch (face) {
        case MPIContext::XLow:
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        packCell(mesh_.index(g, j, k));
            break;
        case MPIContext::XHigh:
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        packCell(mesh_.index(nx_ - nGhost_ + g, j, k));
            break;
        case MPIContext::YLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        packCell(mesh_.index(i, g, k));
            break;
        }
        case MPIContext::YHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        packCell(mesh_.index(i, ny_ - nGhost_ + g, k));
            break;
        }
        case MPIContext::ZLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        packCell(mesh_.index(i, j, g));
            break;
        }
        case MPIContext::ZHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        packCell(mesh_.index(i, j, nz_ - nGhost_ + g));
            break;
        }
    }
}

void HaloExchange::unpackState(SolutionState& state, VarSet varSet, int face) {
    int dim = state.dim();
    std::size_t pos = 0;
    auto& buf = recvBuf_[face];

    auto unpackCell = [&](std::size_t idx) {
        switch (varSet) {
            case VarSet::PRIM:
                state.rho[idx]   = buf[pos++];
                state.velU[idx]  = buf[pos++];
                if (dim >= 2) state.velV[idx] = buf[pos++];
                if (dim >= 3) state.velW[idx] = buf[pos++];
                state.pres[idx]  = buf[pos++];
                state.temp[idx]  = buf[pos++];
                state.sigma[idx] = buf[pos++];
                break;
            case VarSet::CONS:
                state.rho[idx]  = buf[pos++];
                state.rhoU[idx] = buf[pos++];
                if (dim >= 2) state.rhoV[idx] = buf[pos++];
                if (dim >= 3) state.rhoW[idx] = buf[pos++];
                state.rhoE[idx] = buf[pos++];
                break;
            default:
                state.rho[idx]   = buf[pos++];
                state.rhoU[idx]  = buf[pos++];
                if (dim >= 2) state.rhoV[idx] = buf[pos++];
                if (dim >= 3) state.rhoW[idx] = buf[pos++];
                state.rhoE[idx]  = buf[pos++];
                state.velU[idx]  = buf[pos++];
                if (dim >= 2) state.velV[idx] = buf[pos++];
                if (dim >= 3) state.velW[idx] = buf[pos++];
                state.pres[idx]  = buf[pos++];
                state.temp[idx]  = buf[pos++];
                state.sigma[idx] = buf[pos++];
                break;
        }
    };

    switch (face) {
        case MPIContext::XLow:
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        unpackCell(mesh_.index(-nGhost_ + g, j, k));
            break;
        case MPIContext::XHigh:
            for (int k = 0; k < nz_; ++k)
                for (int j = 0; j < ny_; ++j)
                    for (int g = 0; g < nGhost_; ++g)
                        unpackCell(mesh_.index(nx_ + g, j, k));
            break;
        case MPIContext::YLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        unpackCell(mesh_.index(i, -nGhost_ + g, k));
            break;
        }
        case MPIContext::YHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            for (int k = 0; k < nz_; ++k)
                for (int g = 0; g < nGhost_; ++g)
                    for (int i = iLo; i < iHi; ++i)
                        unpackCell(mesh_.index(i, ny_ + g, k));
            break;
        }
        case MPIContext::ZLow: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        unpackCell(mesh_.index(i, j, -nGhost_ + g));
            break;
        }
        case MPIContext::ZHigh: {
            int iLo = -ngx_, iHi = nx_ + ngx_;
            int jLo = -ngy_, jHi = ny_ + ngy_;
            for (int g = 0; g < nGhost_; ++g)
                for (int j = jLo; j < jHi; ++j)
                    for (int i = iLo; i < iHi; ++i)
                        unpackCell(mesh_.index(i, j, nz_ + g));
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// MPI exchange
// ---------------------------------------------------------------------------

void HaloExchange::doExchange(int lowFace, int highFace, std::size_t bufSize) {
    MPI_Request reqs[4];
    int nReqs = 0;

    int lowNeighbor  = mpi_.neighbor(lowFace);
    int highNeighbor = mpi_.neighbor(highFace);
    MPI_Comm comm = mpi_.comm();

    // Send to low, receive from high
    MPI_Isend(sendBuf_[lowFace].data(), static_cast<int>(bufSize), MPI_DOUBLE,
              lowNeighbor, 0, comm, &reqs[nReqs++]);
    MPI_Irecv(recvBuf_[highFace].data(), static_cast<int>(bufSize), MPI_DOUBLE,
              highNeighbor, 0, comm, &reqs[nReqs++]);

    // Send to high, receive from low
    MPI_Isend(sendBuf_[highFace].data(), static_cast<int>(bufSize), MPI_DOUBLE,
              highNeighbor, 1, comm, &reqs[nReqs++]);
    MPI_Irecv(recvBuf_[lowFace].data(), static_cast<int>(bufSize), MPI_DOUBLE,
              lowNeighbor, 1, comm, &reqs[nReqs++]);

    MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void HaloExchange::exchangeScalarDirection(std::vector<double>& field, int direction) {
    int lowFace  = direction * 2;       // XLow=0, YLow=2, ZLow=4
    int highFace = direction * 2 + 1;   // XHigh=1, YHigh=3, ZHigh=5

    std::size_t bufSize = slabSize(lowFace, 1);
    sendBuf_[lowFace].resize(bufSize);
    sendBuf_[highFace].resize(bufSize);
    recvBuf_[lowFace].resize(bufSize);
    recvBuf_[highFace].resize(bufSize);

    packScalar(field, lowFace);
    packScalar(field, highFace);

    doExchange(lowFace, highFace, bufSize);

    // Unpack only from neighbors that exist (MPI_PROC_NULL sends return empty)
    if (!mpi_.isPhysicalBoundary(lowFace))
        unpackScalar(field, lowFace);
    if (!mpi_.isPhysicalBoundary(highFace))
        unpackScalar(field, highFace);
}

void HaloExchange::exchangeScalar(std::vector<double>& field) {
    // Onion-peel: x, then y, then z
    exchangeScalarDirection(field, 0);
    if (mesh_.dim() >= 2) exchangeScalarDirection(field, 1);
    if (mesh_.dim() >= 3) exchangeScalarDirection(field, 2);
}

void HaloExchange::exchangeStateDirection(SolutionState& state, VarSet varSet, int direction) {
    int dim = state.dim();
    int nf = numFields(varSet, dim);

    int lowFace  = direction * 2;
    int highFace = direction * 2 + 1;

    std::size_t bufSize = slabSize(lowFace, nf);
    sendBuf_[lowFace].resize(bufSize);
    sendBuf_[highFace].resize(bufSize);
    recvBuf_[lowFace].resize(bufSize);
    recvBuf_[highFace].resize(bufSize);

    packState(state, varSet, lowFace);
    packState(state, varSet, highFace);

    doExchange(lowFace, highFace, bufSize);

    if (!mpi_.isPhysicalBoundary(lowFace))
        unpackState(state, varSet, lowFace);
    if (!mpi_.isPhysicalBoundary(highFace))
        unpackState(state, varSet, highFace);
}

void HaloExchange::exchangeState(SolutionState& state, VarSet varSet) {
    exchangeStateDirection(state, varSet, 0);
    if (mesh_.dim() >= 2) exchangeStateDirection(state, varSet, 1);
    if (mesh_.dim() >= 3) exchangeStateDirection(state, varSet, 2);
}

} // namespace SemiImplicitFV
