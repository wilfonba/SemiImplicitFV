#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

#ifdef ENABLE_MPI

#include "MPIContext.hpp"
#include "SolutionState.hpp"
#include "RectilinearMesh.hpp"
#include <vector>

namespace SemiImplicitFV {

/// Ghost cell communication for MPI domain decomposition.
///
/// Uses non-blocking MPI sends/receives with onion-peel ordering
/// (x, then y, then z) so edge/corner ghosts are filled correctly.
class HaloExchange {
public:
    HaloExchange(const MPIContext& mpi, const RectilinearMesh& mesh);

    /// Access the underlying MPI context.
    const MPIContext& mpi() const { return mpi_; }

    /// Exchange ghost cells for state variables in a given direction.
    /// direction: 0=x, 1=y, 2=z
    void exchangeStateDirection(SolutionState& state, VarSet varSet, int direction);

    /// Exchange all directions (x, then y, then z) for state variables.
    void exchangeState(SolutionState& state, VarSet varSet);

    /// Exchange ghost cells for a single scalar field in a given direction.
    void exchangeScalarDirection(std::vector<double>& field, int direction);

    /// Exchange all directions for a single scalar field.
    void exchangeScalar(std::vector<double>& field);

private:
    const MPIContext& mpi_;
    const RectilinearMesh& mesh_;

    // Per-face send/receive buffers
    std::vector<double> sendBuf_[6];
    std::vector<double> recvBuf_[6];

    // Number of fields packed per exchange (set dynamically)
    int nGhost_;
    int nx_, ny_, nz_;
    int ngx_, ngy_, ngz_;

    /// Compute buffer size for a face's halo slab (nFields * cells in slab).
    std::size_t slabSize(int face, int nFields) const;

    /// Pack state fields into send buffer for given face and direction.
    void packState(SolutionState& state, VarSet varSet, int face);
    /// Unpack recv buffer into state ghost cells for given face.
    void unpackState(SolutionState& state, VarSet varSet, int face);

    /// Pack a scalar field into send buffer for given face.
    void packScalar(const std::vector<double>& field, int face);
    /// Unpack recv buffer into scalar ghost cells for given face.
    void unpackScalar(std::vector<double>& field, int face);

    /// Perform the actual MPI exchange for two faces of a direction.
    void doExchange(int lowFace, int highFace, std::size_t bufSize);
};

} // namespace SemiImplicitFV

#endif // ENABLE_MPI
#endif // HALO_EXCHANGE_HPP
